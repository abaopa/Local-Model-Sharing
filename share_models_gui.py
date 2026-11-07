#!/usr/bin/env python3
"""
share_models_gui.py - Tkinter/ttk GUI for sharing LM Studio GGUF models
with Ollama and llama.cpp (llama-server / llama-cli).
"""
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path
from typing import List, Optional, Set

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

if (
    sys.platform == "darwin"
    and tk.TkVersion < 8.6
    and not os.environ.get("_SHARE_MODELS_REEXEC")
    and sys.argv
    and sys.argv[0] not in ("-c", "-m", "")
):
    for candidate in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            os.environ["_SHARE_MODELS_REEXEC"] = "1"
            os.execv(candidate, [candidate, *sys.argv])

from share_models import (
    DEFAULT_DIRS,
    Model,
    check_installations,
    check_ollama,
    ollama_existing,
    scan,
    select,
)


def auto_detect_dir() -> Optional[Path]:
    for d in DEFAULT_DIRS:
        if d.is_dir():
            return d
    return None


class ShareModelsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LM Studio Model Sharing — Ollama & llama.cpp")
        self.geometry("1120x760")
        self.minsize(900, 620)

        self.models: List[Model] = []
        self.filtered_models: List[Model] = []
        self.existing_ollama: Set[str] = set()
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.active_process: Optional[subprocess.Popen] = None
        self.worker_running = False

        self._configure_styles()
        self._build_ui()
        self.check_tools_status(log_details=True)

        initial_dir = auto_detect_dir()
        if initial_dir:
            self.dir_var.set(str(initial_dir))
            self.refresh_models()
        else:
            self.log(
                "Could not auto-detect LM Studio models folder.\n"
                "Please select your models folder using 'Browse...' above.\n"
            )

        self.after(100, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "aqua" in style.theme_names():
            style.theme_use("aqua")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        self.ui_font = ".AppleSystemUIFont" if sys.platform == "darwin" else "Segoe UI"
        self.mono_font = "Menlo" if sys.platform == "darwin" else "Consolas"
        self.font_size = 9

        for fn in (
            "TkDefaultFont",
            "TkTextFont",
            "TkHeadingFont",
            "TkMenuFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkIconFont",
            "TkTooltipFont",
        ):
            try:
                tkfont.nametofont(fn).configure(family=self.ui_font, size=self.font_size)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(family=self.mono_font, size=self.font_size)
        except tk.TclError:
            pass

        base_font = (self.ui_font, self.font_size)
        bold_font = (self.ui_font, self.font_size, "bold")
        mono_font = (self.mono_font, self.font_size)

        style.configure(".", font=base_font)
        style.configure("TLabel", font=base_font)
        style.configure("TButton", font=base_font)
        style.configure("TEntry", font=base_font)
        style.configure("TCheckbutton", font=base_font)
        style.configure("TRadiobutton", font=base_font)
        style.configure("TLabelframe.Label", font=bold_font)
        style.configure("TNotebook.Tab", font=base_font)
        style.configure("Treeview", rowheight=26, font=base_font)
        style.configure("Treeview.Heading", font=bold_font)
        style.configure("Header.TLabel", font=bold_font)
        style.configure("Status.TLabel", font=base_font)
        style.configure("ToolOk.TLabel", font=bold_font, foreground="#2da44e")
        style.configure("ToolMissing.TLabel", font=bold_font, foreground="#cf222e")
        style.configure("CmdPreview.TEntry", font=mono_font)

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Top Header: Environment & Tool Status ---
        env_frame = ttk.LabelFrame(main, text="Environment & Tool Status", padding=(10, 6))
        env_frame.pack(fill=tk.X, pady=(0, 8))

        self.lms_status_lbl = ttk.Label(env_frame, text="● LM Studio: Checking...", style="Status.TLabel")
        self.lms_status_lbl.pack(side=tk.LEFT, padx=(0, 24))

        self.ollama_status_lbl = ttk.Label(env_frame, text="● Ollama: Checking...", style="Status.TLabel")
        self.ollama_status_lbl.pack(side=tk.LEFT, padx=(0, 24))

        self.llama_status_lbl = ttk.Label(env_frame, text="● llama.cpp: Checking...", style="Status.TLabel")
        self.llama_status_lbl.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(
            env_frame,
            text="Check Installations",
            command=lambda: self.check_tools_status(log_details=True),
        ).pack(side=tk.RIGHT)

        # --- Models Directory & Search ---
        top_frame = ttk.LabelFrame(main, text="LM Studio Models Source", padding=8)
        top_frame.pack(fill=tk.X, pady=(0, 8))

        dir_row = ttk.Frame(top_frame)
        dir_row.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(dir_row, text="Models Folder:", width=14).pack(side=tk.LEFT)
        self.dir_var = tk.StringVar()
        dir_entry = ttk.Entry(dir_row, textvariable=self.dir_var, font=(self.ui_font, self.font_size))
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        dir_entry.bind("<Return>", lambda _e: self.refresh_models())

        ttk.Button(dir_row, text="Browse...", command=self._browse_models_dir).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(dir_row, text="Rescan", command=self.refresh_models).pack(side=tk.LEFT)

        filter_row = ttk.Frame(top_frame)
        filter_row.pack(fill=tk.X)

        ttk.Label(filter_row, text="Search Filter:", width=14).pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filter())
        search_entry = ttk.Entry(filter_row, textvariable=self.search_var, font=(self.ui_font, self.font_size))
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        ttk.Button(
            filter_row, text="Clear", command=lambda: self.search_var.set("")
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.summary_var = tk.StringVar(value="0 models")
        ttk.Label(filter_row, textvariable=self.summary_var, style="Status.TLabel").pack(
            side=tk.RIGHT
        )

        # --- PanedWindow: Upper Table + Actions, Lower Console ---
        paned = ttk.PanedWindow(main, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        upper_frame = ttk.Frame(paned)
        paned.add(upper_frame, weight=3)

        lower_frame = ttk.Frame(paned)
        paned.add(lower_frame, weight=2)

        # --- Models Treeview ---
        table_frame = ttk.Frame(upper_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        columns = ("publisher", "repo", "filename", "quant", "size", "ollama_name", "flags")
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="extended"
        )

        col_specs = [
            ("publisher", "Publisher", 125, tk.W),
            ("repo", "Repository", 200, tk.W),
            ("filename", "Model File / Folder", 250, tk.W),
            ("quant", "Quant", 85, tk.CENTER),
            ("size", "Size (GB)", 80, tk.E),
            ("ollama_name", "Ollama Name", 190, tk.W),
            ("flags", "Format / Status", 160, tk.CENTER),
        ]
        for col_id, heading, width, anchor in col_specs:
            self.tree.heading(
                col_id, text=heading, command=lambda c=col_id: self._sort_column(c, False)
            )
            self.tree.column(col_id, width=width, anchor=anchor)

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_selection_change())

        # --- Action Notebook (Ollama & llama.cpp) ---
        notebook = ttk.Notebook(upper_frame)
        notebook.pack(fill=tk.X)

        self._build_ollama_tab(notebook)
        self._build_llama_tab(notebook)

        # --- Log Console ---
        log_header = ttk.Frame(lower_frame)
        log_header.pack(fill=tk.X, pady=(4, 2))

        ttk.Label(log_header, text="Output & Process Log", style="Header.TLabel").pack(
            side=tk.LEFT
        )
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(log_header, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.LEFT, padx=(12, 0)
        )

        ttk.Button(log_header, text="Clear Log", command=self._clear_log).pack(
            side=tk.RIGHT
        )
        ttk.Button(
            log_header, text="Reveal Selected in Explorer", command=self._reveal_in_explorer
        ).pack(side=tk.RIGHT, padx=(0, 6))

        log_container = ttk.Frame(lower_frame)
        log_container.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_container,
            height=10,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#ffffff",
            font=(self.mono_font, self.font_size),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        log_vsb = ttk.Scrollbar(
            log_container, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=log_vsb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_ollama_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="  Ollama Import  ")

        row1 = ttk.Frame(tab)
        row1.pack(fill=tk.X)

        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1,
            text="Dry Run (preview without copying files)",
            variable=self.dry_run_var,
        ).pack(side=tk.LEFT, padx=(0, 16))

        self.import_selected_btn = ttk.Button(
            row1,
            text="Import Selected Models to Ollama",
            command=lambda: self._run_ollama_import(selected_only=True),
        )
        self.import_selected_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.import_all_btn = ttk.Button(
            row1,
            text="Import All Filtered Models",
            command=lambda: self._run_ollama_import(selected_only=False),
        )
        self.import_all_btn.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            row1,
            text="Check Ollama Installed Models",
            command=self._check_ollama_status,
        ).pack(side=tk.RIGHT)

        info_lbl = ttk.Label(
            tab,
            text="Tip: Select one or more models in the table above (Ctrl/Shift+Click) to import into Ollama as 'lms/<repo>:<quant>'.",
            foreground="#555555",
        )
        info_lbl.pack(anchor=tk.W, pady=(6, 0))

    def _build_llama_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="  llama.cpp (Server & CLI)  ")

        # Row 1: Mode, Binary override, Quick parameters
        row1 = ttk.Frame(tab)
        row1.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(row1, text="Mode:").pack(side=tk.LEFT)
        self.llama_mode_var = tk.StringVar(value="serve")
        self.llama_mode_var.trace_add("write", lambda *_: self._on_llama_mode_change())

        ttk.Radiobutton(
            row1, text="llama-server", value="serve", variable=self.llama_mode_var
        ).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Radiobutton(
            row1, text="llama-cli", value="cli", variable=self.llama_mode_var
        ).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(row1, text="Binary (--bin):").pack(side=tk.LEFT)
        self.bin_var = tk.StringVar()
        self.bin_var.trace_add("write", lambda *_: self._update_cmd_preview())
        ttk.Entry(row1, textvariable=self.bin_var, width=26, font=(self.ui_font, self.font_size)).pack(
            side=tk.LEFT, padx=(4, 4)
        )
        ttk.Button(row1, text="Browse...", command=self._browse_binary).pack(
            side=tk.LEFT, padx=(0, 14)
        )

        ttk.Label(row1, text="Extra Args:").pack(side=tk.LEFT)
        self._default_server_args = "-c 8192 -ngl 99 --port 8080"
        self._default_cli_args = "-c 8192 -ngl 99"
        self.extra_args_var = tk.StringVar(value=self._default_server_args)
        self.extra_args_var.trace_add("write", lambda *_: self._update_cmd_preview())
        ttk.Entry(row1, textvariable=self.extra_args_var, font=(self.ui_font, self.font_size)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        # Row 2: Command preview & Run buttons
        row2 = ttk.Frame(tab)
        row2.pack(fill=tk.X)

        ttk.Label(row2, text="Command:").pack(side=tk.LEFT)
        self.cmd_preview_var = tk.StringVar()
        cmd_entry = ttk.Entry(
            row2,
            textvariable=self.cmd_preview_var,
            state="readonly",
            style="CmdPreview.TEntry",
            font=(self.mono_font, self.font_size),
        )
        cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 8))

        self.run_llama_btn = ttk.Button(
            row2, text="Launch", command=self._launch_llama
        )
        self.run_llama_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.stop_llama_btn = ttk.Button(
            row2, text="Stop Process", command=self._stop_active_process, state=tk.DISABLED
        )
        self.stop_llama_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.open_browser_btn = ttk.Button(
            row2, text="Open Web UI", command=self._open_server_url
        )
        self.open_browser_btn.pack(side=tk.LEFT)

    # --- Helpers & Event Handlers ---

    def log(self, text: str) -> None:
        self.log_queue.put(text)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _browse_models_dir(self) -> None:
        initial = self.dir_var.get() or str(Path.home())
        chosen = filedialog.askdirectory(
            title="Select LM Studio Models Directory", initialdir=initial
        )
        if chosen:
            self.dir_var.set(chosen)
            self.refresh_models()

    def _browse_binary(self) -> None:
        filetypes = [("Executable", "*.exe"), ("All files", "*.*")] if os.name == "nt" else [("All files", "*.*")]
        chosen = filedialog.askopenfilename(
            title="Select llama.cpp Binary", filetypes=filetypes
        )
        if chosen:
            self.bin_var.set(chosen)
            name = Path(chosen).name.lower()
            if "llama-cli" in name:
                self.llama_mode_var.set("cli")
            elif "llama-server" in name:
                self.llama_mode_var.set("serve")

    def _on_llama_mode_change(self) -> None:
        mode = self.llama_mode_var.get()
        cur = self.extra_args_var.get().strip()
        if mode == "cli" and cur == self._default_server_args:
            self.extra_args_var.set(self._default_cli_args)
        elif mode == "serve" and cur == self._default_cli_args:
            self.extra_args_var.set(self._default_server_args)
        self._update_cmd_preview()

    @staticmethod
    def _short_version(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        for prefix in ("ollama version is ", "version: "):
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix):].strip()
        short = raw.split()[0].strip("(),")
        return f"v{short}" if short and not short.startswith("v") else short

    def check_tools_status(self, log_details: bool = False) -> None:
        custom_bin = self.bin_var.get().strip() or None
        statuses = check_installations(custom_bin)
        if log_details:
            self.log("--- Environment & Tool Check ---\n")

        badge_map = {
            "lm_studio": self.lms_status_lbl,
            "ollama": self.ollama_status_lbl,
            "llama_cpp": self.llama_status_lbl,
        }
        for key, st in statuses.items():
            mark = "Installed" if st.installed else "Not Found"
            lbl = badge_map.get(key)
            if lbl is not None:
                if st.installed:
                    v_str = self._short_version(st.version)
                    extras = [v_str] if v_str else []
                    if key == "ollama" and st.items is not None:
                        extras.append(f"{len(st.items)} models")
                    suffix = f" ({' · '.join(extras)})" if extras else ""
                    lbl.configure(
                        text=f"● {st.name}: Installed{suffix}",
                        style="ToolOk.TLabel",
                    )
                else:
                    lbl.configure(
                        text=f"○ {st.name}: Not Found",
                        style="ToolMissing.TLabel",
                    )

            if log_details:
                ver = f" ({st.version})" if st.version else ""
                loc = f" -> {st.path}" if st.path else ""
                self.log(f"[{mark.upper():9}] {st.name}{ver}{loc}\n")
                if st.details:
                    self.log(f"            {st.details}\n")
                if st.items:
                    for item in st.items:
                        self.log(f"              - {item}\n")

    def _try_fetch_ollama_existing(self) -> Set[str]:
        if not shutil.which("ollama"):
            return set()
        try:
            out = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return set()
        names = set()
        for line in out.splitlines()[1:]:
            if line.strip():
                n = line.split()[0]
                names.add(n)
                if n.endswith(":latest"):
                    names.add(n[: -len(":latest")])
        return names

    def refresh_models(self) -> None:
        raw_dir = self.dir_var.get().strip()
        if not raw_dir:
            return
        root = Path(raw_dir).expanduser()
        if not root.is_dir():
            messagebox.showerror(
                "Directory Not Found", f"Models directory does not exist:\n{root}"
            )
            return

        self.existing_ollama = self._try_fetch_ollama_existing()
        self.models = scan(root)
        self._apply_filter()
        gguf_n = sum(1 for m in self.models if m.fmt == "GGUF")
        other_n = len(self.models) - gguf_n
        self.log(
            f"Scanned {root}: found {len(self.models)} model(s) "
            f"({gguf_n} GGUF, {other_n} MLX/Safetensors).\n"
        )

    def _apply_filter(self) -> None:
        query_str = self.search_var.get().strip()
        queries = query_str.split() if query_str else []
        self.filtered_models = select(self.models, queries)

        self.tree.delete(*self.tree.get_children())
        total_gb = 0.0
        for idx, m in enumerate(self.filtered_models):
            total_gb += m.size_gb
            flags = [m.fmt]
            if m.downloading:
                flags.append("downloading")
            if m.mmproj:
                flags.append("vision")
            if m.split:
                flags.append("split")
            if m.ollama_name in self.existing_ollama:
                flags.append("in Ollama")
            self.tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    m.publisher,
                    m.repo,
                    m.path.name,
                    m.quant,
                    f"{m.size_gb:.1f}",
                    m.ollama_name if m.fmt == "GGUF" else "—",
                    ", ".join(flags),
                ),
            )

        self.summary_var.set(
            f"Showing {len(self.filtered_models)} of {len(self.models)} models ({total_gb:.1f} GB)"
        )
        if self.filtered_models:
            first_iid = self.tree.get_children()[0]
            self.tree.selection_set(first_iid)
        self._update_cmd_preview()

    def _sort_column(self, col: str, reverse: bool) -> None:
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        if col == "size":
            items.sort(key=lambda t: float(t[0]), reverse=reverse)
        else:
            items.sort(key=lambda t: t[0].lower(), reverse=reverse)
        for index, (_val, k) in enumerate(items):
            self.tree.move(k, "", index)
        self.tree.heading(col, command=lambda: self._sort_column(col, not reverse))

    def _get_selected_models(self) -> List[Model]:
        sel = self.tree.selection()
        return [self.filtered_models[int(iid)] for iid in sel if int(iid) < len(self.filtered_models)]

    def _on_selection_change(self) -> None:
        self._update_cmd_preview()

    def _build_llama_command(self, model: Model) -> List[str]:
        mode = self.llama_mode_var.get()
        binary = "llama-server" if mode == "serve" else "llama-cli"
        custom_bin = self.bin_var.get().strip()
        if custom_bin:
            p = Path(custom_bin).expanduser()
            if p.is_dir():
                candidate = p / (f"{binary}.exe" if os.name == "nt" else binary)
                exe = str(candidate) if candidate.exists() else custom_bin
            elif (
                (mode == "serve" and "llama-cli" in p.name.lower())
                or (mode == "cli" and "llama-server" in p.name.lower())
            ):
                sibling = p.with_name(f"{binary}.exe" if os.name == "nt" else binary)
                exe = str(sibling) if sibling.exists() else custom_bin
            else:
                exe = custom_bin
        else:
            exe = shutil.which(binary) or binary

        is_cli = mode == "cli" or "llama-cli" in Path(exe).name.lower()
        cmd = [exe, "-m", str(model.path)]
        if model.mmproj:
            cmd += ["--mmproj", str(model.mmproj)]
        extra_raw = self.extra_args_var.get().strip()
        if extra_raw:
            try:
                extra = shlex.split(extra_raw, posix=(os.name != "nt"))
            except ValueError:
                extra = extra_raw.split()
            cleaned = [a.strip('"') for a in extra if a != "--"]
            if is_cli:
                filtered: List[str] = []
                skip_next = False
                for arg in cleaned:
                    if skip_next:
                        skip_next = False
                        continue
                    if arg in ("--port", "--host"):
                        skip_next = True
                        continue
                    if arg.startswith("--port=") or arg.startswith("--host="):
                        continue
                    filtered.append(arg)
                cleaned = filtered
            cmd += cleaned
        return cmd

    def _update_cmd_preview(self) -> None:
        selected = self._get_selected_models()
        if not selected:
            self.cmd_preview_var.set("(Select a model above to preview command)")
            return
        m = selected[0]
        if m.downloading:
            self.cmd_preview_var.set("(Download still in progress — wait for .part file to finish)")
            return
        if m.fmt != "GGUF":
            self.cmd_preview_var.set(f"({m.fmt} safetensors format — llama.cpp requires a GGUF model)")
            return
        cmd = self._build_llama_command(m)
        formatted = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self.cmd_preview_var.set(formatted)

    def _reveal_in_explorer(self) -> None:
        selected = self._get_selected_models()
        if not selected:
            messagebox.showinfo("No Selection", "Please select a model first.")
            return
        target = selected[0].path
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(target)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent)])

    # --- Ollama Operations ---

    def _check_ollama_status(self) -> None:
        st = check_ollama()
        if not st.installed:
            messagebox.showwarning(
                "Ollama Not Found", "'ollama' was not found on your system."
            )
            return
        self.existing_ollama = self._try_fetch_ollama_existing()
        self._apply_filter()
        ver = f" ({st.version})" if st.version else ""
        self.log(f"\n--- Ollama Installed Models{ver} ---\n")
        if st.items:
            for item in st.items:
                self.log(f"  - {item}\n")
        else:
            self.log(f"  ({st.details or 'No models found'})\n")

    def _run_ollama_import(self, selected_only: bool) -> None:
        if self.worker_running:
            messagebox.showinfo(
                "Busy", "An operation is already running. Please wait for it to finish."
            )
            return

        chosen = self._get_selected_models() if selected_only else list(self.filtered_models)
        if not chosen:
            messagebox.showinfo("No Models", "No models selected to import.")
            return

        if not shutil.which("ollama"):
            messagebox.showerror(
                "Ollama Not Found", "'ollama' was not found on your system PATH."
            )
            return

        dry_run = self.dry_run_var.get()
        self.worker_running = True
        self.status_var.set("Running Ollama import...")
        threading.Thread(
            target=self._ollama_worker, args=(chosen, dry_run), daemon=True
        ).start()

    def _ollama_worker(self, chosen: List[Model], dry_run: bool) -> None:
        try:
            self.log("\n--- Starting Ollama Import ---\n")
            try:
                existing = ollama_existing()
            except SystemExit as e:
                self.log(f"ERROR: {e}\n")
                return

            total = 0.0
            for m in chosen:
                if m.downloading:
                    self.log(f"skip   {m.label}: download still in progress (.part)\n")
                    continue
                if m.fmt != "GGUF":
                    self.log(
                        f"skip   {m.label}: {m.fmt} (safetensors) format cannot be imported as GGUF\n"
                    )
                    continue
                name = m.ollama_name
                if name in existing:
                    self.log(f"skip   {name} (already in Ollama)\n")
                    continue
                if m.split:
                    self.log(
                        f"skip   {m.label}: split GGUFs must be merged first (llama-gguf-split --merge)\n"
                    )
                    continue
                if m.mmproj:
                    self.log(
                        f"note   {m.label}: vision projector is not imported; text-only in Ollama\n"
                    )
                self.log(
                    f"create {name}  <-  {m.path}  ({m.size_gb:.1f} GB copied into Ollama's store)\n"
                )
                total += m.size_gb
                if dry_run:
                    continue

                with tempfile.TemporaryDirectory() as tmp:
                    mf = Path(tmp) / "Modelfile"
                    mf.write_text(f"FROM {m.path}\n", encoding="utf-8")
                    proc = subprocess.Popen(
                        ["ollama", "create", name, "-f", str(mf)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.log(f"  {line}")
                    proc.wait()
                    if proc.returncode != 0:
                        self.log(f"FAILED {name} (exit code {proc.returncode})\n")
                    else:
                        existing.add(name)

            action = "Would copy" if dry_run else "Copied"
            self.log(f"\n{action} about {total:.1f} GB into Ollama.\n")
        finally:
            self.worker_running = False
            self.after(0, self._on_ollama_done)

    def _on_ollama_done(self) -> None:
        self.status_var.set("Idle")
        self.existing_ollama = self._try_fetch_ollama_existing()
        self._apply_filter()

    # --- llama.cpp Operations ---

    def _launch_llama(self) -> None:
        selected = self._get_selected_models()
        if not selected:
            messagebox.showinfo("Select a Model", "Please select a model from the table.")
            return
        if len(selected) > 1:
            messagebox.showinfo(
                "Multiple Models Selected",
                f"Using the first selected model:\n{selected[0].label}",
            )

        model = selected[0]
        if model.downloading:
            messagebox.showwarning(
                "Download in Progress",
                f"'{model.repo}' is still downloading ({model.path.name}).\nWait for LM Studio to finish downloading it first.",
            )
            return
        if model.fmt != "GGUF":
            messagebox.showwarning(
                "Unsupported Format for llama.cpp",
                f"'{model.repo}' is in {model.fmt} (safetensors) format.\n\nllama.cpp requires GGUF (.gguf) models.",
            )
            return
        cmd = self._build_llama_command(model)
        mode = self.llama_mode_var.get()
        formatted_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self.log(f"\n$ {formatted_cmd}\n")

        if mode == "cli":
            # Launch interactive llama-cli in a new console window so user can chat interactively
            try:
                if os.name == "nt":
                    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
                elif sys.platform == "darwin":
                    shell_cmd = " ".join(shlex.quote(c) for c in cmd)
                    apple_script = (
                        'tell application "Terminal"\n'
                        "    activate\n"
                        f'    do script "{shell_cmd.replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34))}"\n'
                        "end tell"
                    )
                    subprocess.Popen(["osascript", "-e", apple_script])
                else:
                    subprocess.Popen(cmd)
                self.log("Launched llama-cli in external console window.\n")
            except FileNotFoundError:
                messagebox.showerror(
                    "Binary Not Found",
                    f"'{cmd[0]}' not found.\nPut llama.cpp on PATH or specify Binary (--bin).",
                )
            return

        # Mode == "serve": run embedded with live log streaming
        if self.active_process and self.active_process.poll() is None:
            messagebox.showwarning(
                "Server Running",
                "A llama-server process is already running. Stop it first.",
            )
            return

        try:
            self.active_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except FileNotFoundError:
            messagebox.showerror(
                "Binary Not Found",
                f"'{cmd[0]}' not found.\nPut llama.cpp on PATH or specify Binary (--bin).",
            )
            return

        self.status_var.set(f"Running llama-server ({model.path.name})...")
        self.run_llama_btn.configure(state=tk.DISABLED)
        self.stop_llama_btn.configure(state=tk.NORMAL)

        threading.Thread(
            target=self._stream_process_output,
            args=(self.active_process,),
            daemon=True,
        ).start()

    def _stream_process_output(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log(line)
            proc.wait()
            self.log(f"\n[Process exited with code {proc.returncode}]\n")
        finally:
            self.after(0, self._on_process_stopped)

    def _stop_active_process(self) -> None:
        if self.active_process and self.active_process.poll() is None:
            self.log("\nStopping active llama-server process...\n")
            self.active_process.terminate()

    def _on_process_stopped(self) -> None:
        self.status_var.set("Idle")
        self.run_llama_btn.configure(state=tk.NORMAL)
        self.stop_llama_btn.configure(state=tk.DISABLED)

    def _open_server_url(self) -> None:
        extra = self.extra_args_var.get()
        port = "8080"
        parts = extra.split()
        for i, p in enumerate(parts):
            if p == "--port" and i + 1 < len(parts):
                port = parts[i + 1]
                break
        url = f"http://127.0.0.1:{port}"
        self.log(f"Opening {url} in browser...\n")
        webbrowser.open(url)

    def _on_close(self) -> None:
        if self.active_process and self.active_process.poll() is None:
            if messagebox.askyesno(
                "Quit",
                "llama-server is still running. Terminate it and exit?",
            ):
                self.active_process.terminate()
            else:
                return
        self.destroy()


def main() -> None:
    app = ShareModelsApp()
    app.mainloop()


if __name__ == "__main__":
    main()

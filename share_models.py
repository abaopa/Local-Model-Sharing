#!/usr/bin/env python3
"""
share_models.py - use LM Studio's downloaded GGUF models in Ollama and llama.cpp.

Commands:
  check                     Check whether LM Studio, Ollama, and llama.cpp are installed
  list                      Show models found in LM Studio's folder
  ollama [query ...]        Import models into Ollama (ollama create)
  serve  <query> [-- args]  Run llama-server on a model (extra args after --)
  cli    <query> [-- args]  Run llama-cli on a model (extra args after --)

Examples:
  python share_models.py check
  python share_models.py list
  python share_models.py ollama --dry-run
  python share_models.py ollama llama-3 qwen
  python share_models.py serve qwen2.5 -- -c 8192 -ngl 99 --port 8080
"""
import argparse
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DIRS = [
    Path.home() / ".lmstudio" / "models",
    Path.home() / ".cache" / "lm-studio" / "models",  # older LM Studio versions
]
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.(?:gguf|safetensors)$", re.I)
QUANT_RE = re.compile(
    r"((?:IQ|Q)\d(?:_[A-Z0-9]+)*|MXFP\d(?:[-_][A-Z0-9]+)*|\d+bit|BF16|F16|F32|FP16|FP8)",
    re.I,
)


@dataclass
class Model:
    path: Path
    publisher: str
    repo: str
    quant: str
    mmproj: Optional[Path]
    split: bool
    fmt: str = "GGUF"
    downloading: bool = False
    total_bytes: Optional[int] = None

    @property
    def label(self) -> str:
        return f"{self.publisher}/{self.repo}/{self.path.name}"

    @property
    def size_gb(self) -> float:
        b = self.total_bytes if self.total_bytes is not None else self.path.stat().st_size
        return b / 1024**3

    @property
    def ollama_name(self) -> str:
        def clean(s: str) -> str:
            return re.sub(r"[^a-z0-9._-]+", "-", s.lower()).strip("-")

        repo = re.sub(r"[-_]?(?:gguf|mlx(?:[-_]\d+bit)?)$", "", self.repo, flags=re.I)
        return f"lms/{clean(repo)}:{clean(self.quant)}"


def find_root(arg: Optional[str]) -> Path:
    if arg:
        root = Path(arg).expanduser()
        if root.is_dir():
            return root
        sys.exit(f"Models directory not found: {root}")
    for d in DEFAULT_DIRS:
        if d.is_dir():
            return d
    sys.exit(
        "Could not find LM Studio's models folder. Pass it with --models-dir "
        "(see LM Studio > My Models > Models Directory)."
    )


def scan(root: Path) -> List[Model]:
    models: List[Model] = []
    seen_dirs_gguf = set()

    # 1. Completed GGUF models
    for p in sorted(root.rglob("*.gguf")):
        name = p.name
        if "mmproj" in name.lower():
            continue
        shard = SHARD_RE.search(name)
        if shard and shard.group(1) != "00001":
            continue  # only list the first shard of a split model
        parts = p.relative_to(root).parts
        if len(parts) >= 3:
            publisher, repo = parts[0], parts[1]
        else:
            publisher, repo = "local", p.parent.name
        q = QUANT_RE.search(name) or QUANT_RE.search(repo)
        mm = next(
            (f for f in sorted(p.parent.glob("*.gguf")) if "mmproj" in f.name.lower()),
            None,
        )
        if shard:
            all_shards = [
                f for f in p.parent.glob("*.gguf") if "mmproj" not in f.name.lower()
            ]
            total_b = sum(f.stat().st_size for f in all_shards)
        else:
            total_b = p.stat().st_size
        seen_dirs_gguf.add(p.parent)
        models.append(
            Model(
                path=p,
                publisher=publisher,
                repo=repo,
                quant=q.group(1).upper() if q else "latest",
                mmproj=mm,
                split=bool(shard),
                fmt="GGUF",
                downloading=False,
                total_bytes=total_b,
            )
        )

    # 2. In-progress GGUF downloads (*.gguf.part)
    for p in sorted(root.rglob("*.gguf.part")):
        if p.parent in seen_dirs_gguf:
            continue
        name = p.name
        if "mmproj" in name.lower():
            continue
        parts = p.relative_to(root).parts
        if len(parts) >= 3:
            publisher, repo = parts[0], parts[1]
        else:
            publisher, repo = "local", p.parent.name
        q = QUANT_RE.search(name) or QUANT_RE.search(repo)
        mm = next(
            (f for f in sorted(p.parent.glob("*.gguf")) if "mmproj" in f.name.lower()),
            None,
        )
        models.append(
            Model(
                path=p,
                publisher=publisher,
                repo=repo,
                quant=q.group(1).upper() if q else "latest",
                mmproj=mm,
                split=False,
                fmt="GGUF",
                downloading=True,
                total_bytes=p.stat().st_size,
            )
        )

    # 3. MLX / Safetensors models (*.safetensors)
    st_by_dir: Dict[Path, List[Path]] = {}
    for p in sorted(root.rglob("*.safetensors")):
        st_by_dir.setdefault(p.parent, []).append(p)

    for model_dir, st_files in sorted(st_by_dir.items()):
        if model_dir in seen_dirs_gguf:
            continue
        first_file = st_files[0]
        parts = first_file.relative_to(root).parts
        if len(parts) >= 3:
            publisher, repo = parts[0], parts[1]
        else:
            publisher, repo = "local", model_dir.name
        q = QUANT_RE.search(repo) or QUANT_RE.search(first_file.name)
        is_mlx = "mlx" in publisher.lower() or "mlx" in repo.lower()
        total_b = sum(f.stat().st_size for f in st_files)
        models.append(
            Model(
                path=model_dir if len(st_files) > 1 else first_file,
                publisher=publisher,
                repo=repo,
                quant=q.group(1).upper() if q else "safetensors",
                mmproj=None,
                split=len(st_files) > 1,
                fmt="MLX" if is_mlx else "Safetensors",
                downloading=any(model_dir.glob("*.part")),
                total_bytes=total_b,
            )
        )

    models.sort(key=lambda m: (m.publisher.lower(), m.repo.lower(), m.path.name.lower()))
    return models


def select(models: List[Model], queries: List[str]) -> List[Model]:
    if not queries:
        return models
    qs = [q.lower() for q in queries]
    return [m for m in models if any(q in m.label.lower() for q in qs)]


def pick_one(models: List[Model], query: str) -> Model:
    matches = select(models, [query])
    if not matches:
        sys.exit(f"No model matches '{query}'. Run the 'list' command to see what's available.")
    if len(matches) > 1:
        print(f"'{query}' matches {len(matches)} models; be more specific:", file=sys.stderr)
        for m in matches:
            print(f"  {m.label}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def cmd_list(models: List[Model], _args) -> None:
    if not models:
        print("No models found.")
        return
    for i, m in enumerate(models, 1):
        flags = [m.fmt]
        if m.downloading:
            flags.append("downloading")
        if m.mmproj:
            flags.append("vision")
        if m.split:
            flags.append("split")
        extra = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{i:>3}. {m.label}  ({m.size_gb:.1f} GB, {m.quant}){extra}")


def ollama_existing() -> set:
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"Could not run 'ollama list' (is Ollama installed and running?): {e}")
    names = set()
    for line in out.splitlines()[1:]:
        if line.strip():
            n = line.split()[0]
            names.add(n)
            if n.endswith(":latest"):
                names.add(n[: -len(":latest")])
    return names


def cmd_ollama(models: List[Model], args) -> None:
    chosen = select(models, args.query)
    if not chosen:
        sys.exit("No matching models.")
    if not shutil.which("ollama"):
        sys.exit("'ollama' was not found on PATH.")
    existing = ollama_existing()
    total = 0.0
    for m in chosen:
        if m.downloading:
            print(f"skip   {m.label}: download still in progress (.part)")
            continue
        if m.fmt != "GGUF":
            print(f"skip   {m.label}: {m.fmt} (safetensors) format cannot be imported as GGUF")
            continue
        name = m.ollama_name
        if name in existing:
            print(f"skip   {name} (already in Ollama)")
            continue
        if m.split:
            print(f"skip   {m.label}: split GGUFs must be merged first (llama-gguf-split --merge)")
            continue
        if m.mmproj:
            print(f"note   {m.label}: vision projector is not imported; text-only in Ollama")
        print(f"create {name}  <-  {m.path}  ({m.size_gb:.1f} GB copied into Ollama's store)")
        total += m.size_gb
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            mf = Path(tmp) / "Modelfile"
            mf.write_text(f"FROM {m.path}\n", encoding="utf-8")
            r = subprocess.run(["ollama", "create", name, "-f", str(mf)])
            if r.returncode != 0:
                print(f"FAILED {name}", file=sys.stderr)
    print(f"\n{'Would copy' if args.dry_run else 'Copied'} about {total:.1f} GB into Ollama.")


def run_llama(binary: str, models: List[Model], args) -> None:
    m = pick_one(models, args.query)
    if m.downloading:
        sys.exit(f"Cannot run '{m.label}': download is still in progress (.part).")
    if m.fmt != "GGUF":
        sys.exit(f"Cannot run '{m.label}' with llama.cpp: {m.fmt} (safetensors) requires MLX, not llama.cpp.")
    exe = args.bin or shutil.which(binary) or binary
    cmd = [exe, "-m", str(m.path)]
    if m.mmproj:
        cmd += ["--mmproj", str(m.mmproj)]
    cmd += [a for a in args.extra if a != "--"]
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    try:
        sys.exit(subprocess.call(cmd))
    except FileNotFoundError:
        sys.exit(f"'{exe}' not found. Put llama.cpp on PATH or pass --bin /path/to/{binary}")
    except KeyboardInterrupt:
        sys.exit(130)


@dataclass
class ToolStatus:
    name: str
    installed: bool
    path: Optional[str] = None
    version: Optional[str] = None
    details: Optional[str] = None
    items: Optional[List[str]] = None


def _first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _run_version_cmd(cmd: List[str]) -> Optional[str]:
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        out = (res.stdout or res.stderr or "").strip()
        for line in out.splitlines():
            line = line.strip()
            if line:
                return line
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def check_lm_studio() -> ToolStatus:
    home = Path.home()
    cli_candidates = [
        Path(p) for p in [shutil.which("lms")] if p
    ] + [
        home / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms"),
    ]
    cli_path = _first_existing_path(cli_candidates)

    app_candidates: List[Path] = []
    if sys.platform == "darwin":
        app_candidates = [
            Path("/Applications/LM Studio.app"),
            home / "Applications" / "LM Studio.app",
        ]
    elif os.name == "nt":
        local_app = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
        prog_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        app_candidates = [
            local_app / "Programs" / "LM Studio" / "LM Studio.exe",
            prog_files / "LM Studio" / "LM Studio.exe",
        ]
    else:
        which_app = shutil.which("lm-studio")
        if which_app:
            app_candidates.append(Path(which_app))

    app_path = _first_existing_path(app_candidates)
    models_dir = _first_existing_path(DEFAULT_DIRS)

    version: Optional[str] = None
    if sys.platform == "darwin" and app_path and (app_path / "Contents" / "Info.plist").is_file():
        try:
            with (app_path / "Contents" / "Info.plist").open("rb") as f:
                plist = plistlib.load(f)
                version = plist.get("CFBundleShortVersionString")
        except Exception:
            pass

    installed = bool(app_path or cli_path or models_dir)
    primary_path = str(app_path or cli_path or models_dir) if installed else None
    details_parts = []
    if cli_path:
        details_parts.append(f"CLI: {cli_path}")
    if models_dir:
        details_parts.append(f"Models: {models_dir}")
    return ToolStatus(
        name="LM Studio",
        installed=installed,
        path=primary_path,
        version=version,
        details=", ".join(details_parts) if details_parts else None,
    )


def check_ollama() -> ToolStatus:
    home = Path.home()
    candidates: List[Path] = []
    which_ollama = shutil.which("ollama")
    if which_ollama:
        candidates.append(Path(which_ollama))
    if sys.platform == "darwin":
        candidates += [
            Path("/opt/homebrew/bin/ollama"),
            Path("/usr/local/bin/ollama"),
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
        ]
    elif os.name == "nt":
        local_app = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
        candidates.append(local_app / "Programs" / "Ollama" / "ollama.exe")

    exe_path = _first_existing_path(candidates)
    if not exe_path:
        return ToolStatus(name="Ollama", installed=False)

    ver_line = _run_version_cmd([str(exe_path), "--version"])
    version = ver_line
    installed_models: List[str] = []
    details: Optional[str] = None
    try:
        res = subprocess.run(
            [str(exe_path), "list"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        for ln in lines[1:]:
            parts = re.split(r"\s{2,}", ln)
            if len(parts) >= 3:
                installed_models.append(f"{parts[0]} ({parts[2]})")
            elif parts:
                installed_models.append(parts[0])
        details = f"{len(installed_models)} model(s) installed"
    except (OSError, subprocess.SubprocessError):
        details = "Installed (server not running)"

    return ToolStatus(
        name="Ollama",
        installed=True,
        path=str(exe_path),
        version=version,
        details=details,
        items=installed_models,
    )


def check_llama_cpp(custom_bin: Optional[str] = None) -> ToolStatus:
    server_exe: Optional[Path] = None
    cli_exe: Optional[Path] = None

    if custom_bin:
        p = Path(custom_bin).expanduser()
        if p.is_dir():
            s = p / ("llama-server.exe" if os.name == "nt" else "llama-server")
            c = p / ("llama-cli.exe" if os.name == "nt" else "llama-cli")
            if s.exists():
                server_exe = s
            if c.exists():
                cli_exe = c
        elif p.exists():
            if "llama-cli" in p.name.lower():
                cli_exe = p
                sib = p.with_name("llama-server.exe" if os.name == "nt" else "llama-server")
                if sib.exists():
                    server_exe = sib
            else:
                server_exe = p
                sib = p.with_name("llama-cli.exe" if os.name == "nt" else "llama-cli")
                if sib.exists():
                    cli_exe = sib

    extra_dirs = [Path("/opt/homebrew/bin"), Path("/usr/local/bin")] if sys.platform == "darwin" else []
    if not server_exe:
        w = shutil.which("llama-server")
        server_exe = Path(w) if w else _first_existing_path([d / "llama-server" for d in extra_dirs])
    if not cli_exe:
        w = shutil.which("llama-cli")
        cli_exe = Path(w) if w else _first_existing_path([d / "llama-cli" for d in extra_dirs])

    # Check LM Studio's bundled llama.cpp backend as fallback
    bundled_server: Optional[Path] = None
    backends_dir = Path.home() / ".lmstudio" / "extensions" / "backends"
    if not server_exe and backends_dir.is_dir():
        found = sorted(backends_dir.rglob("llama-server.exe" if os.name == "nt" else "llama-server"))
        if found:
            bundled_server = found[-1]

    primary = server_exe or cli_exe or bundled_server
    if not primary:
        return ToolStatus(name="llama.cpp", installed=False)

    version = _run_version_cmd([str(primary), "--version"])
    parts = []
    if server_exe:
        parts.append(f"llama-server: {server_exe}")
    if cli_exe:
        parts.append(f"llama-cli: {cli_exe}")
    if not server_exe and not cli_exe and bundled_server:
        parts.append(f"LM Studio bundled llama-server: {bundled_server}")

    return ToolStatus(
        name="llama.cpp",
        installed=True,
        path=str(primary),
        version=version,
        details=", ".join(parts) if parts else None,
    )


def check_installations(custom_llama_bin: Optional[str] = None) -> Dict[str, ToolStatus]:
    return {
        "lm_studio": check_lm_studio(),
        "ollama": check_ollama(),
        "llama_cpp": check_llama_cpp(custom_llama_bin),
    }


def cmd_check(args) -> None:
    statuses = check_installations(getattr(args, "bin", None))
    for status in statuses.values():
        badge = "INSTALLED" if status.installed else "MISSING  "
        ver = f" ({status.version})" if status.version else ""
        loc = f" -> {status.path}" if status.path else ""
        print(f"[{badge}] {status.name}{ver}{loc}")
        if status.details:
            print(f"            {status.details}")
        if status.items:
            for item in status.items:
                print(f"              - {item}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", help="LM Studio models folder (auto-detected by default)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="check whether LM Studio, Ollama, and llama.cpp are installed")
    p_check.add_argument("--bin", help="optional path to llama.cpp binary or folder")

    sub.add_parser("list", help="list LM Studio models")

    p = sub.add_parser("ollama", help="import models into Ollama")
    p.add_argument("query", nargs="*", help="substring filters (default: all models)")
    p.add_argument("--dry-run", action="store_true", help="show what would be imported")

    for name in ("serve", "cli"):
        p = sub.add_parser(name, help=f"run llama-{'server' if name == 'serve' else 'cli'}")
        p.add_argument("query", help="substring identifying the model")
        p.add_argument("--bin", help="path to the llama.cpp binary")
        p.add_argument("extra", nargs=argparse.REMAINDER, help="extra args after --")

    args = ap.parse_args()
    if args.cmd == "check":
        cmd_check(args)
        return

    models = scan(find_root(args.models_dir))

    if args.cmd == "list":
        cmd_list(models, args)
    elif args.cmd == "ollama":
        cmd_ollama(models, args)
    elif args.cmd == "serve":
        run_llama("llama-server", models, args)
    elif args.cmd == "cli":
        run_llama("llama-cli", models, args)


if __name__ == "__main__":
    main()

# Local-Model — Share LM Studio Models with Ollama & llama.cpp

Zero-dependency Python CLI and Tkinter/ttk GUI utilities to discover local models downloaded by **LM Studio** (across all publisher folders) and share/run **GGUF** models directly in **Ollama** and **llama.cpp** (`llama-server` / `llama-cli`).

---

## Features

- **Environment & Installation Check**:
  - Detects installed versions and paths of **LM Studio**, **Ollama** (including all currently installed Ollama models and sizes), and **llama.cpp** (`llama-server` / `llama-cli`).
  - Displayed in a color-coded **Environment & Tool Status** bar at the top of the GUI window and via `python3 share_models.py check`.
- **Comprehensive Model Discovery**:
  - Scans LM Studio's default model directories (`~/.lmstudio/models` and `~/.cache/lm-studio/models`) or any custom folder across all publisher subdirectories (e.g., `lmstudio-community`, `mlx-community`, etc.).
  - Lists **completed GGUF models** (`*.gguf`), **Apple MLX / Safetensors models** (`*.safetensors`), and **in-progress downloads** (`*.gguf.part`) with accurate total shard sizes.
- **Smart Metadata Detection**:
  - Extracts publisher, repository, quantization tag (`Q4_K_M`, `Q4_0`, `4BIT`, `MXFP4-Q8`, `BF16`, etc.), format (`GGUF`, `MLX`, `Safetensors`), and total size in GB.
  - Detects multimodal vision projectors (`mmproj*.gguf`) and automatically passes `--mmproj` when launching `llama.cpp`.
  - Groups multi-part split shards (`-00001-of-0000N.gguf` / `.safetensors`) into a single entry with combined file size.
- **Ollama Integration**:
  - Generates a temporary `Modelfile` (`FROM /path/to/model.gguf`) and imports GGUF models into Ollama as `lms/<repo>:<quant>`.
  - Detects models already registered in Ollama to skip redundant imports.
  - Supports `--dry-run` to preview imports and disk usage before copying.
- **llama.cpp Integration (`llama-server` & `llama-cli`)**:
  - Launch `llama-server` with live embedded log streaming and one-click Web UI (`http://127.0.0.1:8080`).
  - Launch `llama-cli` in an interactive terminal session (`Terminal.app` on macOS, new console on Windows).
  - Mode-aware default arguments and automatic sanitization of server-only flags (`--port`, `--host`) when using `llama-cli`.
- **Cross-Platform GUI (`share_models_gui.py`)**:
  - Top-level **Environment & Tool Status** badges, sortable model table, live search filter, command preview, and integrated process log console.
  - Automatic Homebrew Python/Tk handoff on macOS if launched via Apple's legacy system Python (Tk 8.5).

---

## Requirements

- **Python 3.8+** (standard library only — no `pip install` needed)
- **For GUI on macOS**: Python with modern Tcl/Tk 8.6+ (e.g., via Homebrew):
  ```bash
  brew install python-tk
  ```
- **Optional Runtime Tools**:
  - [LM Studio](https://lmstudio.ai/) (optional if pointing `--models-dir` at any folder of `.gguf` files)
  - [Ollama](https://ollama.com/) (`ollama` on `PATH`)
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) (`brew install llama.cpp` or `llama-server` / `llama-cli` on `PATH` or specified via `--bin`)

---

## Quick Start

### 1. Desktop GUI (`share_models_gui.py`)

```bash
python3 share_models_gui.py
```

1. **Environment & Tool Status**: View at-a-glance installation status, versions, and Ollama model count at the top of the window, or click **Check Installations** to log full details.
2. **Models Folder**: Automatically detects your LM Studio models folder (or click **Browse...** to pick another directory).
3. **Search & Select**: Filter models by name/publisher/quantization/format and click column headers to sort.
4. **Ollama Import Tab**:
   - Check **Dry Run** to preview without copying files.
   - Click **Import Selected Models to Ollama** or **Import All Filtered Models**.
   - Click **Check Ollama Installed Models** to list all models currently in Ollama.
5. **llama.cpp (Server & CLI) Tab**:
   - Choose **Mode**: `llama-server` (runs embedded with live output and **Open Web UI** button) or `llama-cli` (opens an interactive chat terminal).
   - Optionally specify a custom binary path in **Binary (`--bin`)** and customize **Extra Args**.
   - Click **Launch**.

---

### 2. Command-Line Interface (`share_models.py`)

#### Check if LM Studio, Ollama (with installed models), and `llama.cpp` are installed
```bash
python3 share_models.py check
```

#### List discovered models in LM Studio's folder
```bash
python3 share_models.py list
```

#### Import GGUF models into Ollama
```bash
# Preview what would be imported (no files copied)
python3 share_models.py ollama --dry-run

# Import specific models matching substring queries
python3 share_models.py ollama qwen gemma

# Import all discovered GGUF models
python3 share_models.py ollama
```

#### Serve a model with `llama-server`
```bash
python3 share_models.py serve qwen3.8 -- -c 8192 -ngl 99 --port 8080
```

#### Chat interactively with `llama-cli`
```bash
python3 share_models.py cli qwen3.8 -- -c 8192 -ngl 99
```

#### Custom models directory or custom `llama.cpp` binary
```bash
python3 share_models.py --models-dir /path/to/models serve qwen3.8 --bin /path/to/llama-server -- -c 8192
```

---

## Notes & Troubleshooting

- **GGUF vs. Apple MLX (`*.safetensors`) Compatibility**:
  Models from any publisher (`lmstudio-community`, `bartowski`, `unsloth`, etc.) in **GGUF (`*.gguf`)** format are 100% compatible with both **Ollama** and **llama.cpp**. Models downloaded in **Apple MLX (`*-MLX-4bit`, `*.safetensors`)** format are displayed in the model list for visibility, but use Apple's MLX-specific quantization and cannot be loaded directly by `llama.cpp` or `ollama create`.
- **macOS `DEPRECATION WARNING: The system version of Tk is deprecated` / Blank Window**:
  macOS's Xcode Command Line Tools Python (`/usr/bin/python3`) links against Apple's legacy Tk 8.5, which fails to render windows on recent macOS releases. Install Homebrew's Tk support (`brew install python-tk`); `share_models_gui.py` automatically detects legacy Tk and re-executes itself with `/opt/homebrew/bin/python3`.
- **Disk Space with Ollama**:
  `ollama create` copies the GGUF weights into Ollama's blob store (`~/.ollama/models`). Use `llama-server` or `llama-cli` if you want to run models directly from LM Studio's folder with zero duplication.
- **Split GGUFs**:
  Multi-shard GGUF files (`*-00001-of-0000N.gguf`) work directly with `llama.cpp`, but must be merged with `llama-gguf-split --merge` before importing into Ollama.

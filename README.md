# Nates3DRenderer (Block Pipeline)

A small block-based pipeline runner with a PyQt5 GUI.  
You compose a pipeline as `stage1|stage2|stage3`, pass a payload, and optionally override parameters via “extras”.

This repo is structured so **gui.py is the entrypoint** (including when frozen via PyInstaller).

## Features
- PyQt5 GUI to run pipelines and manage presets
- Block registry system (`registry.py`)
- Simple pipeline executor (`pipeline.py`)
- Builder-style pipeline block (`builder_blocks.py`)
- Presets saved to a per-user app data directory (safe for PyInstaller builds)

## Project Layout
- `gui.py` — GUI entrypoint (run this)
- `main.py` — optional CLI runner (falls back to GUI if no args)
- `registry.py` — block registry (BLOCKS)
- `block.py` — BaseBlock + BuilderBlock base
- `pipeline.py` — PipelineRunner + PipelineBlock
- `builder_blocks.py` — SimpleBuilder (builds a plan from params)
- `blocks.py` — registers blocks (identity/pretty/etc + pipeline + builder)
- `utils.py` — app_dir(), json helpers, extras parsing

## Install (dev)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

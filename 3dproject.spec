# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

spec_dir = Path(globals().get("SPEC", os.getcwd())).resolve()
project_dir = spec_dir

hiddenimports = []
hiddenimports += collect_submodules("PyQt5")

datas = []
datas += collect_data_files("PyQt5", include_py_files=False)

try:
    import PIL  # noqa
    hiddenimports += collect_submodules("PIL")
    datas += collect_data_files("PIL", include_py_files=False)
except Exception:
    pass

try:
    import requests  # noqa
    hiddenimports += collect_submodules("requests")
except Exception:
    pass

block_cipher = None

a = Analysis(
    ["gui.py"],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="Nates3DRenderer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    onefile=True,          # ✅ single exe
)

# ✅ no COLLECT() in onefile builds

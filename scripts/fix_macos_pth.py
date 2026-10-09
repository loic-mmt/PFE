"""Keep the macOS environment outside a dot-directory and repair hidden .pth files.

The .venv symlink preserves uv's default path, activation scripts and shebangs.
The original environment is renamed, never deleted or reinstalled.
"""

import os
import stat
import sys
from pathlib import Path

if sys.platform == "darwin":
    project = Path(__file__).resolve().parents[1]
    default = project / ".venv"
    destination = project / "venv"
    if default.is_dir() and not default.is_symlink():
        if destination.exists() or destination.is_symlink():
            raise SystemExit("Cannot migrate .venv: venv already exists")
        default.rename(destination)
        default.symlink_to(destination.name, target_is_directory=True)
    root = default.resolve()
    # Directories renamed from .venv can retain inherited UF_HIDDEN flags.
    # Clear these too so subsequent editable installs inherit visible parents.
    directories = [root, *(p for p in root.rglob("*") if p.is_dir() and not p.is_symlink())]
    for path in directories:
        if not path.exists():
            continue
        flags = path.stat().st_flags
        if flags & stat.UF_HIDDEN:
            os.chflags(path, flags & ~stat.UF_HIDDEN)
    for path in root.glob("lib/python*/site-packages/*.pth"):
        flags = path.stat().st_flags
        if flags & stat.UF_HIDDEN:
            os.chflags(path, flags & ~stat.UF_HIDDEN)

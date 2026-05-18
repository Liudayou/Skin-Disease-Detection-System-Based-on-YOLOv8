"""Resolve ``backbones.*`` from ``<repo>/scripts/backbones/``.

PyPI distributes an unrelated top-level package also named ``backbones``; if it is
installed, ``import backbones`` can pick that copy and break
``from backbones.mobilenetv3_backbone import ...``.

Call :func:`ensure_local_backbones` once (after ``scripts/`` is on ``sys.path``)
before importing project backbone modules.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path


def ensure_local_backbones(scripts_root: Path) -> None:
    """Register ``backbones`` as a package whose only search path is ``scripts_root/backbones``."""
    bb = (scripts_root / "backbones").resolve()
    if not bb.is_dir():
        raise FileNotFoundError(f"missing local backbones directory: {bb}")
    bb_dir = str(bb)

    # Drop PyPI `backbones` (or anything shadowing) from the import cache.
    for k in list(sys.modules):
        if k != "backbones" and not k.startswith("backbones."):
            continue
        mod = sys.modules[k]
        fp = getattr(mod, "__file__", "") or ""
        if fp and "site-packages" in fp.replace("\\", "/"):
            del sys.modules[k]

    existing = sys.modules.get("backbones")
    if existing is not None:
        locs = getattr(existing, "__path__", None)
        if locs and any(str(Path(p).resolve()) == bb_dir for p in locs):
            return
        del sys.modules["backbones"]

    spec = importlib.machinery.ModuleSpec(
        name="backbones",
        loader=None,
        is_package=True,
    )
    spec.submodule_search_locations = [bb_dir]
    pkg = types.ModuleType("backbones")
    pkg.__spec__ = spec
    pkg.__path__ = [bb_dir]  # type: ignore[attr-defined]
    sys.modules["backbones"] = pkg

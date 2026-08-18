"""
Auto-patch MIE-YOLO custom modules into the installed ultralytics nn/tasks.py.

Detects the installed ultralytics package, applies the four required edits
(module loader, base_modules, repeat_modules, detect-heads frozenset) and
verifies by building the MIE-YOLO model. Safe to re-run (idempotent).

Usage (inside the activated virtual env):
    python patch_mie.py
"""

import importlib
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

MARK_LOADER = "mie_extra_block"


def build_loader_block():
    block_path = (PROJECT_DIR / "ultralytics" / "nn" / "extra_modules" / "block.py").as_posix()
    head_path = (PROJECT_DIR / "ultralytics" / "nn" / "extra_modules" / "head.py").as_posix()
    return f"""# === MIE-YOLO custom modules (auto-patched by patch_mie.py) ===
import importlib.util as _mie_ilu
import sys as _mie_sys


def _mie_load(name, path):
    _spec = _mie_ilu.spec_from_file_location(name, path)
    _mod = _mie_ilu.module_from_spec(_spec)
    _mie_sys.modules[name] = _mod
    _spec.loader.exec_module(_mod)
    return _mod


_mie_block = _mie_load("mie_extra_block", r"{block_path}")
_mie_head = _mie_load("mie_extra_head", r"{head_path}")

C3k2_MutilScaleEdgeInformationSelect = _mie_block.C3k2_MutilScaleEdgeInformationSelect
C3k2_AdditiveBlock_CGLU = _mie_block.C3k2_AdditiveBlock_CGLU
Detect_LSDECD = _mie_head.Detect_LSDECD
# === end MIE-YOLO custom modules ===


"""


def apply_edit(text, anchor, replacement, label, already_marker=None):
    if already_marker and already_marker in text:
        print(f"[skip ] {label}: already applied")
        return text, True
    if anchor not in text:
        print(f"[FAIL ] {label}: anchor not found, manual patch required")
        return text, False
    count = text.count(anchor)
    if count > 1:
        print(f"[FAIL ] {label}: anchor is ambiguous ({count} occurrences)")
        return text, False
    return text.replace(anchor, replacement, 1), True


def main():
    try:
        import ultralytics
    except ImportError:
        print("[FAIL ] ultralytics not installed. Run: pip install 'ultralytics==8.3.253'")
        sys.exit(1)

    print(f"ultralytics version: {ultralytics.__version__}")
    if not ultralytics.__version__.startswith("8.3"):
        print("[WARN ] untested ultralytics version; 8.3.x is required for MIE-YOLO compatibility")

    tasks_path = Path(importlib.import_module("ultralytics.nn.tasks").__file__)
    if not tasks_path.is_file():
        print(f"[FAIL ] cannot locate tasks.py at: {tasks_path}")
        sys.exit(1)
    print(f"tasks.py: {tasks_path}")

    text = tasks_path.read_text(encoding="utf-8")
    ok = True

    text, s = apply_edit(
        text,
        "class BaseModel(",
        build_loader_block() + "class BaseModel(",
        "1/4 module loader",
        already_marker=MARK_LOADER,
    )
    ok &= s

    text, s = apply_edit(
        text,
        "            C3k2,\n            RepNCSPELAN4,",
        "            C3k2,\n            C3k2_MutilScaleEdgeInformationSelect,\n            C3k2_AdditiveBlock_CGLU,\n            RepNCSPELAN4,",
        "2/4 base_modules",
        already_marker="            C3k2_MutilScaleEdgeInformationSelect,\n            C3k2_AdditiveBlock_CGLU,\n            RepNCSPELAN4,",
    )
    ok &= s

    text, s = apply_edit(
        text,
        "            C3k2,\n            C2fAttn,",
        "            C3k2,\n            C3k2_MutilScaleEdgeInformationSelect,\n            C3k2_AdditiveBlock_CGLU,\n            C2fAttn,",
        "3/4 repeat_modules",
        already_marker="            C3k2_MutilScaleEdgeInformationSelect,\n            C3k2_AdditiveBlock_CGLU,\n            C2fAttn,",
    )
    ok &= s

    text, s = apply_edit(
        text,
        "elif m in frozenset(\n            {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}\n        ):",
        "elif m in frozenset(\n            {\n                Detect,\n                Detect_LSDECD,\n                WorldDetect,\n                YOLOEDetect,\n                Segment,\n                YOLOESegment,\n                Pose,\n                OBB,\n                ImagePoolingAttn,\n                v10Detect,\n            }\n        ):",
        "4/4 detect-heads frozenset",
        already_marker="                Detect_LSDECD,",
    )
    ok &= s

    if not ok:
        print("[RESULT] patch incomplete, fix the failed steps manually (see REPRODUCE.md 3.2)")
        sys.exit(1)

    backup = tasks_path.with_suffix(".py.mie_bak")
    if not backup.exists():
        backup.write_text(tasks_path.read_text(encoding="utf-8"), encoding="utf-8")
    tasks_path.write_text(text, encoding="utf-8")
    print("[OK    ] tasks.py patched (backup saved to %s)" % backup.name)

    import subprocess
    code = ('import sys; sys.path.insert(0, r"%s"); '
            'from ultralytics import YOLO; '
            'm = YOLO("MIE-YOLO.yaml"); '
            'print("[TEST  ] MIE-YOLO build OK:", m.info())') % PROJECT_DIR.as_posix()
    res = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_DIR, capture_output=True, text=True)
    if res.returncode == 0:
        for line in res.stdout.strip().splitlines()[-3:]:
            print(line)
        print("[RESULT] patch verified: model builds successfully")
    else:
        print("[RESULT] patch applied but build test failed:")
        print(res.stderr[-2000:])


if __name__ == "__main__":
    main()
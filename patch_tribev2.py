"""
Apply CPU-compatibility patches to the installed tribev2 package.

Run this after any `pip install tribev2` or upgrade:
    python patch_tribev2.py

Patches applied:
  - eventstransforms.py: use float32 compute_type on CPU (float16 is CUDA-only in
    faster-whisper / ctranslate2 and crashes on Mac without an NVIDIA GPU).
"""

import importlib.util
import sys
from pathlib import Path


def find_package_file(package: str, relative: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        sys.exit(
            f"Package '{package}' not found in this Python ({sys.executable}).\n"
            f"Install it first, then re-run this script, e.g.\n"
            f"  {sys.executable} -m pip install tribev2"
        )
    pkg_dir = Path(spec.origin).parent
    target = pkg_dir / relative
    if not target.exists():
        sys.exit(f"Target file not found: {target}")
    return target


def patch_eventstransforms(path: Path) -> None:
    original = '        compute_type = "float16"'
    patched = (
        '        # float16 is invalid on CPU (faster-whisper / ctranslate2); Mac has no CUDA.\n'
        '        compute_type = "float16" if device == "cuda" else "float32"'
    )
    text = path.read_text()
    if 'compute_type = "float16" if device == "cuda" else "float32"' in text:
        print(f"[skip] {path.name} already patched.")
        return
    if original not in text:
        print(f"[warn] Expected pattern not found in {path.name}; skipping.")
        return
    path.write_text(text.replace(original, patched))
    print(f"[ok]   Patched {path}")


def main() -> None:
    et = find_package_file("tribev2", "eventstransforms.py")
    patch_eventstransforms(et)
    print("Done.")


if __name__ == "__main__":
    main()

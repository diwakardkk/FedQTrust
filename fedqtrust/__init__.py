"""Import shim so `python -m fedqtrust` works from a source checkout."""

from pathlib import Path

_src_pkg = Path(__file__).resolve().parent.parent / "src" / "fedqtrust"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))  # type: ignore[name-defined]


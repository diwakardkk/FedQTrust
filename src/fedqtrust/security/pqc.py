"""liboqs wrappers with strict availability checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys


def oqs_available() -> bool:
    return importlib.util.find_spec("oqs") is not None


def enabled_mechanisms() -> dict[str, list[str]]:
    code = (
        "import json, oqs; "
        "print(json.dumps({'kem': list(getattr(oqs, 'get_enabled_kem_mechanisms', lambda: [])()), "
        "'sig': list(getattr(oqs, 'get_enabled_sig_mechanisms', lambda: [])())}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {"kem": [], "sig": []}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"kem": [], "sig": []}

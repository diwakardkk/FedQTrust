#!/usr/bin/env python3
"""Single experiment entry point in the source tree and the upload package."""
from pathlib import Path
import sys

source = Path(__file__).resolve().parent / 'src'
if source.is_dir():
    sys.path.insert(0, str(source))

from fedqtrust.manuscript.cli import main

if __name__ == '__main__':
    raise SystemExit(main())

"""Make `common` importable for the artifact test suite.

Mirrors the bootstrap every `artifact/` entry script performs: put the artifact
root (computed from this file's location, never a hardcoded relative string) on
`sys.path` so `import common.*` resolves regardless of the working directory
pytest was invoked from.
"""

import sys
from pathlib import Path

_ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(_ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARTIFACT_ROOT))

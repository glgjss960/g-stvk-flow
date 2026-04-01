from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_flow_matching_path() -> Path | None:
    """
    Make local `flow_matching/` importable without requiring pip install.

    Expected repository layout:
      K-Flow/
        flow_matching/
          flow_matching/
        g-stvk-flow/
          g_stvk_flow/
    """
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "flow_matching"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return candidate
    return None


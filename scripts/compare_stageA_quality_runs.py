from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def _resolve_metrics_path(p: Path) -> Path:
    if p.is_dir():
        return p / "quality_metrics.jsonl"
    return p


def _index_by_key(rows: list[dict[str, Any]], key: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        if key not in r:
            continue
        out[int(r[key])] = r
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v)


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare StageA quality metrics for two runs")
    parser.add_argument("--run-a", type=Path, required=True, help="run dir or quality_metrics.jsonl path")
    parser.add_argument("--run-b", type=Path, required=True, help="run dir or quality_metrics.jsonl path")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--align-key", type=str, default="epoch", choices=["epoch"])
    args = parser.parse_args()

    path_a = _resolve_metrics_path(args.run_a)
    path_b = _resolve_metrics_path(args.run_b)

    rows_a = _load_jsonl(path_a)
    rows_b = _load_jsonl(path_b)

    idx_a = _index_by_key(rows_a, args.align_key)
    idx_b = _index_by_key(rows_b, args.align_key)

    common = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    if not common:
        raise RuntimeError(f"No overlapping {args.align_key} between {path_a} and {path_b}")

    metrics = ["fvd_proxy_r3d18", "gen_tlpips", "gen_warping_error"]

    print(f"Comparing {len(common)} aligned points by {args.align_key}")
    for key in common:
        ra = idx_a[key]
        rb = idx_b[key]
        print(f"\n{args.align_key}={key}")
        for m in metrics:
            av = ra.get(m, None)
            bv = rb.get(m, None)
            dv = _delta(av, bv)
            print(f"  {m:20s} {args.label_a}={_fmt(av):>12}  {args.label_b}={_fmt(bv):>12}  delta(B-A)={_fmt(dv):>12}")


if __name__ == "__main__":
    main()

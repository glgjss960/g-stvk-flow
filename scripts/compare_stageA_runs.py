from __future__ import annotations

import argparse
import csv
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
        return p / "metrics.jsonl"
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
        fv = float(v)
        return f"{fv:.6f}"
    except Exception:
        return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two StageA runs by aligned training step/epoch")
    parser.add_argument("--run-a", type=Path, required=True, help="run dir or metrics.jsonl path")
    parser.add_argument("--run-b", type=Path, required=True, help="run dir or metrics.jsonl path")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--align-key", type=str, default="global_step", choices=["global_step", "epoch"])
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    path_a = _resolve_metrics_path(args.run_a)
    path_b = _resolve_metrics_path(args.run_b)

    rows_a = _load_jsonl(path_a)
    rows_b = _load_jsonl(path_b)
    idx_a = _index_by_key(rows_a, args.align_key)
    idx_b = _index_by_key(rows_b, args.align_key)

    common_keys = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    if not common_keys:
        raise RuntimeError(f"No overlapping {args.align_key} between {path_a} and {path_b}")

    band_cols = [
        "val_band_ls_lt",
        "val_band_ls_ht",
        "val_band_hs_lt",
        "val_band_hs_ht",
    ]

    print(f"Comparing {len(common_keys)} aligned points by {args.align_key}")
    print(
        f"{args.align_key:>10}  "
        f"{args.label_a}_val_loss{'':>8}  {args.label_b}_val_loss{'':>8}  delta(B-A)"
    )
    for k in common_keys:
        ra = idx_a[k]
        rb = idx_b[k]
        a_val = ra.get("val_loss", None)
        b_val = rb.get("val_loss", None)
        delta = (float(b_val) - float(a_val)) if (a_val is not None and b_val is not None) else None
        print(f"{k:10d}  {_fmt(a_val):>16}  {_fmt(b_val):>16}  {_fmt(delta):>10}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                args.align_key,
                f"{args.label_a}_val_loss",
                f"{args.label_b}_val_loss",
                "delta_b_minus_a",
            ]
            for c in band_cols:
                header.extend([f"{args.label_a}_{c}", f"{args.label_b}_{c}", f"delta_{c}_b_minus_a"])
            writer.writerow(header)

            for k in common_keys:
                ra = idx_a[k]
                rb = idx_b[k]
                a_val = ra.get("val_loss", None)
                b_val = rb.get("val_loss", None)
                row = [
                    k,
                    a_val,
                    b_val,
                    (float(b_val) - float(a_val)) if (a_val is not None and b_val is not None) else None,
                ]
                for c in band_cols:
                    av = ra.get(c, None)
                    bv = rb.get(c, None)
                    row.extend([
                        av,
                        bv,
                        (float(bv) - float(av)) if (av is not None and bv is not None) else None,
                    ])
                writer.writerow(row)
        print(f"Saved comparison CSV to {args.out_csv}")


if __name__ == "__main__":
    main()

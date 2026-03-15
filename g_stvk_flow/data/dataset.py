from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class CachedVideoDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        lines = [ln.strip() for ln in self.manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.items: List[Dict[str, object]] = [json.loads(ln) for ln in lines]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.items[idx]
        payload = torch.load(rec["tensor_path"], map_location="cpu")

        if isinstance(payload, dict):
            video = payload["video"].float()
            label = int(payload.get("label", rec.get("label", 0)))
        else:
            video = payload.float()
            label = int(rec.get("label", 0))

        return {
            "video": video,  # [C,T,H,W]
            "label": torch.tensor(label, dtype=torch.long),
        }

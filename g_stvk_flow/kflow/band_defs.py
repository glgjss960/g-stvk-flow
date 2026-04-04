from __future__ import annotations

FOUR_BAND_NAMES: tuple[str, ...] = ("ls_lt", "ls_ht", "hs_lt", "hs_ht")
SPATIAL_ONLY_BANDS: tuple[str, ...] = ("ls", "hs")
TEMPORAL_ONLY_BANDS: tuple[str, ...] = ("lt", "ht")
VANILLA_BAND: tuple[str, ...] = ("full",)

PATH_A: tuple[str, ...] = ("ls_lt", "ls_ht", "hs_lt", "hs_ht")
PATH_B: tuple[str, ...] = ("ls_lt", "hs_lt", "ls_ht", "hs_ht")

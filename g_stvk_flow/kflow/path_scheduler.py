from __future__ import annotations

from dataclasses import dataclass

from .band_defs import PATH_A, PATH_B


@dataclass(frozen=True)
class FixedBandPath:
    """
    Fixed hand-crafted band generation order for phase-A.
    """

    band_names: tuple[str, ...]
    path_name: str = "A"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_name", str(self.path_name).upper())
        object.__setattr__(self, "band_names", tuple(self.band_names))

        if self.path_name not in {"A", "B"}:
            raise ValueError(f"Unknown path_name={self.path_name}, expected A or B")

        order = self.order
        if set(order) != set(self.band_names):
            raise ValueError(
                "Path and decomposer bands mismatch: "
                f"path={order}, bands={self.band_names}"
            )

    @property
    def order(self) -> tuple[str, ...]:
        if len(self.band_names) == 4 and set(self.band_names) == set(PATH_A):
            return PATH_A if self.path_name == "A" else PATH_B
        return self.band_names

    def index_of(self, band_name: str) -> int:
        return self.order.index(str(band_name))

    def previous_bands(self, band_name: str) -> tuple[str, ...]:
        idx = self.index_of(band_name)
        return self.order[:idx]

    def future_bands(self, band_name: str) -> tuple[str, ...]:
        idx = self.index_of(band_name)
        return self.order[idx + 1 :]

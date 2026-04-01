from __future__ import annotations

from dataclasses import dataclass

import torch

from g_stvk_flow.fm_core.bootstrap import bootstrap_flow_matching_path
from g_stvk_flow.gstvk import GSTVKInterpolant

bootstrap_flow_matching_path()

try:
    from flow_matching.path.path import ProbPath as _FMProbPath
    from flow_matching.path.path_sample import PathSample as _FMPathSample
except Exception:
    _FMProbPath = None
    _FMPathSample = None


class _FallbackProbPath:
    def assert_sample_shape(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> None:
        if t.ndim != 1:
            raise ValueError(f"time vector t must have shape [batch], got {tuple(t.shape)}")
        if x_0.shape[0] != x_1.shape[0] or x_0.shape[0] != t.shape[0]:
            raise ValueError(
                f"batch mismatch: x_0={x_0.shape[0]}, x_1={x_1.shape[0]}, t={t.shape[0]}"
            )


if _FMPathSample is None:
    @dataclass
    class _BasePathSample:
        x_1: torch.Tensor
        x_0: torch.Tensor
        t: torch.Tensor
        x_t: torch.Tensor
        dx_t: torch.Tensor
else:
    _BasePathSample = _FMPathSample


_BaseProbPath = _FMProbPath if _FMProbPath is not None else _FallbackProbPath


@dataclass
class GSTVKPathSample(_BasePathSample):
    """
    Path sample compatible with flow_matching conventions, plus g-STVK phase features.
    """

    phase_features: torch.Tensor


class GSTVKProbPath(_BaseProbPath):
    """
    g-STVK probability path:
      - x_1: target data video
      - x_0: source noise video
      - t:   time in [0,1]

    Output follows flow_matching's `(x_t, dx_t)` interface while preserving g-STVK
    phase features for the velocity network.
    """

    def __init__(self, interpolant: GSTVKInterpolant) -> None:
        self.interpolant = interpolant

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> GSTVKPathSample:
        if hasattr(self, "assert_sample_shape"):
            self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        out = self.interpolant.build(x_data=x_1, eps=x_0, tau=t)
        return GSTVKPathSample(
            x_1=x_1,
            x_0=x_0,
            t=t,
            x_t=out.psi_tau,
            dx_t=out.v_target,
            phase_features=out.phase_features,
        )


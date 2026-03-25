from typing import Tuple

import torch
from torch import Tensor

from models import EPS, dtype


class ClaSSEVectorTransportCache:
    """Transport ClaSSE partial vectors without materializing KxK propagators."""

    def __init__(self, time_grid: Tensor, E_vals: Tensor, B: Tensor, lam: Tensor):
        self.time_grid = time_grid
        self.B = B
        self.lam = lam
        self.dts = time_grid[1:] - time_grid[:-1]

        E_mid = (E_vals[:-1] + E_vals[1:]) / 2.0
        q_mid = E_mid @ B.transpose(0, 1)
        self.gamma = 2.0 * q_mid * lam.unsqueeze(0)

    def _rhs(self, V: Tensor, interval_idx: int) -> Tensor:
        gamma = self.gamma[interval_idx].unsqueeze(1)
        return -self.lam.unsqueeze(1) * V + gamma * (self.B @ V)

    def _rk4_step(self, V: Tensor, interval_idx: int) -> Tensor:
        dt = self.dts[interval_idx]
        k1 = self._rhs(V, interval_idx)
        k2 = self._rhs(V + 0.5 * dt * k1, interval_idx)
        k3 = self._rhs(V + 0.5 * dt * k2, interval_idx)
        k4 = self._rhs(V + dt * k3, interval_idx)
        return V + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _propagate_same_range(self, child_logs: Tensor, child_time_idx: int, parent_time_idx: int) -> Tensor:
        if child_time_idx == parent_time_idx:
            return child_logs
        if child_time_idx > parent_time_idx:
            raise ValueError("child_time_idx must be <= parent_time_idx")

        out = torch.full_like(child_logs, float("-inf"))
        finite_mask = torch.isfinite(child_logs).any(dim=1)
        if not finite_mask.any():
            return out

        work_logs = child_logs[finite_mask]
        log_scales = work_logs.max(dim=1).values
        V = torch.exp(work_logs - log_scales.unsqueeze(1)).transpose(0, 1)

        for interval_idx in range(child_time_idx, parent_time_idx):
            V = self._rk4_step(V, interval_idx).clamp_min(EPS)
            col_scales = V.max(dim=0).values.clamp_min(EPS)
            V = V / col_scales.unsqueeze(0)
            log_scales = log_scales + torch.log(col_scales)

        out[finite_mask] = torch.log(V.transpose(0, 1).clamp_min(EPS)) + log_scales.unsqueeze(1)
        return out

    def propagate_log_batch(
        self,
        child_logs: Tensor,
        child_time_idxs: Tensor,
        parent_time_idxs: Tensor,
    ) -> Tensor:
        if child_logs.numel() == 0:
            return child_logs

        pair_tensor = torch.stack((child_time_idxs, parent_time_idxs), dim=1)
        unique_pairs, inverse = torch.unique(pair_tensor, dim=0, return_inverse=True)
        propagated = torch.empty_like(child_logs, dtype=dtype)

        for group_idx in range(unique_pairs.shape[0]):
            mask = inverse == group_idx
            child_time_idx = int(unique_pairs[group_idx, 0].item())
            parent_time_idx = int(unique_pairs[group_idx, 1].item())
            propagated[mask] = self._propagate_same_range(
                child_logs[mask],
                child_time_idx,
                parent_time_idx,
            )

        return propagated

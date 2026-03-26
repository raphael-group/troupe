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

    def propagate_log_batch(
        self,
        child_logs: Tensor,
        child_time_idxs: Tensor,
        parent_time_idxs: Tensor,
    ) -> Tensor:
        if child_logs.numel() == 0:
            return child_logs
        if child_logs.shape[0] != child_time_idxs.shape[0] or child_logs.shape[0] != parent_time_idxs.shape[0]:
            raise ValueError("child_logs, child_time_idxs, and parent_time_idxs must agree in batch size")
        if (child_time_idxs > parent_time_idxs).any():
            raise ValueError("child_time_idxs must be <= parent_time_idxs")

        out = torch.full_like(child_logs, float("-inf"))
        finite_mask = torch.isfinite(child_logs).any(dim=1)
        if not finite_mask.any():
            return out

        work_logs = child_logs[finite_mask]
        work_child_times = child_time_idxs[finite_mask]
        work_parent_times = parent_time_idxs[finite_mask]

        log_scales = work_logs.max(dim=1).values
        V = torch.exp(work_logs - log_scales.unsqueeze(1)).transpose(0, 1)

        min_start = int(work_child_times.min().item())
        max_end = int(work_parent_times.max().item())

        for interval_idx in range(min_start, max_end):
            active_cols = ((work_child_times <= interval_idx) & (interval_idx < work_parent_times)).nonzero(as_tuple=True)[0]
            if active_cols.numel() == 0:
                continue

            V_active = self._rk4_step(V.index_select(1, active_cols), interval_idx).clamp_min(EPS)
            col_scales = V_active.max(dim=0).values.clamp_min(EPS)
            V_active = V_active / col_scales.unsqueeze(0)
            V[:, active_cols] = V_active
            log_scales[active_cols] = log_scales[active_cols] + torch.log(col_scales)

        out[finite_mask] = torch.log(V.transpose(0, 1).clamp_min(EPS)) + log_scales.unsqueeze(1)
        return out

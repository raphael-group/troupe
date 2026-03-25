import math
import logging
from typing import Optional, Iterable, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, DeviceObjType
from ete3 import TreeNode

from models import BaseLikelihoodModel, dtype, EPS, INF, INF_SM

import torchode as ode

logger = logging.getLogger(__name__)


class DaughterKernelBuilder(torch.nn.Module):
    """Construct a masked row-stochastic daughter kernel B."""

    def __init__(
        self,
        num_states: int,
        potency_constraints: Optional[Dict[int, tuple]] = None,
        device: Optional[DeviceObjType] = None,
        init_logits: Optional[Tensor] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.num_states = num_states

        if potency_constraints is None:
            mask = torch.ones((num_states, num_states), dtype=torch.bool, device=self.device)
        else:
            mask = self._build_support_mask(potency_constraints, num_states).to(self.device)
        self.register_buffer("support_mask", mask)

        free_rc = mask.nonzero(as_tuple=True)
        self.register_buffer("free_row_idx", free_rc[0])
        self.register_buffer("free_col_idx", free_rc[1])

        if init_logits is None:
            init_vec = torch.zeros(len(self.free_row_idx), dtype=dtype, device=self.device)
        else:
            if init_logits.ndim == 2:
                init_vec = init_logits.to(self.device, dtype=dtype)[mask]
            else:
                init_vec = init_logits.to(self.device, dtype=dtype).reshape(-1)
                if len(init_vec) != len(self.free_row_idx):
                    raise ValueError("init_logits has wrong length for daughter kernel support mask")
        self.free_params = torch.nn.Parameter(init_vec, requires_grad=True)

    @staticmethod
    def _build_support_mask(potency_constraints: Dict[int, tuple], num_states: int) -> Tensor:
        if len(potency_constraints) != num_states:
            raise ValueError("potency_constraints must have one entry per state")
        mask = torch.zeros((num_states, num_states), dtype=torch.bool)
        for i in range(num_states):
            from_potency = potency_constraints[i]
            for j in range(num_states):
                to_potency = potency_constraints[j]
                mask[i, j] = all(t in from_potency for t in to_potency)
        return mask

    def _full_logits(self) -> Tensor:
        logits = torch.full(
            (self.num_states, self.num_states),
            -1e30,
            dtype=dtype,
            device=self.device,
        )
        logits[self.free_row_idx, self.free_col_idx] = self.free_params
        return logits

    def forward(self) -> Tensor:
        return torch.softmax(self._full_logits(), dim=1)

    def log_forward(self) -> Tensor:
        return torch.log_softmax(self._full_logits(), dim=1)


class _AugmentedClaSSETerm(torch.nn.Module):
    """Joint ODE for E(t) and Y(t)=Phi(t,0)."""

    def __init__(self, B: Tensor, lam: Tensor):
        super().__init__()
        self.B = B
        self.lam = lam
        self.K = B.shape[0]
        self.register_buffer("minus_diag_lam", -torch.diag(lam))

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        del t
        batch_size = z.shape[0]
        K = self.K
        E = z[:, :K]
        Y = z[:, K:].reshape(batch_size, K, K)

        # q_b[i] = sum_j B[i,j] E_b[j]
        q = E @ self.B.transpose(0, 1)
        dE = self.lam.unsqueeze(0) * (q * q - E)

        A = self.minus_diag_lam.unsqueeze(0) + 2.0 * torch.diag_embed(self.lam.unsqueeze(0) * q) @ self.B.unsqueeze(0)
        dY = A @ Y
        return torch.cat((dE, dY.reshape(batch_size, K * K)), dim=1)


class _EClaSSETerm(torch.nn.Module):
    """ODE RHS for E(t) alone (K-dimensional)."""

    def __init__(self, B: Tensor, lam: Tensor):
        super().__init__()
        self.B = B
        self.lam = lam

    def forward(self, t: Tensor, E: Tensor) -> Tensor:
        del t
        q = E @ self.B.transpose(0, 1)
        return self.lam.unsqueeze(0) * (q * q - E)


class ClaSSELikelihoodModel(BaseLikelihoodModel):
    """Restricted ClaSSE model with terminal sampling.

    Parameters
    ----------
    birth_kernel_params
        Unconstrained logits for the daughter kernel B.
    growth_params
        Unconstrained birth-rate parameters. Transformed by softplus.
    sampling_prob
        Sampling probability of each leaf
    """

    def __init__(
        self,
        trees: Iterable[TreeNode],
        num_states: int,
        birth_kernel_params: Tensor,
        pi_params: Tensor,
        growth_params: Tensor,
        optimize_growth: bool = True,
        num_hidden: Optional[int] = None,
        idx2potency: Optional[dict] = None,
        device: Optional[DeviceObjType] = None,
        idx2state: Optional[dict] = None,
        start_state: Optional[int] = None,
        sampling_prob: float = None,
        optimize_sampling: bool = False,
        integration_max_step: float = 0.05,
        ode_method: str = "Dopri5",
        ode_atol: float = 1e-8,
        ode_rtol: float = 1e-6,
    ):
        super().__init__(trees, num_states, device)
        self.num_hidden = num_hidden
        self.idx2state = idx2state
        self.state2idx = {state: idx for idx, state in idx2state.items()} if idx2state is not None else None
        self.states = set(idx2state.values()) if idx2state is not None else None
        self.idx2potency = idx2potency
        self.integration_max_step = float(integration_max_step)
        self.ode_method = str(ode_method)
        self.ode_atol = float(ode_atol)
        self.ode_rtol = float(ode_rtol)
        self._ode_cache = None

        self.kernel_builder = DaughterKernelBuilder(
            num_states=num_states,
            potency_constraints=idx2potency,
            device=self.device,
            init_logits=birth_kernel_params,
        )
        self.growth_params = torch.nn.Parameter(growth_params.to(self.device, dtype=dtype), requires_grad=True)

        if sampling_prob is None:
            self.sampling_prob = torch.tensor(1.0, device=self.device, dtype=dtype)
        else:
            assert sampling_prob <= 1 and sampling_prob > 0
            self.sampling_prob = torch.as_tensor(sampling_prob, device=self.device, dtype=dtype).reshape(())

        self.pi_params = torch.nn.Parameter(pi_params.to(self.device, dtype=dtype), requires_grad=True)
        if start_state is not None:
            self.pi_params = torch.nn.Parameter(
                -INF_SM * torch.ones(num_states, device=self.device, dtype=dtype), requires_grad=False
            )
            self.pi_params.data[start_state] = INF_SM
        else:
            self.pi_params.requires_grad = True

        if idx2potency is not None:
            root_idx = 0
            for idx, potency in idx2potency.items():
                if len(potency) > len(idx2potency[root_idx]):
                    root_idx = idx
            self.root_idx = root_idx
            self.terminal_idx = [idx for idx, potency in idx2potency.items() if potency == (idx,)]

        self._tree_tensors = [self.tree_tens.get_tree_tensors(i) for i in range(len(self.tree_tens.trees))]
        self._prepare_time_metadata()

    # ----------------------------- public getters -----------------------------

    def get_birth_kernel_params(self) -> Tensor:
        return self.kernel_builder.free_params

    def get_pi_params(self) -> Tensor:
        return self.pi_params

    def get_rate_matrix(self) -> Tensor:
        # Backward-compatible name used by optimizer.py. Here it returns B.
        return self.get_daughter_kernel()

    def get_root_distribution(self) -> Tensor:
        return F.softmax(self.pi_params, dim=0)

    def get_growth_rates(self) -> Tensor:
        return F.softplus(self.growth_params, threshold=10)

    def get_sampling_probability(self) -> Tensor:
        return self.sampling_prob

    def get_daughter_kernel(self) -> Tensor:
        return self.kernel_builder.forward()

    def get_log_daughter_kernel(self) -> Tensor:
        return self.kernel_builder.log_forward()

    def get_regularization_matrix(self) -> Tensor:
        return self.get_daughter_kernel()

    # ------------------------------ ODE utilities -----------------------------

    @staticmethod
    def _time_key(t: Tensor) -> float:
        return round(float(t.item()), 12)

    def _unique_observation_times(self, abs_times: Tensor) -> Tensor:
        unique_times = sorted({float(t.item()) for t in abs_times})
        if not unique_times:
            return torch.tensor([0.0], device=self.device, dtype=dtype)
        if unique_times[0] != 0.0:
            unique_times = [0.0] + unique_times
        return torch.tensor(unique_times, device=self.device, dtype=dtype)

    def _dense_time_grid(self, abs_times: Tensor) -> Tensor:
        max_time = float(abs_times.max().item())
        if max_time <= EPS:
            return torch.tensor([0.0], device=self.device, dtype=dtype)

        unique_times = sorted({float(t.item()) for t in abs_times})
        grid = [0.0]
        for target in unique_times[1:]:
            prev = grid[-1]
            dt = target - prev
            n_steps = max(1, int(math.ceil(dt / max(self.integration_max_step, EPS))))
            for k in range(1, n_steps + 1):
                grid.append(prev + dt * (k / n_steps))
        if grid[-1] < max_time:
            grid.append(max_time)
        return torch.tensor(grid, device=self.device, dtype=dtype)

    def _prepare_time_metadata(self) -> None:
        """Precompute tree-independent time indexing metadata.

        Tree topologies and branch lengths are fixed during optimization, so
        the dense integration grid and the mapping from edges to unique
        (child_time, parent_time) pairs are also fixed. Building them once
        removes repeated Python dict/set work from every closure evaluation.
        """
        if not self._tree_tensors:
            self._global_time_grid = torch.tensor([0.0], device=self.device, dtype=dtype)
            self._global_time_to_idx = {self._time_key(self._global_time_grid[0]): 0}
            self._pair_child_time_idxs = torch.empty(0, dtype=torch.long, device=self.device)
            self._pair_parent_time_idxs = torch.empty(0, dtype=torch.long, device=self.device)
            self._tree_time_metadata = []
            return

        union_times = torch.cat([tree_tensors[7] for tree_tensors in self._tree_tensors])
        self._global_time_grid = self._dense_time_grid(union_times)
        self._global_time_to_idx = {
            self._time_key(t): idx for idx, t in enumerate(self._global_time_grid)
        }

        pair_to_idx: Dict[Tuple[int, int], int] = {}
        tree_metadata = []

        for postorder, children, _parents, _blens, init_p, _leaf_idxs, levels, abs_times in self._tree_tensors:
            node_time_idxs = torch.tensor(
                [self._global_time_to_idx[self._time_key(t)] for t in abs_times],
                dtype=torch.long,
                device=self.device,
            )
            pair_ids = torch.full(children.shape, -1, dtype=torch.long, device=self.device)

            for node_idx in postorder.tolist():
                child_pair = children[node_idx]
                valid_slots = (child_pair >= 0).nonzero(as_tuple=True)[0]
                if valid_slots.numel() == 0:
                    continue

                parent_time_idx = int(node_time_idxs[node_idx].item())
                for slot in valid_slots.tolist():
                    child_idx = int(child_pair[slot].item())
                    pair = (int(node_time_idxs[child_idx].item()), parent_time_idx)
                    pair_idx = pair_to_idx.setdefault(pair, len(pair_to_idx))
                    pair_ids[node_idx, slot] = pair_idx

            level_batches = []
            max_level = int(levels.max().item())
            for lvl in range(1, max_level + 1):
                mask = levels == lvl
                node_idxs = postorder[mask.nonzero(as_tuple=True)[0]]
                if node_idxs.numel() == 0:
                    continue
                level_batches.append(
                    {
                        "node_idxs": node_idxs,
                        "child_idxs": children[node_idxs],
                        "pair_ids": pair_ids[node_idxs],
                    }
                )

            root_idx = int((levels == int(levels.max().item())).nonzero(as_tuple=True)[0][0].item())
            tree_metadata.append(
                {
                    "leaf_mask": torch.isfinite(init_p).any(dim=1),
                    "pair_ids": pair_ids,
                    "postorder": postorder.tolist(),
                    "level_batches": level_batches,
                    "root_idx": root_idx,
                }
            )

        ordered_pairs = [None] * len(pair_to_idx)
        for pair, pair_idx in pair_to_idx.items():
            ordered_pairs[pair_idx] = pair

        if ordered_pairs:
            self._pair_child_time_idxs = torch.tensor(
                [pair[0] for pair in ordered_pairs], dtype=torch.long, device=self.device
            )
            self._pair_parent_time_idxs = torch.tensor(
                [pair[1] for pair in ordered_pairs], dtype=torch.long, device=self.device
            )
        else:
            self._pair_child_time_idxs = torch.empty(0, dtype=torch.long, device=self.device)
            self._pair_parent_time_idxs = torch.empty(0, dtype=torch.long, device=self.device)

        self._tree_time_metadata = tree_metadata

    def _e_rhs(self, E_vec: Tensor, B: Tensor, lam: Tensor) -> Tensor:
        q = B @ E_vec
        return lam * (q * q - E_vec)

    def _build_A(self, E_vec: Tensor, B: Tensor, lam: Tensor) -> Tensor:
        q = B @ E_vec
        return -torch.diag(lam) + 2.0 * torch.diag(lam * q) @ B

    def _solve_augmented_system_torchode(
        self,
        time_points: Tensor,
        B: Tensor,
        lam: Tensor,
        eta: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Solve the original augmented ClaSSE ODE for compatibility tests."""
        K = self.num_states
        E0 = (1.0 - eta).expand(K)
        Y0 = torch.eye(K, device=self.device, dtype=dtype).reshape(-1)
        z0 = torch.cat((E0, Y0), dim=0).unsqueeze(0)
        t_eval = time_points.unsqueeze(0)

        term = ode.ODETerm(_AugmentedClaSSETerm(B=B, lam=lam))
        step_method_cls = getattr(ode, self.ode_method, None)
        if step_method_cls is None:
            raise ValueError(f"Unknown torchode method '{self.ode_method}'")
        step_method = step_method_cls(term=term)
        step_controller = ode.IntegralController(atol=self.ode_atol, rtol=self.ode_rtol, term=term)
        solver = ode.AutoDiffAdjoint(step_method, step_controller)
        sol = solver.solve(ode.InitialValueProblem(y0=z0, t_eval=t_eval))

        z = sol.ys[0]
        E_vals = z[:, :K].clamp(min=EPS, max=1.0 - EPS)
        fundamental = z[:, K:].reshape(time_points.numel(), K, K)
        return E_vals, fundamental

    def _solve_e_torchode(
        self,
        time_grid: Tensor,
        B: Tensor,
        lam: Tensor,
        eta: Tensor,
    ) -> Tensor:
        """Solve only the K-dimensional E ODE via torchode.

        Compared to the augmented system this reduces the ODE state from
        K + K² to K — a (K+1)-fold reduction in state dimension.
        """
        K = self.num_states
        E0 = (1.0 - eta).expand(K).clone().unsqueeze(0)   # (1, K)
        t_eval = time_grid.unsqueeze(0)

        term = ode.ODETerm(_EClaSSETerm(B=B, lam=lam))
        step_method_cls = getattr(ode, self.ode_method, None)
        if step_method_cls is None:
            raise ValueError(f"Unknown torchode method '{self.ode_method}'")
        step_method = step_method_cls(term=term)
        step_controller = ode.IntegralController(atol=self.ode_atol, rtol=self.ode_rtol, term=term)
        solver = ode.AutoDiffAdjoint(step_method, step_controller)
        sol = solver.solve(ode.InitialValueProblem(y0=E0, t_eval=t_eval))

        return sol.ys[0].clamp(min=EPS, max=1.0 - EPS)   # (T, K)

    def _compute_fundamental_piecewise(
        self,
        time_grid: Tensor,
        E_vals: Tensor,
        B: Tensor,
        lam: Tensor,
    ) -> Tensor:
        """Approximate the fundamental matrix Y(t) via piecewise matrix exponentials.

        Given E(t) on the grid, A(t) = -diag(λ) + 2·diag(λ⊙q(t))·B is known.
        On each interval [tₖ, tₖ₊₁] we treat A as constant at its midpoint value
        and set Y(tₖ₊₁) ≈ exp(A_mid · Δt) · Y(tₖ).  All matrix exponentials are
        computed in a single batched call; the sequential product is then
        accumulated in a short Python loop (T-1 steps, each a KxK matmul).

        This is the zeroth-order Magnus / piecewise-constant approximation.
        Accuracy is controlled by integration_max_step (smaller → better).
        """
        K = self.num_states
        T = len(time_grid)

        if T == 1:
            return torch.eye(K, device=self.device, dtype=dtype).unsqueeze(0)

        # Midpoint E and derived quantities for each interval.
        E_mid = (E_vals[:-1] + E_vals[1:]) / 2          # (T-1, K)
        q_mid = E_mid @ B.T                               # (T-1, K)
        dts = time_grid[1:] - time_grid[:-1]              # (T-1,)

        # Build A at each midpoint in one vectorised operation: (T-1, K, K)
        A_mid = (
            -torch.diag(lam).unsqueeze(0)
            + 2.0 * torch.diag_embed(lam.unsqueeze(0) * q_mid) @ B.unsqueeze(0)
        )

        # Batched matrix exponential for all intervals at once.
        step_mats = torch.matrix_exp(A_mid * dts[:, None, None])   # (T-1, K, K)

        # Accumulate Y(tₖ₊₁) = step_mats[k] @ Y(tₖ),  Y(t₀) = I.
        fundamentals = [torch.eye(K, device=self.device, dtype=dtype)]
        for k in range(T - 1):
            fundamentals.append(step_mats[k] @ fundamentals[-1])

        return torch.stack(fundamentals)   # (T, K, K)

    def _solve_system(
        self,
        abs_times: Tensor,
        B: Tensor,
        lam: Tensor,
        eta: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Solve for E(t) and the fundamental matrix Y(t) on a dense time grid.

        Uses a K-dimensional ODE for E and a piecewise matrix-exponential
        approximation for Y, replacing the original (K+K²)-dimensional
        augmented torchode solve.
        """
        time_grid = self._dense_time_grid(abs_times)
        E_vals = self._solve_e_torchode(time_grid, B, lam, eta)
        fundamental = self._compute_fundamental_piecewise(time_grid, E_vals, B, lam)
        return time_grid, E_vals, fundamental

    def _log_transition_from_fundamental(self, Y_start: Tensor, Y_end: Tensor) -> Tensor:
        # Phi(t_end, t_start) = Phi(t_end, 0) Phi(t_start, 0)^{-1}
        P = torch.linalg.solve(Y_start.transpose(0, 1), Y_end.transpose(0, 1)).transpose(0, 1)
        return torch.log(P.clamp_min(EPS))

    # -------------------------------- ODE caching ----------------------------

    def precompute_ode(self) -> None:
        """Solve the ODE once and precompute all edge propagators across all trees.

        One call per closure evaluation suffices because the ODE (and therefore
        the fundamental matrix and all propagators) depends only on B, λ, η —
        not on tree topology.

        This method does three things in one shot:
          1. Solves the augmented ODE on the union of all trees' time points.
          2. Batch-inverts the fundamental matrix for every time point.
          3. Batch-computes log-propagators for every unique (child_t, parent_t)
             pair across all trees using a single bmm call.

        The cached log-propagators are then looked up directly in forward(),
        eliminating per-edge linalg.solve calls and redundant B/lam/eta
        recomputation.
        """
        B = self.get_daughter_kernel()
        lam = self.get_growth_rates()
        eta = self.get_sampling_probability()
        log_B = torch.log(B.clamp_min(EPS))
        log_lam = torch.log(lam.clamp_min(EPS))
        log_eta = torch.log(eta.clamp_min(EPS))

        time_grid = self._global_time_grid
        E_vals = self._solve_e_torchode(time_grid, B, lam, eta)
        fundamental = self._compute_fundamental_piecewise(time_grid, E_vals, B, lam)

        # Invert the fundamental matrix for all time points in one batched call.
        fundamental_inv = torch.linalg.inv(fundamental)  # (T, K, K)

        # Batch-compute Phi(t_parent, t_child) = Y(t_parent) @ Y(t_child)^{-1}
        # for all pairs in a single bmm, then take log.
        if self._pair_child_time_idxs.numel() > 0:
            log_props = torch.log(
                torch.bmm(
                    fundamental[self._pair_parent_time_idxs],
                    fundamental_inv[self._pair_child_time_idxs],
                ).clamp_min(EPS)
            )  # (N_pairs, K, K)
        else:
            log_props = torch.empty(
                0, self.num_states, self.num_states, device=self.device, dtype=dtype
            )

        self._ode_cache = {
            "log_B": log_B,
            "log_lam": log_lam,
            "log_eta": log_eta,
            "log_props": log_props,
        }

    def clear_ode_cache(self) -> None:
        """Release the cached ODE solution."""
        self._ode_cache = None

    # -------------------------------- likelihood ------------------------------

    def forward(self, tree_idx: int) -> Tensor:
        postorder, children, _parents, _blens, init_p, _leaf_idxs, levels, abs_times = self._tree_tensors[tree_idx]
        tree_metadata = self._tree_time_metadata[tree_idx]

        log_two = torch.log(torch.tensor(2.0, device=self.device, dtype=dtype))

        if self._ode_cache is not None:
            log_B = self._ode_cache["log_B"]
            log_lam = self._ode_cache["log_lam"]
            log_eta = self._ode_cache["log_eta"]
            log_props = self._ode_cache["log_props"]
            pair_ids = tree_metadata["pair_ids"]

            def _get_log_P(node_idx: int, child_slot: int) -> Tensor:
                return log_props[pair_ids[node_idx, child_slot]]
        else:
            B = self.get_daughter_kernel()
            log_B = torch.log(B.clamp_min(EPS))
            lam = self.get_growth_rates()
            log_lam = torch.log(lam.clamp_min(EPS))
            eta = self.get_sampling_probability()
            log_eta = torch.log(eta.clamp_min(EPS))
            time_grid, _E_vals, fundamental = self._solve_system(abs_times, B, lam, eta)
            time_to_idx = {self._time_key(t): i for i, t in enumerate(time_grid)}

            def _get_log_P(c_age_idx, p_age_idx):
                return self._log_transition_from_fundamental(
                    fundamental[c_age_idx], fundamental[p_age_idx]
                )

        N = postorder.numel()
        log_partials = torch.full((N, self.num_states), -INF, device=self.device, dtype=dtype)

        # Initialize leaves.
        leaf_mask = tree_metadata["leaf_mask"]
        log_partials[leaf_mask] = init_p[leaf_mask] + log_eta

        if self._ode_cache is not None:
            for level_batch in tree_metadata["level_batches"]:
                node_idxs = level_batch["node_idxs"]
                child_idxs = level_batch["child_idxs"]
                pair_ids = level_batch["pair_ids"]

                left_idxs = child_idxs[:, 0]
                left_terms = log_props[pair_ids[:, 0]] + log_partials[left_idxs].unsqueeze(1)
                left_child_logs = torch.logsumexp(left_terms, dim=2)
                node_values = left_child_logs.clone()

                right_mask = child_idxs[:, 1] >= 0
                if right_mask.any():
                    valid = right_mask.nonzero(as_tuple=True)[0]
                    right_idxs = child_idxs[valid, 1]
                    right_terms = log_props[pair_ids[valid, 1]] + log_partials[right_idxs].unsqueeze(1)
                    right_child_logs = torch.logsumexp(right_terms, dim=2)
                    left_mix = torch.logsumexp(
                        log_B.unsqueeze(0) + left_child_logs[valid].unsqueeze(1),
                        dim=2,
                    )
                    right_mix = torch.logsumexp(
                        log_B.unsqueeze(0) + right_child_logs.unsqueeze(1),
                        dim=2,
                    )
                    node_values[valid] = log_two + log_lam + left_mix + right_mix

                log_partials[node_idxs] = node_values
        else:
            for node_idx in tree_metadata["postorder"]:
                child_pair = children[node_idx]
                valid_slots = (child_pair >= 0).nonzero(as_tuple=True)[0]
                valid_children = child_pair[valid_slots]
                if valid_children.numel() == 0:
                    continue

                child_logs = []
                for slot, child_idx in zip(valid_slots.tolist(), valid_children.tolist()):
                    log_P = _get_log_P(
                        time_to_idx[self._time_key(abs_times[child_idx])],
                        time_to_idx[self._time_key(abs_times[node_idx])],
                    )
                    child_log = torch.logsumexp(log_P + log_partials[child_idx].unsqueeze(0), dim=1)
                    child_logs.append(child_log)

                if len(child_logs) == 1:
                    log_partials[node_idx] = child_logs[0]
                    continue

                if len(child_logs) != 2:
                    raise ValueError("ClaSSE likelihood expects a binary reconstructed tree (plus optional unary stem root).")

                left_mix = torch.logsumexp(log_B + child_logs[0].unsqueeze(0), dim=1)
                right_mix = torch.logsumexp(log_B + child_logs[1].unsqueeze(0), dim=1)
                log_partials[node_idx] = log_two + log_lam + left_mix + right_mix

        log_pi = self.pi_params - torch.logsumexp(self.pi_params, dim=0)
        return torch.logsumexp(log_partials[tree_metadata["root_idx"]] + log_pi, dim=0)

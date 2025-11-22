import torch
import torch.nn.functional as F
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
from torch import Tensor, DeviceObjType
from ete3 import TreeNode
from typing import Optional, Iterable, List
import warnings
from itertools import chain, combinations
import sys
import math

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000) 


# TODO: Make these more fitting for float64 precision
EPS = 1e-30                 # Used as numerically stable zero
INF = float('inf')          # Used to init log likelihoods 
INF_SM = 30                 # Largest value in params that are softmax'd
dtype=torch.float64
torch.set_default_dtype(dtype)

ode_atol = 1e-12
ode_rtol = 1e-8

class RateMatrixBuilder(torch.nn.Module):
    """
    Constructs rate matrix Q using constraints (potency, terminal, etc).
    """
    def __init__(self,
                 num_states: int,                                   # Total num states: hidden + observed
                 num_hidden_states: Optional[int] = 0,
                 potency_constraints: Optional[dict] = None,        # idx -> potency of original states
                 device: Optional[DeviceObjType] = None,
                 init_Q_params: Optional[Tensor] = None
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.num_states = num_states
        self.num_hidden_states = num_hidden_states
        self.num_observed_states = num_states-num_hidden_states

        def _build_mask(potency_constraints):
            assert len(potency_constraints) == self.num_states
            mask = torch.zeros((self.num_states, self.num_states), dtype=bool) # 0 if there should be a zero at this entry
            for i in range(num_states):
                from_potency = potency_constraints[i]
                for j in range(num_states):
                    to_potency = potency_constraints[j]
                    # `to` is contained in `from`
                    mask[i, j] = all([t in from_potency for t in to_potency]) and i != j

            return mask

        if potency_constraints is not None:
            mask = _build_mask(potency_constraints)
        else:
            mask = torch.ones((self.num_states, self.num_states), dtype=bool)
        self.register_buffer("potency_constraint_mask", mask)

        # Precompute free indices and keep them as buffers (on the right device)
        free_rc = self.potency_constraint_mask.nonzero(as_tuple=True)  # tuple(row, col)
        self.register_buffer("free_row_idx", free_rc[0])
        self.register_buffer("free_col_idx", free_rc[1])

        if init_Q_params is None:
            init_vec = torch.zeros(len(self.free_row_idx), dtype=dtype, device=self.device)
        else:
            # Accept either a full matrix or an already-packed vector
            if init_Q_params.ndim == 2:
                init_vec = init_Q_params.to(self.device, dtype=dtype)[self.potency_constraint_mask]
            else:
                init_vec = init_Q_params.to(self.device, dtype=dtype).reshape(-1)
                assert len(init_vec) == len(self.free_row_idx), "init_Q_params has wrong length."
        self.free_params = torch.nn.Parameter(init_vec, requires_grad=True)

    def forward(self) -> Tensor:
        n = self.num_states
        Q_off = torch.zeros((n, n), dtype=dtype, device=self.device)
        # Only free entries get positive rates via softplus
        Q_off[self.free_row_idx, self.free_col_idx] = F.softplus(self.free_params, threshold=10)
        Q = Q_off - torch.diag(Q_off.sum(dim=1))
        return Q


class TreeTensorizer:
    """
    Converts ete3 TreeNodes into tensor representations for pruning.
    """
    def __init__(self, trees: Iterable[TreeNode], num_states: int, device: DeviceObjType):
        self.num_states = num_states
        self.device = device
        self.trees = trees
        self.postorders = []
        self.children = []
        self.parents = []
        self.branch_lens = []
        self.partials = []
        self.leaf_idxs = []
        self.levels = []
        self.abs_times = []
        self._prepare_all()
        # Build one global, sorted, unique time grid across the forest
        all_times = torch.cat([t for t in self.abs_times], dim=0)
        self.global_time_grid, _ = torch.sort(torch.unique(all_times))

    def _prepare_all(self):
        for tree in self.trees:
            post, childs, parents, blens, partial, leaf_idxs, levels, abs_times = self._prep_single(tree)
            self.postorders.append(post)
            self.children.append(childs)
            self.parents.append(parents)
            self.branch_lens.append(blens)
            self.partials.append(partial)
            self.leaf_idxs.append(leaf_idxs)
            self.levels.append(levels)
            self.abs_times.append(abs_times)

    def _prep_single(self, tree: TreeNode):
        tree = tree.copy("deepcopy")
        if len(tree.children) != 1:
            warnings.warn("We require a unifurcating root with branch length 0. Adding one now.")
            root = TreeNode()
            root.dist = 0
            root.add_child(tree)
            tree = root

        nodes = list(tree.traverse("postorder"))
        N = len(nodes)

        postorder   = []
        children    = []
        branch_lens = torch.zeros(N, device=self.device, dtype=dtype)
        partials    = torch.full((N, self.num_states), -INF, device=self.device, dtype=dtype)
        leaf_idxs   = []
        parents     = torch.full((N,), -1, device=self.device, dtype=torch.long)

        # Name nodes by postorder index for stable tensor indexing
        for idx, node in enumerate(nodes):
            node.name = idx
            postorder.append(idx)

        for idx, node in enumerate(nodes):  # postorder: children before parent
            branch_lens[idx] = node.dist
            if not node.is_root():
                parents[idx] = int(node.up.name)

            kid_ids = [int(c.name) for c in node.get_children()]
            # Pad kid_ids w/ -1
            if len(kid_ids) == 0:
                kid_ids = [-1, -1]
            elif len(kid_ids) == 1:
                kid_ids.append(-1)
            children.append(torch.tensor(kid_ids, device=self.device))

            if node.is_leaf():
                s = int(node.state)
                partials[idx, s] = 0.0
                leaf_idxs.append(idx)

        # Set absolute times (time before present w/ leaves=0)
        abs_times  = torch.zeros(N, device=self.device, dtype=dtype)
        levels     = [0] * N  # filled below
        for idx, node in enumerate(nodes):  # postorder: children before parent
            if node.is_leaf():
                abs_times[idx] = 0.0
                levels[idx] = 0
            else:
                kid_ids = [int(c.name) for c in node.get_children()]
                t = max(abs_times[k] + nodes[k].dist for k in kid_ids)
                abs_times[idx] = t
                levels[idx] = 1 + max(levels[k] for k in kid_ids)

        return (
            torch.tensor(postorder, dtype=torch.long, device=self.device),
            torch.stack(children),
            parents,                          # (N,)
            branch_lens,                      # (N,)
            partials,                         # (N,S)
            torch.tensor(leaf_idxs, dtype=torch.long, device=self.device),
            torch.tensor(levels, dtype=torch.long, device=self.device),
            abs_times                         # (N,)
        )


class FelsensteinPruner:
    """
    Implements log-space likelihood computation via pruning.
    """
    def __init__(self, num_states: int):
        self.num_states = num_states
        self._cache = None  # will hold {time_grid, Phi, invPhi, rho, k}

    class Logit_E_ODE(torch.nn.Module):
        def __init__(self, lam, Q, eps=EPS):
            super().__init__()
            self.lam = lam
            self.Q = Q
            self.eps = eps
        def forward(self, t, u):
            E = torch.sigmoid(u)
            QE = self.Q @ E
            return -self.lam + QE / (E * (1.0 - E) + self.eps)

    @staticmethod
    def _solve_E_timeseries(lam, Q, rho, T, n_grid=1024):
        k = len(lam)
        one_minus_rho = max(EPS, float(1.0 - rho))
        u0_scalar = math.log(one_minus_rho) - math.log(1.0 - one_minus_rho)
        u0 = torch.full((k,), u0_scalar, device=lam.device, dtype=lam.dtype)
        rhs = Logit_E_ODE(lam, Q)
        ts = torch.linspace(0.0, float(T), steps=n_grid, device=lam.device, dtype=lam.dtype)
        us = odeint(rhs, u0, ts, rtol=ode_rtol, atol=ode_atol)  # (m,k)
        Es = torch.sigmoid(us)                                  # (m,k)
        return ts, Es

    class _EInterp(torch.nn.Module):
        def __init__(self, ts, Es):
            super().__init__()
            self.ts = ts
            self.Es = Es
            self.m = len(ts)
        def forward(self, t):
            t = t.clamp(min=self.ts[0], max=self.ts[-1])
            idx = torch.searchsorted(self.ts, t).clamp(1, self.m - 1)
            t0 = self.ts[idx-1]; t1 = self.ts[idx]
            w = (t - t0) / (t1 - t0 + 1e-12)
            E0 = self.Es[idx-1]; E1 = self.Es[idx]
            return (1.0 - w) * E0 + w * E1

    class _FundPhi_ODE(torch.nn.Module):
        """Phi'(t) = A(t) Phi(t), A(t)=Q + diag(-lam + 2lam E(t))"""
        def __init__(self, lam, Q, E_interp):
            super().__init__()
            self.lam = lam
            self.Q = Q
            self.E_interp = E_interp
        def forward(self, t, Phi):
            E_t = self.E_interp(t)                         # (k,)
            A_diag = -self.lam + 2.0 * self.lam * E_t      # (k,)
            return (self.Q @ Phi) + A_diag.unsqueeze(1) * Phi

    # Precompute fundamental matrix for all trees
    def prepare_global_cache(self,
                             time_grid: Tensor,
                             lam: Tensor,
                             Q: Tensor,
                             rho: float,
                             n_grid: Optional[int] = 1024):
        """
        time_grid: sorted unique absolute times across ALL trees, shape (M,)
        """
        device = lam.device
        dtype = lam.dtype
        k = len(lam)
        assert time_grid.ndim == 1 and torch.all(time_grid[:-1] <= time_grid[1:]), "time_grid must be sorted."

        # 1) Solve E(t) once on [0, T_max] to build an interpolant
        T_max = float(time_grid[-1].item())
        # Use a dense grid for E(t) interpolation
        ts_dense = torch.linspace(0.0, T_max, steps=n_grid, device=device, dtype=dtype)
        u0_scalar = torch.log(torch.tensor(1 - rho, device=device, dtype=torch.float64).clamp_min(1e-12)) \
                  - torch.log(torch.tensor(rho, device=device, dtype=torch.float64))
        u0 = u0_scalar.repeat(k)
        us = odeint(self.Logit_E_ODE(lam, Q), u0, ts_dense, rtol=ode_rtol, atol=ode_atol) # (n_grid,k)
        Es = torch.sigmoid(us)                                                            # (n_grid,k)   
        E_interp = self._EInterp(ts_dense, Es)

        # 2) Solve a single matrix ODE to get Phi at all required times
        Phi0 = torch.eye(k, device=device, dtype=dtype)
        # Phi_seq: (M, k, k) with Phi(time_grid[i])
        Phi_seq = odeint(self._FundPhi_ODE(lam, Q, E_interp), Phi0, time_grid, rtol=ode_rtol, atol=ode_atol)
        
        # # TODO: Remove this check for a speed up
        # # This should be cheap to check for small k
        # cond = torch.linalg.cond(Phi_seq)   # (M,)
        # if torch.any(cond > 1e14):
        #     max_condition = torch.max(cond)
        #     warnings.warn(f"Phi(t) is ill-conditioned at some times (cond = {max_condition})." + \
        #                    "Consider tighter rtol/atol or smaller time span.")

        self._cache = {
            "time_grid": time_grid,
            "Phi": Phi_seq,
            "rho": float(rho),
            "k": k
        }

    def build_transitions_from_cache(self,
                                     abs_times: Tensor,
                                     parents: Tensor,
                                     branch_lens: Tensor) -> Tensor:
        """
        Build per-edge transition matrices for one tree using cached Phi.

        Returns: trans_mat (N, S, S) where row i is the edge parent(i)->i (I for root or zero-length).
        """
        assert self._cache is not None, "Call prepare_global_cache(...) first."
        Phi = self._cache["Phi"]         # (M,k,k)
        time_grid = self._cache["time_grid"]
        device = Phi.device
        dtype = Phi.dtype
        N = len(abs_times)
        k = Phi.shape[1]

        # Map abs_times to indices into the global grid (exact values --> searchsorted)
        # Both are produced by the tensorizer; equality holds numerically.
        child_idx = torch.searchsorted(time_grid, abs_times)              # (N,)
        # Parents < 0 (root) get dummy index 0; we'll mask them to identity
        parent_idx = torch.where(parents >= 0,
                                 torch.searchsorted(time_grid, abs_times[parents.clamp_min(0)]),
                                 torch.zeros_like(parents))

        Phi_p = Phi[parent_idx]     # (N,k,k)
        Phi_c = Phi[child_idx]
        
        I = torch.eye(k, device=device, dtype=dtype)
        X_T = torch.linalg.solve(Phi_c.transpose(-1,-2) + EPS * I, Phi_p.transpose(-1,-2))
        trans_mat = X_T.transpose(-1,-2)

        # Identity on roots / zero-length edges
        I = torch.eye(k, device=device, dtype=dtype)
        mask = (parents < 0) | (branch_lens <= EPS)
        if mask.any():
            trans_mat = trans_mat.clone()
            trans_mat[mask] = I

        # Numerical safety
        trans_mat = trans_mat.clamp_min(EPS)
        return trans_mat

    def log_prune(
        self,
        postorder: Tensor,
        children: Tensor,
        parents: Tensor,
        branch_lens: Tensor,
        init_partials: Tensor,
        Q: Tensor,
        levels: Tensor,
        leaf_idxs: Optional[Tensor] = None,
        growth_rates: Optional[Tensor] = None,
        rho: Optional[float] = 1.0,
        abs_times: Optional[Tensor] = None) -> Tensor:

        log_partials = init_partials.clone()

        # Apply present-day sampling probability rho at the tips:
        if (rho is not None) and (rho < 1.0) and (leaf_idxs is not None):
            log_partials[leaf_idxs] = log_partials[leaf_idxs] + math.log(rho)

        if rho == 1.0:
            rate_matrix = Q - torch.diag(growth_rates) if growth_rates is not None else Q
            scaled      = rate_matrix.unsqueeze(0) * branch_lens.view(-1,1,1)
            trans_mat   = torch.matrix_exp(scaled).to(Q.dtype)
            trans_mat[branch_lens <= EPS] = torch.eye(self.num_states, device=Q.device, dtype=Q.dtype)
        else:
            assert abs_times is not None, "Need abs_times for rho < 1.0"
            assert growth_rates is not None, "Need growth_rates (lam) for rho < 1.0"

            # Use precomputed global cache
            trans_mat = self.build_transitions_from_cache(abs_times, parents, branch_lens)
        
        assert not torch.isnan(trans_mat).any()
        log_trans = torch.log(trans_mat.clamp_min(EPS))

        # Batched log-space pruning by levels
        max_level = int(levels.max().item())
        for lvl in range(1, max_level+1):
            mask       = levels == lvl
            node_idxs  = postorder[mask.nonzero(as_tuple=True)[0]]  # (M,)
            child_idxs = children[node_idxs]                        # (M,2)
            left_idxs  = child_idxs[:,0]
            right_idxs = child_idxs[:,1]

            P_left     = log_trans[left_idxs]                 # (M,S,S)
            L_left     = log_partials[left_idxs]              # (M,S)
            left_terms = P_left + L_left.unsqueeze(1)         # (M,S,S)
            left_contrib = torch.logsumexp(left_terms, dim=2) # (M,S)

            right_contrib = torch.zeros_like(left_contrib)
            mask_r = right_idxs >= 0
            if mask_r.any():
                valid      = mask_r.nonzero(as_tuple=True)[0]
                P_right    = log_trans[right_idxs[valid]]
                L_right    = log_partials[right_idxs[valid]]
                right_terms = P_right + L_right.unsqueeze(1)
                rc = torch.logsumexp(right_terms, dim=2)
                if growth_rates is not None:
                    rc = rc + torch.log(growth_rates).unsqueeze(0)
                right_contrib[valid] = rc

            log_partials[node_idxs] = left_contrib + right_contrib

        root_idx = (levels == max_level).nonzero(as_tuple=True)[0][0]
        return log_partials[root_idx]


class BaseLikelihoodModel(torch.nn.Module):
    """
    Abstract base for likelihood models.
    """
    def __init__(
        self,
        trees: Iterable[TreeNode],
        num_states: int,
        device: Optional[DeviceObjType] = None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.trees = trees
        self.num_states = num_states
        self.tree_tens = TreeTensorizer(trees, num_states, self.device)
        self.pruner = FelsensteinPruner(num_states)

    def forward(self, tree_idx: int) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward")


class CTMCLikelihoodModel(BaseLikelihoodModel):
    """
    ... .
    """
    def __init__(
        self,
        trees: Iterable[TreeNode],
        num_states: int,
        Q_params: Optional[Tensor] = None,
        pi_params: Optional[Tensor] = None,
        device: Optional[DeviceObjType] = None,
        growth_params: Optional[Tensor] = None,
        rho: Optional[float] = 1.0
    ):
        super().__init__(trees, num_states, device)
        self.rho = rho
        self.rate_builder = RateMatrixBuilder(num_states, device=self.device, init_Q_params=Q_params)
        # self.Q_params = torch.nn.Parameter(
        #     Q_params.to(dtype) if Q_params is not None else
        #     torch.ones((num_states, num_states), device=self.device, dtype=dtype), requires_grad=True
        # )
        self.pi_params = torch.nn.Parameter(
            pi_params.to(dtype) if pi_params is not None else
            torch.zeros(num_states, device=self.device, dtype=dtype), requires_grad=True
        )
        if growth_params is not None:
            self.growth_params = torch.nn.Parameter(growth_params.to(dtype), requires_grad=True)
        else:
            self.growth_params = None
    
    def prepare_numerical_pruner(self,
                                 lam: Tensor,
                                 Q: Tensor,
                                 rho: float):
        """
        Precompute E(t) and Phi(t) once over all birth times for rho < 1.
        Call this once per parameter update before iterating over trees.
        """
        if rho != self.rho:
            warnings.warn(f"Resetting rho in model {self.rho} -> {rho}")
            self.rho = rho
        time_grid = self.tree_tens.global_time_grid.to(device=self.device)
        self.pruner.prepare_global_cache(time_grid, lam, Q, rho)

    def forward(self, tree_idx: int) -> Tensor:
        Q = self.rate_builder.forward()
        post, children, parents, blens, init_p, leaf_idxs, levels, abs_times = (
            self.tree_tens.postorders[tree_idx],
            self.tree_tens.children[tree_idx],
            self.tree_tens.parents[tree_idx],
            self.tree_tens.branch_lens[tree_idx],
            self.tree_tens.partials[tree_idx],
            self.tree_tens.leaf_idxs[tree_idx],
            self.tree_tens.levels[tree_idx],
            self.tree_tens.abs_times[tree_idx]
        )
        if self.growth_params is not None:
            growth_rates = F.softplus(self.growth_params)
        else:
            growth_rates = None
        
        root_log_partial = self.pruner.log_prune(
            post,
            children,
            parents,
            blens,
            init_p,
            Q,
            levels,
            leaf_idxs,
            growth_rates=growth_rates,
            abs_times=abs_times,
            rho=self.rho
        )
        log_pi = self.pi_params - torch.logsumexp(self.pi_params, dim=0)
        return torch.logsumexp(root_log_partial + log_pi, dim=0)
    
    def get_Q_params(self):
        return self.rate_builder.free_params
    
    def get_pi_params(self):
        return self.pi_params
    
    def get_rate_matrix(self) -> Tensor:
        """
        Returns the rate matrix.
        """
        rate_matrix = self.rate_builder.forward()
        return rate_matrix
    
    def get_root_distribution(self) -> Tensor:
        """
        Returns the state distribution at the root node.
        """
        root_distr = F.softmax(self.pi_params, dim=0)
        return root_distr


class PureBirthLikelihoodModel(CTMCLikelihoodModel):
    """
    Pure-birth branching process likelihood model for a list of trees.

    NOTE: Assumes that tree leaves are labeled with the indices. E.g., we've already ran
          for leaf in tree   leaf.state = state2idx[leaf.state]
    """
    def __init__(
        self,
        trees: Iterable[TreeNode],
        num_states: int,
        Q_params: Tensor,
        pi_params: Tensor,
        growth_params: Tensor,
        terminal_states: Optional[List] = None,
        num_hidden: Optional[int] = None,
        idx2potency: Optional[dict] = None,
        device: Optional[DeviceObjType] = None,
        idx2state: Optional[dict] = None,
        start_state: Optional[int] = None,
        subsampling_rate: Optional[float] = 1.0
    ):
        super().__init__(trees = trees,
                         num_states = num_states,
                         Q_params = Q_params,
                         pi_params = pi_params,
                         device = device,
                         growth_params = growth_params,
                         rho = subsampling_rate)
        assert self.growth_params is not None
        assert num_hidden is not None or idx2potency is not None

        print(f"=> Saving RateMatrixBuilder with potency sets={idx2potency}")
        self.rate_builder = RateMatrixBuilder(num_states,
                                              num_hidden_states=num_hidden,
                                              potency_constraints=idx2potency,
                                              device=self.device,
                                              init_Q_params=Q_params)
        
        # # NOTE: For debugging purposes
        # print(idx2state)
        # print(self.rate_builder.potency_constraint_mask)
        
        if start_state is not None:
            if pi_params is not None:
                print("=> Overriding pi params and using a starting state of", idx2state[start_state])
                if idx2potency is not None:
                    print("=> Starting state has potency of", idx2potency[start_state])
            # Initialize root distribution to be concentrated on starting state
            self.pi_params = torch.nn.Parameter(
                -INF_SM * torch.ones(num_states, device=self.device, dtype=dtype), requires_grad=False
            )
            self.pi_params[start_state] = INF_SM
        else:
            # infer root distribution only if there are no observed progenitors
            print("=> Inferring root distribution")
            self.pi_params.requires_grad = True

        if idx2potency is not None:
            self.idx2potency = idx2potency
            # Find the most potent state
            root_idx = 0
            for idx, potency in idx2potency.items():
                if len(potency) > len(idx2potency[root_idx]):
                    root_idx = idx
            self.root_idx = root_idx

            state2idx = {state: idx for idx, state in idx2state.items()}
            # self.terminal_idx = torch.tensor([state2idx[state] for state in terminal_states], device=self.device)
        
        self.num_hidden = num_hidden
        self.growth_params.requires_grad = True
        self.idx2state = idx2state
        self.state2idx = state2idx
        self.states = set(idx2state.values())
        self.idx2potency = idx2potency

    def get_growth_rates(self):
        return F.softplus(self.growth_params, threshold=10)




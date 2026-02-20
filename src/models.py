import torch
import torch.nn.functional as F
from torch import Tensor, DeviceObjType
from ete3 import TreeNode
from typing import Optional, Iterable, List
import warnings
import sys
import logging

logger = logging.getLogger(__name__)

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000)


EPS = 1e-30                 # Used as numerically stable zero
INF = float('inf')          # Used to init log likelihoods
INF_SM = 30                 # Largest value in params that are softmax'd
dtype=torch.float64
torch.set_default_dtype(dtype)

class RateMatrixBuilder(torch.nn.Module):
    """Constructs a CTMC rate matrix Q from unconstrained parameters.

    Uses softplus parameterization to ensure positive off-diagonal entries
    and applies potency constraint masks to enforce biological realism.
    The diagonal is set so that each row sums to zero.

    Args:
        num_states: Total number of states (hidden + observed).
        num_hidden_states: Number of hidden (unobserved) states.
        potency_constraints: Optional dict mapping state index to a tuple
            of reachable state indices. If None, all transitions are allowed.
        device: Torch device for parameters and buffers.
        init_Q_params: Optional initial values for the free parameters,
            either as a full matrix or a packed vector.
    """
    def __init__(self,
                 num_states: int,
                 num_hidden_states: Optional[int] = 0,
                 potency_constraints: Optional[dict] = None,
                 device: Optional[DeviceObjType] = None,
                 init_Q_params: Optional[Tensor] = None
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.num_states = num_states
        self.num_hidden_states = num_hidden_states
        self.num_observed_states = num_states-num_hidden_states

        if potency_constraints is not None:
            mask = RateMatrixBuilder._build_mask(potency_constraints, num_states)
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

    @staticmethod
    def _build_mask(potency_constraints, num_states):
        """Builds a boolean mask from potency constraints.

        Entry (i, j) is True if state j's potency is a subset of state i's
        potency and i != j, meaning state i can transition to state j.

        Args:
            potency_constraints: Dict mapping state index to a tuple of
                reachable state indices.
            num_states: Total number of states.

        Returns:
            A boolean tensor of shape (num_states, num_states).
        """
        assert len(potency_constraints) == num_states
        mask = torch.zeros((num_states, num_states), dtype=bool)
        for i in range(num_states):
            from_potency = potency_constraints[i]
            for j in range(num_states):
                to_potency = potency_constraints[j]
                mask[i, j] = all([t in from_potency for t in to_potency]) and i != j
        return mask

    def forward(self) -> Tensor:
        """Constructs the rate matrix Q from the free parameters.

        Returns:
            The rate matrix Q (shape: [num_states, num_states]) with
            non-negative off-diagonals and rows summing to zero.
        """
        n = self.num_states
        Q_off = torch.zeros((n, n), dtype=dtype, device=self.device)
        # Only free entries get positive rates via softplus
        Q_off[self.free_row_idx, self.free_col_idx] = F.softplus(self.free_params, threshold=10)
        Q = Q_off - torch.diag(Q_off.sum(dim=1))
        return Q


class TreeTensorizer:
    """Converts ete3 TreeNodes into tensor representations for batched pruning.

    Processes a collection of trees into postorder indices, child arrays,
    parent arrays, branch lengths, leaf partials, level assignments, and
    absolute times suitable for the ``FelsensteinPruner``.

    Args:
        trees: An iterable of ete3 TreeNode objects.
        num_states: The total number of states.
        device: Torch device for all tensors.
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

    def get_tree_tensors(self, tree_idx):
        """Returns all tensor data for a single tree by index.

        Args:
            tree_idx: Integer index into the tree list.

        Returns:
            A tuple of (postorder, children, parents, branch_lens,
            partials, leaf_idxs, levels, abs_times).
        """
        return (
            self.postorders[tree_idx],
            self.children[tree_idx],
            self.parents[tree_idx],
            self.branch_lens[tree_idx],
            self.partials[tree_idx],
            self.leaf_idxs[tree_idx],
            self.levels[tree_idx],
            self.abs_times[tree_idx]
        )

    def _prepare_all(self):
        """Tensorizes all trees and stores the results."""
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
        """Tensorizes a single ete3 tree.

        Deep-copies the tree, adds a unifurcating root if needed, and
        extracts postorder indices, child arrays, parent arrays, branch
        lengths, leaf partials, level assignments, and absolute times.

        Args:
            tree: An ete3 TreeNode with ``.state`` attributes on leaves.

        Returns:
            A tuple of tensors: (postorder, children, parents, branch_lens,
            partials, leaf_idxs, levels, abs_times).
        """
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
    """Implements Felsenstein's pruning algorithm in log-space on tensorized trees.

    Processes nodes in batched level-order passes, computing transition
    matrices via ``torch.matrix_exp`` and accumulating log partial
    likelihoods.

    Args:
        num_states: The total number of states.
    """
    def __init__(self, num_states: int):
        self.num_states = num_states

    def log_prune(
        self,
        postorder: Tensor,
        children: Tensor,
        branch_lens: Tensor,
        init_partials: Tensor,
        Q: Tensor,
        levels: Tensor,
        growth_rates: Optional[Tensor] = None) -> Tensor:
        """Computes root log-partial-likelihoods via batched level-order pruning.

        Args:
            postorder: 1D tensor of node indices in postorder.
            children: 2D tensor of child indices (shape: [N, 2]), padded
                with -1 for missing children.
            branch_lens: 1D tensor of branch lengths (shape: [N]).
            init_partials: Initial log-partial matrix (shape: [N, num_states]).
            Q: Rate matrix (shape: [num_states, num_states]).
            levels: 1D tensor of node level assignments.
            growth_rates: Optional 1D tensor of birth rates per type.

        Returns:
            The log-partial-likelihood vector at the root (shape:
            [num_states]).
        """

        log_partials = init_partials.clone()

        rate_matrix = Q - torch.diag(growth_rates) if growth_rates is not None else Q
        scaled      = rate_matrix.unsqueeze(0) * branch_lens.view(-1,1,1)
        trans_mat   = torch.matrix_exp(scaled).to(Q.dtype)
        trans_mat[branch_lens <= EPS] = torch.eye(self.num_states, device=Q.device, dtype=Q.dtype)

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
    """Abstract base class for CTMC likelihood models.

    Handles tree tensorization and pruner initialization. Subclasses must
    implement ``forward(tree_idx)``.

    Args:
        trees: An iterable of ete3 TreeNode objects.
        num_states: The total number of states.
        device: Torch device (default: CPU).
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
    """CTMC likelihood model for phylogenetic trees.

    Combines a ``RateMatrixBuilder`` for rate matrix construction with
    ``FelsensteinPruner`` for log-likelihood computation. Supports optional
    per-type growth rates.

    Args:
        trees: An iterable of ete3 TreeNode objects.
        num_states: The total number of states.
        Q_params: Optional initial rate matrix parameters.
        pi_params: Optional initial root distribution logits.
        device: Torch device (default: CPU).
        growth_params: Optional initial growth rate parameters.
    """
    def __init__(
        self,
        trees: Iterable[TreeNode],
        num_states: int,
        Q_params: Optional[Tensor] = None,
        pi_params: Optional[Tensor] = None,
        device: Optional[DeviceObjType] = None,
        growth_params: Optional[Tensor] = None
    ):
        super().__init__(trees, num_states, device)
        self.rate_builder = RateMatrixBuilder(num_states, device=self.device, init_Q_params=Q_params)
        self.pi_params = torch.nn.Parameter(
            pi_params.to(dtype) if pi_params is not None else
            torch.zeros(num_states, device=self.device, dtype=dtype), requires_grad=True
        )
        if growth_params is not None:
            self.growth_params = torch.nn.Parameter(growth_params.to(dtype), requires_grad=True)
        else:
            self.growth_params = None

    def forward(self, tree_idx: int) -> Tensor:
        """Computes the log-likelihood for a single tree.

        Args:
            tree_idx: Index of the tree to evaluate.

        Returns:
            The log-likelihood (a scalar tensor).
        """
        Q = self.rate_builder.forward()
        post, children, _parents, blens, init_p, _leaf_idxs, levels, _abs_times = (
            self.tree_tens.get_tree_tensors(tree_idx)
        )
        if self.growth_params is not None:
            growth_rates = F.softplus(self.growth_params, threshold=10)
        else:
            growth_rates = None

        root_log_partial = self.pruner.log_prune(
            post,
            children,
            blens,
            init_p,
            Q,
            levels,
            growth_rates=growth_rates
        )
        log_pi = self.pi_params - torch.logsumexp(self.pi_params, dim=0)
        return torch.logsumexp(root_log_partial + log_pi, dim=0)

    def get_Q_params(self):
        """Returns the free parameters of the rate matrix builder."""
        return self.rate_builder.free_params

    def get_pi_params(self):
        """Returns the root distribution logit parameters."""
        return self.pi_params

    def get_rate_matrix(self) -> Tensor:
        """Returns the constructed rate matrix Q."""
        rate_matrix = self.rate_builder.forward()
        return rate_matrix

    def get_root_distribution(self) -> Tensor:
        """Returns the root state distribution (softmax of pi_params)."""
        root_distr = F.softmax(self.pi_params, dim=0)
        return root_distr


class PureBirthLikelihoodModel(CTMCLikelihoodModel):
    """Pure-birth branching process likelihood model.

    Extends ``CTMCLikelihoodModel`` with per-type growth rates, potency
    constraints, and optional fixed start state. Assumes tree leaves are
    labeled with integer state indices.

    Args:
        trees: An iterable of ete3 TreeNode objects.
        num_states: Total number of states (hidden + observed).
        Q_params: Initial rate matrix parameters.
        pi_params: Initial root distribution logits.
        growth_params: Initial growth rate parameters.
        num_hidden: Number of hidden states.
        idx2potency: Optional dict mapping state index to potency tuple.
        device: Torch device (default: CPU).
        idx2state: Optional dict mapping state index to state label.
        start_state: Optional index of the fixed starting state.
    """
    def __init__(
        self,
        trees: Iterable[TreeNode],
        num_states: int,
        Q_params: Tensor,
        pi_params: Tensor,
        growth_params: Tensor,
        optimize_growth: bool = True,
        num_hidden: Optional[int] = None,
        idx2potency: Optional[dict] = None,
        device: Optional[DeviceObjType] = None,
        idx2state: Optional[dict] = None,
        start_state: Optional[int] = None
    ):
        super().__init__(trees = trees,
                         num_states = num_states,
                         Q_params = Q_params,
                         pi_params = pi_params,
                         device = device,
                         growth_params = growth_params)
        assert self.growth_params is not None
        assert num_hidden is not None or idx2potency is not None

        logger.info("Saving RateMatrixBuilder with potency sets=%s", idx2potency)
        self.rate_builder = RateMatrixBuilder(num_states,
                                              num_hidden_states=num_hidden,
                                              potency_constraints=idx2potency,
                                              device=self.device,
                                              init_Q_params=Q_params)

        logger.debug("idx2state: %s", idx2state)
        logger.debug("potency_constraint_mask:\n%s", self.rate_builder.potency_constraint_mask)

        if start_state is not None:
            if pi_params is not None:
                logger.info("Overriding pi params and using a starting state of %s", idx2state[start_state])
                if idx2potency is not None:
                    logger.info("Starting state has potency of %s", idx2potency[start_state])
            # Initialize root distribution to be concentrated on starting state
            self.pi_params = torch.nn.Parameter(
                -INF_SM * torch.ones(num_states, device=self.device, dtype=dtype), requires_grad=False
            )
            self.pi_params[start_state] = INF_SM
        else:
            # infer root distribution only if there are no observed progenitors
            logger.info("Inferring root distribution")
            self.pi_params.requires_grad = True

        if idx2potency is not None:
            self.idx2potency = idx2potency
            # Find the most potent state
            root_idx = 0
            for idx, potency in idx2potency.items():
                if len(potency) > len(idx2potency[root_idx]):
                    root_idx = idx
            self.root_idx = root_idx
            # Terminal states: those whose potency set contains only themselves
            self.terminal_idx = [idx for idx, potency in idx2potency.items() if potency == (idx,)]

        self.num_hidden = num_hidden
        self.growth_params.requires_grad = optimize_growth
        self.idx2state = idx2state
        self.state2idx = {state: idx for idx, state in idx2state.items()} if idx2state is not None else None
        self.states = set(idx2state.values()) if idx2state is not None else None
        self.idx2potency = idx2potency

    def get_growth_rates(self):
        """Returns the growth rates (softplus of growth_params)."""
        return F.softplus(self.growth_params, threshold=10)

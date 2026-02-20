"""
Script for optimizing log likelihood objective on given trees.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging

from models import PureBirthLikelihoodModel

import pickle
import time
import sys
import math

logger = logging.getLogger(__name__)

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000)

INF = float('inf')
EPS = 1e-30

dtype=torch.float64
torch.set_default_dtype(dtype)


def _save_model_dict(llh, model_info, output_dir):
    """Saves the model's current state to a pickle file.

    Constructs a dict with the rate matrix, root distribution, and
    (if available) growth rates, then merges in any extra keys from
    model_info before pickling.

    Args:
        llh: The likelihood model instance.
        model_info: Optional dict of additional metadata to include.
        output_dir: Directory in which to save ``model_dict.pkl``.
    """
    model_dict = {
        "rate_matrix": llh.get_rate_matrix(),
        "root_distribution": llh.get_root_distribution()
    }
    if hasattr(llh, "get_growth_rates"):
        model_dict["growth_rates"] = llh.get_growth_rates()
    if model_info is not None:
        for k, v in model_info.items():
            if k not in model_dict:
                model_dict[k] = v
    with open(f"{output_dir}/model_dict.pkl", "wb") as fp:
        pickle.dump(model_dict, fp)
        logger.info("Saving model_dict to %s/model_dict.pkl", output_dir)


def compute_mle(trees,
                num_states,
                device,
                output_dir,
                l1_regularization_strength=0.0,
                do_logging=True,
                model_type="PureBirthLikelihoodModel",
                model_info=None):
    """Runs LBFGS optimization to learn CTMC parameters via MLE.

    Initializes the rate matrix and root distribution parameters, then
    trains a ``PureBirthLikelihoodModel`` by minimizing the negative
    log-likelihood (plus optional regularization). Saves the best model
    checkpoint and final model dict to ``output_dir``.

    Args:
        trees: List of ete3 TreeNode objects.
        num_states: A tuple (n_obs, n_hidden) specifying observed and
            hidden state counts.
        device: Torch device (CPU or GPU).
        output_dir: Directory for saving checkpoints and results.
        l1_regularization_strength: L1 penalty on rate matrix entries.
        do_logging: If True, log training progress at regular intervals.
        model_type: Model class name (default: "PureBirthLikelihoodModel").
        model_info: Dict with at least ``idx2potency`` and ``idx2state``
            keys, plus optional ``start_state``, ``growth_params_init``,
            and ``Q_params_init``.

    Returns:
        A tuple (llh, loss) of the trained model and final loss value.
    """

    llh, loss = _run_lbfgs(trees,
                           num_states,
                           device,
                           output_dir,
                           l1_regularization_strength=l1_regularization_strength,
                           do_logging=do_logging,
                           model_type=model_type,
                           model_info=model_info)

    _save_model_dict(llh, model_info, output_dir)

    return llh, loss

def constant_rate_mle(trees):
    """Estimates a constant birth rate from branch lengths via MLE.

    Computes the average ratio of internal node count to twice the total
    branch length across all trees, giving a simple moment-based birth
    rate estimate.

    Args:
        trees: List of ete3 TreeNode objects.

    Returns:
        The estimated constant birth rate (float).
    """
    avg_branch_lens = []
    for tree in trees:
        branch_sum = 0
        num_nodes = 0
        for node in tree.traverse():
            if node.is_leaf():
                continue
            branch_sum += node.dist
            num_nodes += 1
        if branch_sum > 0:
            avg_branch_lens.append(num_nodes/ (2 * branch_sum))
    return sum(avg_branch_lens) / len(avg_branch_lens)


def loss_fn(llh, tree_idxs, l1_regularization_strength=0):
    """Computes the regularized negative log-likelihood loss.

    Averages the negative log-likelihood over the given tree indices
    and adds optional L1 penalty.

    Args:
        llh: The likelihood model instance.
        tree_idxs: List of tree indices to evaluate.
        l1_regularization_strength: L1 penalty weight.

    Returns:
        The total loss (a scalar tensor).
    """
    loss = 0.0
    for j in tree_idxs:             # TODO: Parallelize this with tree indices
        tree_llh = llh(j)
        loss = loss - tree_llh
    loss = loss / len(tree_idxs)    # TODO: Rethink whether you should be using this normalization
    if l1_regularization_strength > 0:
        loss = loss + sparse_regularization(llh) * l1_regularization_strength
    return loss

def sparse_regularization(llh):
    """Computes L1 regularization on off-diagonal rate matrix entries.

    Args:
        llh: The likelihood model instance.

    Returns:
        The sum of absolute values of off-diagonal Q entries (scalar tensor).
    """
    Q = llh.get_rate_matrix()
    mask = ~torch.eye(llh.num_states, dtype=torch.bool, device=Q.device)
    rate_regularization = torch.sum(torch.abs(Q[mask]))
    return rate_regularization


def _run_lbfgs(trees,
               num_states,
               device,
               output_dir,
               l1_regularization_strength=0.0,
               do_logging=True,
               model_type="PureBirthLikelihoodModel",
               model_info=None):
    """Runs the LBFGS optimization loop for MLE inference.

    Initializes rate matrix and growth parameters (using constant-rate
    MLE if not provided), constructs a ``PureBirthLikelihoodModel``, and
    iterates LBFGS until convergence or max iterations. Saves the best
    checkpoint when loss improves.

    Args:
        trees: List of ete3 TreeNode objects.
        num_states: A tuple (n_obs, n_hidden).
        device: Torch device.
        output_dir: Directory for saving checkpoints.
        l1_regularization_strength: L1 penalty weight.
        do_logging: If True, log progress periodically.
        model_type: Model class name.
        model_info: Dict with model configuration (must contain
            ``idx2potency`` and ``idx2state``).

    Returns:
        A tuple (llh, final_loss) of the trained model and last loss value.
    """

    assert model_info is not None and "idx2potency" in model_info and "idx2state" in model_info

    num_iter = 50
    log_iter = 2
    rel_loss_thresh = 1e-5 / len(trees)
    n_obs, n_hidden = num_states
    n_states = n_obs + n_hidden

    losses = []

    start_state = None
    if "start_state" in model_info:
        start_state = model_info["start_state"]
    logger.info("Start state: %s", start_state)

    if "optimize_growth" in model_info:
        optimize_growth = model_info["optimize_growth"]
    else:
        optimize_growth = True
    logger.info("optimize_growth: %s", optimize_growth)

    idx2potency = model_info["idx2potency"]
    idx2state = model_info["idx2state"]

    pi_params_init = torch.zeros(n_states, device=device)

    if "growth_params_init" not in model_info:
        lam = constant_rate_mle(trees)
        growth_params_init = torch.log(torch.exp(torch.ones(n_states, device=device) * lam) - 1)
    else:
        growth_params_init = model_info["growth_params_init"].to(device)
    logger.info("Initializing growth params as: %s", F.softplus(growth_params_init))

    if "Q_params_init" not in model_info:
        # Initialize parameters uniformly among all descendant states so that process is critical
        max_potency_size = max([len(potency) for potency in idx2potency.values()])
        Q_params_init = torch.log(torch.exp(torch.zeros((n_states,n_states), device=device)) - 1)
        for i, potency in idx2potency.items():
            if len(potency) == max_potency_size:    # The totipotent state is potent for all other states
                for j in range(n_states):
                    if i != j:
                        Q_params_init[i, j] = math.log(math.exp(lam / (n_states-1)) - 1)
            else:
                descendant_states = [k for k, child_potency in idx2potency.items() \
                                    if all([l in potency for l in child_potency]) and potency != child_potency]
                for j in descendant_states:
                    Q_params_init[i, j] = math.log(math.exp(lam / len(descendant_states)) - 1)
    else:
        Q_params_init = model_info["Q_params_init"].to(device)

    Q_init = F.softplus(Q_params_init)
    Q_init = Q_init.fill_diagonal_(0)
    Q_init -= torch.diag(Q_init.sum(dim=1))
    logger.info("Initializing transition rates as:\n%s", Q_init)

    llh = PureBirthLikelihoodModel(trees,
                                    n_states,
                                    Q_params=Q_params_init,
                                    pi_params=pi_params_init,
                                    growth_params=growth_params_init,
                                    optimize_growth=optimize_growth,
                                    num_hidden=n_hidden,
                                    idx2potency=idx2potency,
                                    device=device,
                                    idx2state=idx2state,
                                    start_state=start_state)

    tree_idxs = [i for i in range(len(trees))]
    changeable_params = [p for p in llh.parameters(recurse=True) if p.requires_grad]
    logger.debug("Q_params: %s", llh.get_Q_params())

    logger.info("Using strong wolfe line search in LBFGS")
    # TODO: Using smaller lr rate and strong_wolfe --> less NaNs for some reason
    optimizer = optim.LBFGS(
        changeable_params,
        lr=0.1,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        objective = loss_fn(llh,
                            tree_idxs,
                            l1_regularization_strength=l1_regularization_strength)
        if not torch.isfinite(objective):
            raise RuntimeError(f"Non-finite loss: {objective.item()}")
        objective.backward()

        g = grad_norm(llh)
        if torch.isnan(g) or torch.isinf(g):
            raise RuntimeError(f"Gradient blew up: {g}")

        return objective

    start = time.time()

    for i in range(num_iter):

        with torch.no_grad():
            loss = loss_fn(llh,
                        tree_idxs,
                        l1_regularization_strength=l1_regularization_strength)
            logger.info("Iteration %d loss: %s", i, loss)

        loss = optimizer.step(closure)

        assert llh.get_Q_params().requires_grad == True

        with torch.no_grad():
            losses.append(loss.item())
            rate_matrix = llh.get_rate_matrix()
            pi = llh.get_root_distribution()
            if torch.isnan(rate_matrix).any() or torch.isnan(pi).any() or loss.isnan():
                raise ValueError(f"Invalid value: Nan in parameters/loss\n {llh.get_Q_params()}\n {llh.get_pi_params()}\n {loss}")

        if losses[-1] <= min(losses) and len(losses) >= 2:
            logger.info("Curr loss <= prev best loss: %s vs %s", losses[-1], min(losses[:-1]))
            with torch.no_grad():
                file_name = f"{output_dir}/state_dict.pth"
                torch.save(llh.state_dict(), file_name)
                _save_model_dict(llh, model_info, output_dir)

        if do_logging and i % log_iter == 0:
            logger.info("Iteration %d loss: %s", i, loss.item())
            logger.info("Rate matrix:\n%s", rate_matrix)
            logger.info("Root distribution: %s", pi)
            if hasattr(llh, "get_growth_rates"):
                logger.info("Growth rates: %s", llh.get_growth_rates())

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("%.4f s/iter", (time.time() - start) / min(i+1, log_iter))
            start = time.time()


        if len(losses) > 10:
            rel = abs(losses[-1] - losses[-2]) / (abs(losses[-2]) + EPS)
            if rel <= rel_loss_thresh:
                break

    return llh, losses[-1]


def grad_norm(llh):
    """Computes the L2 norm of all parameter gradients.

    Args:
        llh: The likelihood model instance.

    Returns:
        The total gradient norm (scalar tensor).
    """
    return torch.sqrt(sum((p.grad.detach().norm()**2 for p in llh.parameters() if p.grad is not None)))

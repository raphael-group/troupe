"""
Patched optimizer that can handle either the original PureBirthLikelihoodModel
or the new ClaSSELikelihoodModel.
"""
import math
import logging
import pickle
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import PureBirthLikelihoodModel
from classe_model import ClaSSELikelihoodModel

logger = logging.getLogger(__name__)

sys.setrecursionlimit(5000)

INF = float("inf")
EPS = 1e-30

dtype = torch.float64
torch.set_default_dtype(dtype)

MODEL_REGISTRY = {
    "PureBirthLikelihoodModel": PureBirthLikelihoodModel,
    "ClaSSELikelihoodModel": ClaSSELikelihoodModel,
}


def _safe_softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=EPS)
    return torch.log(torch.expm1(x))


def _safe_logit(p: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, min=EPS, max=1.0 - EPS)
    return torch.log(p) - torch.log1p(-p)


def _uniform_kernel_logits(idx2potency, n_states, device):
    logits = torch.full((n_states, n_states), -1e30, device=device, dtype=dtype)
    if idx2potency is None:
        logits[:] = 0.0
        return logits
    for i, potency in idx2potency.items():
        allowed = [
            j for j, child_potency in idx2potency.items()
            if all(t in potency for t in child_potency)
        ]
        logits[i, allowed] = 0.0
    return logits


def _save_model_dict(llh, model_info, output_dir):
    model_dict = {
        "rate_matrix": llh.get_rate_matrix(),
        "root_distribution": llh.get_root_distribution(),
    }
    if hasattr(llh, "get_growth_rates"):
        model_dict["growth_rates"] = llh.get_growth_rates()
    if hasattr(llh, "get_daughter_kernel"):
        model_dict["daughter_kernel"] = llh.get_daughter_kernel()
    if hasattr(llh, "get_sampling_probability"):
        model_dict["sampling_probability"] = llh.get_sampling_probability()
    if model_info is not None:
        for k, v in model_info.items():
            if k not in model_dict:
                model_dict[k] = v
    with open(f"{output_dir}/model_dict.pkl", "wb") as fp:
        pickle.dump(model_dict, fp)
        logger.info("Saving model_dict to %s/model_dict.pkl", output_dir)


def compute_mle(
    trees,
    num_states,
    device,
    output_dir,
    l1_regularization_strength=0.0,
    do_logging=True,
    model_type="PureBirthLikelihoodModel",
    model_info=None,
):
    llh, loss = _run_lbfgs(
        trees,
        num_states,
        device,
        output_dir,
        l1_regularization_strength=l1_regularization_strength,
        do_logging=do_logging,
        model_type=model_type,
        model_info=model_info,
    )
    _save_model_dict(llh, model_info, output_dir)
    return llh, loss


def constant_rate_mle(trees):
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
            avg_branch_lens.append(num_nodes / (2 * branch_sum))
    return sum(avg_branch_lens) / len(avg_branch_lens)


def loss_fn(llh, tree_idxs, l1_regularization_strength=0.0):
    loss = 0.0
    for j in tree_idxs:
        loss = loss - llh(j)
    loss = loss / len(tree_idxs)
    if l1_regularization_strength > 0:
        loss = loss + sparse_regularization(llh) * l1_regularization_strength
    return loss


def sparse_regularization(llh):
    if hasattr(llh, "get_regularization_matrix"):
        M = llh.get_regularization_matrix()
    else:
        M = llh.get_rate_matrix()
    mask = ~torch.eye(llh.num_states, dtype=torch.bool, device=M.device)
    return torch.sum(torch.abs(M[mask]))


def _run_lbfgs(
    trees,
    num_states,
    device,
    output_dir,
    l1_regularization_strength=0.0,
    do_logging=True,
    model_type="PureBirthLikelihoodModel",
    model_info=None,
):
    assert model_info is not None and "idx2potency" in model_info and "idx2state" in model_info
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type={model_type}. Options: {sorted(MODEL_REGISTRY)}")

    model_cls = MODEL_REGISTRY[model_type]

    num_iter = 50
    log_iter = 2
    rel_loss_thresh = 1e-5 / len(trees)
    n_obs, n_hidden = num_states
    n_states = n_obs + n_hidden
    losses = []

    start_state = model_info.get("start_state")
    optimize_growth = model_info.get("optimize_growth", True)
    optimize_sampling = model_info.get("optimize_sampling", False)
    idx2potency = model_info["idx2potency"]
    idx2state = model_info["idx2state"]
    integration_max_step = model_info.get("integration_max_step", 0.05)
    ode_backend = model_info.get("ode_backend", "torchode")
    ode_method = model_info.get("ode_method", "Dopri5")
    ode_atol = model_info.get("ode_atol", 1e-8)
    ode_rtol = model_info.get("ode_rtol", 1e-6)

    logger.info("Start state: %s", start_state)
    logger.info("optimize_growth: %s", optimize_growth)

    pi_params_init = torch.zeros(n_states, device=device, dtype=dtype)

    if "growth_params_init" not in model_info:
        lam = constant_rate_mle(trees)
        growth_params_init = _safe_softplus_inverse(torch.ones(n_states, device=device, dtype=dtype) * lam)
    else:
        growth_params_init = model_info["growth_params_init"].to(device=device, dtype=dtype)
    logger.info("Initializing growth params as: %s", F.softplus(growth_params_init))

    if model_type == "ClaSSELikelihoodModel":
        if "Q_params_init" in model_info:
            Q_params_init = model_info["Q_params_init"].to(device=device, dtype=dtype)
        elif "B_params_init" in model_info:
            Q_params_init = model_info["B_params_init"].to(device=device, dtype=dtype)
        else:
            Q_params_init = _uniform_kernel_logits(idx2potency, n_states, device)
        logger.info("Initializing daughter-kernel logits for ClaSSE model")

        if "sampling_param_init" in model_info:
            sampling_param_init = torch.as_tensor(model_info["sampling_param_init"], device=device, dtype=dtype)
        else:
            eta0 = model_info.get("sampling_prob", model_info.get("eta", 1.0))
            sampling_param_init = _safe_logit(torch.as_tensor(eta0, device=device, dtype=dtype))

        llh = model_cls(
            trees,
            n_states,
            Q_params=Q_params_init,
            pi_params=pi_params_init,
            growth_params=growth_params_init,
            optimize_growth=optimize_growth,
            num_hidden=n_hidden,
            idx2potency=idx2potency,
            device=device,
            idx2state=idx2state,
            start_state=start_state,
            sampling_param=sampling_param_init,
            optimize_sampling=optimize_sampling,
            integration_max_step=integration_max_step,
            ode_backend=ode_backend,
            ode_method=ode_method,
            ode_atol=ode_atol,
            ode_rtol=ode_rtol,
        )
    else:
        if "Q_params_init" not in model_info:
            max_potency_size = max(len(potency) for potency in idx2potency.values())
            lam0 = float(F.softplus(growth_params_init[0]).item())
            Q_params_init = _safe_softplus_inverse(torch.zeros((n_states, n_states), device=device, dtype=dtype))
            for i, potency in idx2potency.items():
                if len(potency) == max_potency_size:
                    for j in range(n_states):
                        if i != j:
                            Q_params_init[i, j] = math.log(math.exp(lam0 / (n_states - 1)) - 1)
                else:
                    descendant_states = [
                        k for k, child_potency in idx2potency.items()
                        if all(l in potency for l in child_potency) and potency != child_potency
                    ]
                    for j in descendant_states:
                        Q_params_init[i, j] = math.log(math.exp(lam0 / len(descendant_states)) - 1)
        else:
            Q_params_init = model_info["Q_params_init"].to(device=device, dtype=dtype)

        Q_init = F.softplus(Q_params_init)
        Q_init = Q_init.fill_diagonal_(0)
        Q_init -= torch.diag(Q_init.sum(dim=1))
        logger.info("Initializing transition rates as:\n%s", Q_init)

        llh = model_cls(
            trees,
            n_states,
            Q_params=Q_params_init,
            pi_params=pi_params_init,
            growth_params=growth_params_init,
            optimize_growth=optimize_growth,
            num_hidden=n_hidden,
            idx2potency=idx2potency,
            device=device,
            idx2state=idx2state,
            start_state=start_state,
        )

    tree_idxs = list(range(len(trees)))
    changeable_params = [p for p in llh.parameters(recurse=True) if p.requires_grad]
    logger.debug("Q_params: %s", llh.get_Q_params())

    optimizer = optim.LBFGS(changeable_params, lr=0.1, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        objective = loss_fn(llh, tree_idxs, l1_regularization_strength=l1_regularization_strength)
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
            loss = loss_fn(llh, tree_idxs, l1_regularization_strength=l1_regularization_strength)
            logger.info("Iteration %d loss: %s", i, loss)

        loss = optimizer.step(closure)
        assert llh.get_Q_params().requires_grad is True

        with torch.no_grad():
            losses.append(loss.item())
            primary_matrix = llh.get_rate_matrix()
            pi = llh.get_root_distribution()
            if torch.isnan(primary_matrix).any() or torch.isnan(pi).any() or loss.isnan():
                raise ValueError(
                    f"Invalid value: Nan in parameters/loss\n {llh.get_Q_params()}\n {llh.get_pi_params()}\n {loss}"
                )

        if losses[-1] <= min(losses) and len(losses) >= 2:
            with torch.no_grad():
                torch.save(llh.state_dict(), f"{output_dir}/state_dict.pth")
                _save_model_dict(llh, model_info, output_dir)

        if do_logging and i % log_iter == 0:
            logger.info("Iteration %d loss: %s", i, loss.item())
            if hasattr(llh, "get_daughter_kernel"):
                logger.info("Daughter kernel:\n%s", llh.get_daughter_kernel())
            else:
                logger.info("Rate matrix:\n%s", primary_matrix)
            logger.info("Root distribution: %s", pi)
            if hasattr(llh, "get_growth_rates"):
                logger.info("Growth rates: %s", llh.get_growth_rates())
            if hasattr(llh, "get_sampling_probability"):
                logger.info("Sampling probability: %s", llh.get_sampling_probability())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("%.4f s/iter", (time.time() - start) / min(i + 1, log_iter))
            start = time.time()

        if len(losses) > 10:
            rel = abs(losses[-1] - losses[-2]) / (abs(losses[-2]) + EPS)
            if rel <= rel_loss_thresh:
                break

    return llh, losses[-1]


def grad_norm(llh):
    return torch.sqrt(sum((p.grad.detach().norm() ** 2 for p in llh.parameters() if p.grad is not None)))

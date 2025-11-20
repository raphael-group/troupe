import torch
import pytest
from ete3 import Tree
from models import PureBirthLikelihoodModel

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_tiny_tree():
    # ((0:0.5,1:0.5):0.3,2:0.8);
    t = Tree("((0:0.5,1:0.5):0.3,2:0.8);", format=1)
    for lf in t:
        lf.add_features(state=int(lf.name))
    return t

def test_gradients_flow_with_rho_lt_1():
    k = 3
    trees = [make_tiny_tree()]
    Q_params = torch.zeros((k, k), device=device)    # start near 0
    pi_params = torch.zeros(k, device=device)        # uniform root prior
    growth_params = torch.zeros(k, device=device)    # lam = softplus ~ 0.693

    model = PureBirthLikelihoodModel(
        trees, num_states=k,
        Q_params=Q_params, pi_params=pi_params, growth_params=growth_params,
        terminal_states=[], num_hidden=0, idx2potency=None,
        device=device, idx2state=None, start_state=None,
        subsampling_rate=0.7,
    )

    # Precompute global cache once per parameter setting
    lam = model.get_growth_rates()              # (k,)
    Q   = model.rate_builder.forward()          # (k,k)
    # Use the pruner on the model and the tensorized time grid
    tg  = model.tree_tens.global_time_grid.to(device=device, dtype=torch.float64)
    model.pruner.prepare_global_cache(tg, lam, Q, rho=model.rho)

    # Forward + backward
    loglik = model(0)
    loss = -loglik
    loss.backward()

    assert model.get_Q_params().grad is not None
    assert model.growth_params.grad is not None
    assert torch.isfinite(model.get_Q_params().grad).all()
    assert torch.isfinite(model.growth_params.grad).all()
    # at least one gradient non-zero
    assert (model.get_Q_params().grad.abs().sum() > 0) or (model.growth_params.grad.abs().sum() > 0)

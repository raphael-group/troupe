"""
Script for optimizing log likelihood objective on given trees.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import CTMCLikelihoodModel, PureBirthLikelihoodModel

import pickle
import time
import warnings
import sys
import math

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000) 

INF = float('inf')
EPS = 1e-30

dtype=torch.float64
torch.set_default_dtype(dtype)

def compute_mle(trees,
                num_states,
                device,
                output_dir,
                l1_regularization_strength=0.0,
                group_lasso_strength=0.0,
                nmp_transition_strength=0.0,
                do_logging=True,
                model_type="PureBirthLikelihoodModel",
                model_info=None):
    """
    Runs optimization method of choice (GD or LBFGS) to learn the parameters of a continuous-time 
    Markov chain (CTMC) model by maximizing the log likelihood objective.
    
    This function initializes the rate matrix parameters (using a differentiable reparameterization
    via the logarithm of a softplus function) and a set of  progenitor parameters. It then trains
    a LogLikelihoodModel instance using either Adam or LFBGS. The loss function is the negative log
    likelihood (summed over all trees) plus along with a regularization penalty. The training loop
    stops early if the change in loss between iterations falls below a specified threshold.
    
    Parameters:
        trees (List[TreeNode]): 
            List of tree objects (e.g., ete3.TreeNode instances) representing the 
            phylogenetic or lineage trees for which the likelihood is computed.
            
        num_states (tuple): 
            A tuple (n_obs, n_progenitors) where:
                - n_obs is the number of observed states.
                - n_progenitors is the number of progenitor states.
            The total number of states is computed as n_obs + n_progenitors.
            
        device (torch.device): 
            The device (CPU or GPU) on which to run the training.
            
        output_dir (str): 
            Directory path where training information (losses, iteration count, learning rate)
            and the trained model's state dictionary will be saved.
            
        regularization_strength (float): 
            The strength of the regularization term applied to the models parameters.
            This term encourages sparsity in the rate matrix and root distribution.
            
        do_logging (bool): 
            If True, training progress (including current loss, rate matrix, and root distribution)
            is printed at regular intervals.
    
    Returns:
        None. The function saves a pickle file containing the training information (losses,
        number of iterations, learning rate) and a PyTorch state dictionary for the trained 
        model into the specified output directory.
    
    Notes:
        - The rate matrix is initialized via an unconstrained reparameterization such that
          Q = softplus(R) - diag(sum(softplus(R), dim=1)), ensuring that each row sums to 0,
          off-diagonals are positive, and diagonal entries are negative.
    """

    llh, loss = _run_lbfgs(trees,
                           num_states,
                           device,
                           output_dir,
                           l1_regularization_strength=l1_regularization_strength,
                           group_lasso_strength=group_lasso_strength,
                           nmp_transition_strength=nmp_transition_strength,
                           do_logging=do_logging,
                           model_type=model_type,
                           model_info=model_info)
    
    #  Save relevant model info to a dict
    model_dict = {
        "rate_matrix": llh.get_rate_matrix(),
        "root_distribution": llh.get_root_distribution()
    }
    if hasattr(llh, "get_growth_rates"):
        model_dict["growth_rates"] = llh.get_growth_rates()
    if not model_info is None:
        for k, v in model_info.items():
            if k not in model_dict:
                model_dict[k] = v
    with open(f"{output_dir}/model_dict.pkl", "wb") as fp:
        pickle.dump(model_dict, fp)
        print(f"Saving model_dict to {output_dir}/model_dict.pkl")

    return llh, loss

def contstant_rate_mle(trees):
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
            

def loss_fn(llh, tree_idxs, l1_regularization_strength=0, group_lasso_strength=0, nmp_transition_strength=0,
            is_numerical_likelihood=False):
    if is_numerical_likelihood:
        lam = llh.get_growth_rates()  
        Q   = llh.get_rate_matrix()
        llh.prepare_numerical_pruner(lam, Q, rho=llh.rho)  # precompute

    loss = 0.0
    for j in tree_idxs:             # TODO: Parallelize this with tree indices
        tree_llh = llh(j)
        loss = loss - tree_llh
    loss = loss / len(tree_idxs)    # TODO: Rethink whether you should be using this normalization
    if l1_regularization_strength > 0:
        loss = loss + sparse_regularization(llh) * l1_regularization_strength
    if group_lasso_strength > 0:
        loss = loss + group_lasso(llh) * group_lasso_strength
    if nmp_transition_strength > 0:
        loss = loss + equal_nmp_transition_regularization(llh) * nmp_transition_strength
    return loss

def sparse_regularization(llh):
    Q = llh.get_rate_matrix()
    mask = ~torch.eye(llh.num_states, dtype=torch.bool, device=Q.device)
    rate_regularization = torch.sum(torch.abs(Q[mask]))
    return rate_regularization

def equal_nmp_transition_regularization(llh):
    # TODO: Only to be used for LRT on these transitions having different rates
    assert "NMPs" in llh.states and "NeuralTube" in llh.states and "Somite" in llh.states
    Q = llh.get_rate_matrix()
    state2idx = llh.state2idx
    nmp_to_nt = Q[state2idx["NMPs"], state2idx["NeuralTube"]]
    nmp_to_somite = Q[state2idx["NMPs"], state2idx["Somite"]]
    return (nmp_to_nt - nmp_to_somite)**2

def group_lasso(llh):
    Q = llh.get_rate_matrix()
    column_l2_norms = torch.linalg.norm(Q, dim=0, ord=2)

    potency_constraints = llh.rate_builder.potency_constraint_mask
    degrees_of_freedom = torch.sum(potency_constraints, dim=0)
    column_weights = 1.0 / torch.sqrt(degrees_of_freedom)
    # Zero out root and terminals because we assume they must have non-zero columns
    column_weights[llh.root_idx] = 0
    column_weights[llh.terminal_idx] = 0

    loss = torch.sum(column_weights * column_l2_norms)
    return loss

def _run_lbfgs(trees,
               num_states,                              # TODO: Change this to two separate args instead of a tuple
               device,
               output_dir,
               l1_regularization_strength=0.0,
               group_lasso_strength=0.0,
               nmp_transition_strength=0.0,
               do_logging=True,
               model_type="PureBirthLikelihoodModel",   # TODO: Deprecate this, we always use PureBirth
               model_info=None):
    
    assert model_info is not None and "idx2potency" in model_info and "idx2state" in model_info

    num_iter = 50
    log_iter = 2
    rel_loss_thresh = 1e-5
    n_obs, n_hidden = num_states
    n_states = n_obs + n_hidden

    losses = []

    start_state = None
    if "start_state" in model_info:
        start_state = model_info["start_state"]
    print("=> Start state:", start_state)

    subsampling_rate = 1.0
    if "subsampling_rate" in model_info:
        subsampling_rate = model_info["subsampling_rate"]
    print("=> Subsampling rate:", subsampling_rate)

    idx2potency = model_info["idx2potency"]
    idx2state = model_info["idx2state"]

    pi_params_init = torch.zeros(n_states, device=device)

    if "growth_params_init" not in model_info:
        lam = contstant_rate_mle(trees)
        growth_params_init = torch.log(torch.exp(torch.ones(n_states, device=device) * lam) - 1)
    else:
        growth_params_init = model_info["growth_params_init"].to(device)
    # print("=> Initializing growth params as:", F.softplus(growth_params_init))

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
    # print("=> Initializing transition rates as:\n", Q_init)

    llh = PureBirthLikelihoodModel(trees,
                                    n_states,
                                    Q_params=Q_params_init,
                                    pi_params=pi_params_init,
                                    growth_params=growth_params_init,
                                    num_hidden=n_hidden,
                                    idx2potency=idx2potency,
                                    device=device,
                                    idx2state=idx2state,
                                    start_state=start_state,
                                    subsampling_rate=subsampling_rate)

    tree_idxs = [i for i in range(len(trees))]
    changeable_params = [p for p in llh.parameters(recurse=True) if p.requires_grad]
    # print("Q_params")
    # print(llh.get_Q_params())

    # NOTE: Add for debugging
    # torch.autograd.set_detect_anomaly(True)

    if subsampling_rate is not None and subsampling_rate < 1.0:
        print("Using default LBFGS")
        optimizer = optim.LBFGS(
            changeable_params,
            lr=1.0,
            max_iter=25,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-12
        )
    else:
        print("Using strong wolfe line search in LBFGS")
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
                            l1_regularization_strength=l1_regularization_strength,
                            group_lasso_strength=group_lasso_strength,
                            nmp_transition_strength=nmp_transition_strength,
                            is_numerical_likelihood=llh.rho<1.0)
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
                        l1_regularization_strength=l1_regularization_strength,
                        group_lasso_strength=group_lasso_strength,
                        nmp_transition_strength=nmp_transition_strength,
                        is_numerical_likelihood=llh.rho<1.0)
            print(i, "loss:", loss)

        loss = optimizer.step(closure)

        assert llh.get_Q_params().requires_grad == True

        with torch.no_grad():
            losses.append(loss.item())
            rate_matrix = llh.get_rate_matrix()
            pi = llh.get_root_distribution()
            if torch.isnan(rate_matrix).any() or torch.isnan(pi).any() or loss.isnan():
                raise ValueError(f"Invalid value: Nan in parameters/loss\n {llh.get_Q_params()}\n {llh.get_pi_params()}\n {loss}")
            
        if losses[-1] <= min(losses) and len(losses) >= 2:
            print(f"Curr loss <= prev best loss: {losses[-1]} vs {min(losses[:-1])}")
            with torch.no_grad():
                file_name = f"{output_dir}/state_dict.pth"
                torch.save(llh.state_dict(), file_name)
                model_dict = {
                    "rate_matrix": llh.get_rate_matrix(),
                    "root_distribution": llh.get_root_distribution()
                }
                if hasattr(llh, "get_growth_rates"):
                    model_dict["growth_rates"] = llh.get_growth_rates()
                if not model_info is None:
                    for k, v in model_info.items():
                        if k not in model_dict:
                            model_dict[k] = v
                with open(f"{output_dir}/model_dict.pkl", "wb") as fp:
                    pickle.dump(model_dict, fp)
                    print(f"Saving model_dict to {output_dir}/model_dict.pkl")

        # if do_logging and i % log_iter == 0:
        #     print(f"{i} loss: {loss.item()}")
        #     print("Rate matrix:")
        #     print(rate_matrix)
        #     print("Root distribution", pi)
        #     if hasattr(llh, "get_growth_rates"):
        #         print("Growth rates:")
        #         print(llh.get_growth_rates())
        #     if torch.cuda.is_available():
        #         torch.cuda.synchronize()
        #     print(f"\t{(time.time() - start) / min(i+1, log_iter):.4f} s/iter")
        #     start = time.time()

        
        if len(losses) > 10:
            rel = abs(losses[-1] - losses[-2]) / (abs(losses[-2]) + EPS)
            if rel <= rel_loss_thresh:
                break
    
    return llh, losses[-1]

    

def grad_norm(llh):
    return torch.sqrt(sum((p.grad.detach().norm()**2 for p in llh.parameters() if p.grad is not None)))
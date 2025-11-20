# tests/test_gradients_exist.py
import torch
from likelihood import _prep_log_tree, log_vec_likelihood_numerical, build_transition_tensor_numerical
from ete3 import Tree

def test_gradients_finite():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    newick = "((0:0.7,1:0.7):0.5,(2:0.4,0:0.4):0.8);"
    t = Tree(newick, format=1)
    for node in t.traverse():
        if node.is_leaf():
            node.state = int(node.name)
    k = 16
    tree = _prep_log_tree(t, k)
    rho = 0.7
    # unconstrained params
    alpha = torch.nn.Parameter(torch.zeros(k, device=device))       # lam = softplus(alpha)
    R     = torch.nn.Parameter(torch.zeros(k, k, device=device))    # Q   = make_Q(R)
    root_logits = torch.zeros(k, device=device)


    # Build constrained inside closure to test autograd over ODEs
    def make_Q(R):
        sp = torch.nn.functional.softplus
        Qoff = sp(R) * (1 - torch.eye(k, device=device))
        Qoff.fill_diagonal_(0.0)
        return Qoff - torch.diag(Qoff.sum(dim=1))

    opt = torch.optim.LBFGS([alpha, R], max_iter=1)

    lam = torch.nn.functional.softplus(alpha) + 1e-12
    Q = make_Q(R)
    assert Q.requires_grad
    assert lam.requires_grad
    TT = build_transition_tensor_numerical([tree], lam, Q, rho)[0]

    assert TT.requires_grad 
    assert TT.grad_fn is not None

    def closure():
        opt.zero_grad(set_to_none=True)
        lam = torch.nn.functional.softplus(alpha) + 1e-12
        Q = make_Q(R)
        logL = log_vec_likelihood_numerical([tree], Q, root_logits, growth_rates=lam, rho=rho)
        loss = -logL
        loss.backward()
        # grads should be finite and non-NaN
        for p in (alpha, R):
            print(p.grad)
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
        return loss

    loss0 = float(closure())
    print(loss0)

from gillespie import simulate_tree_gillespie
from OLD_branching_simulation import simulate_tree
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math
from collections import Counter
np.random.seed(123)


def main():
    # test_tree_length()
    # test_expected_number_of_leaves()
    # test_irreducible()
    # test_fixed_num_leaves()

    # print()
    # print("Testing label proportions...")
    # test_label_proportions()

    # print()
    # print("Testing branch len stats...")
    # test_branch_lens_statistics()

    test_marginal_distribution_over_time()

def test_marginal_distribution_over_time():
    """
    
    """
    growth_rate = 2.0
    Q = np.array(
        [
            [-3.0, 2.0, 0.0, 0.0, 1.0],
            [0.0, -2.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    )
    lam = np.array([2, 1, 0.5, 0.5, 0.5]) * growth_rate

    num_trials = 100000
    times = [0.1, 0.7, 1.0, 1.6]
    max_int = 100 * int(max(times) * (num_trials+1))
    seeds = np.random.choice(range(max_int), size=max_int, replace=False)

    for time in times:
        total_type_counts = Counter()
        for i in range(num_trials):
            seed = seeds[int(time * num_trials * 10 + i)]
            tree = simulate_tree(Q, lam, 0, T=time, seed=seed)
            leaf_types = [leaf.state for leaf in tree.get_leaves()]
            total_type_counts.update(leaf_types)

        average_types = [total_type_counts[i]/num_trials for i in range(len(total_type_counts))]

        expected_types = expm((Q + np.diag(lam)) * time)[0]
        
        print("Time:  ", time)
        print("Avg type counts:      ", average_types)
        print("Expected type counts: ", expected_types)
        print()

def simulate_tree_with_k_leaves(Q, lam, start_state, k, seed, num_tries=1000):
    """
    Uses rejection sampling to simulate trees until 

    WARNING: This can take a long time...
    """

    # NOTE: Use expected time to reach n nodes
    avg_rate = np.mean(lam)
    time = np.log(k) / avg_rate

    for i in range(num_tries):
        tree = simulate_tree(Q, lam, start_state, time, seed+i)
        if len(tree.get_leaves()) == k:
            return tree
        
    print(f"Error: Could not simulate {k}-leaf tree after {num_tries} tries")
    return None


def test_branch_lens_statistics():
    """
    Applies a variety of branch length-based statistical tests to the pure-birth tree simulation.

    References:
    - [1]: "Testing macro-evolutionary models using incomplete molecular phylogenies"
    - [2]: "Distribution of branch lengths and phylogenetic diversity under homogeneous speciation models"
    """
    # NOTE: [1] does not actually pertain to multiype and conditions on number of leaves (not time)

    def gamma(tree):
        # # Implements the gamma statistic. Under a pure birth model, gamma ~ N(0, 1).
        # # see [1] for the full details.
        # # 1) collect internal-node heights from present=0 up to root
        # heights = [n.get_farthest_leaf()[1] for n in tree.traverse() if not n.is_root() and not n.is_leaf()]
        # heights.sort()
        # # 2) compute inter-node intervals
        # times = [0.0] + heights
        # gs = np.flip(np.diff(times))
        # n_tips = len(gs) + 2
        # # print(n_tips, len(tree.get_leaves()))
        # assert n_tips == len(tree.get_leaves())
        # # 3) compute T and the average cumulative time S
        # idx_list = np.arange(len(gs))+2
        # T = (gs * idx_list).sum()
        # cums = np.cumsum((gs * idx_list)[:-1])
        # S_bar = cums.mean()
        # # 4) gamma
        # denom = T * math.sqrt(1.0/(12*(n_tips-2)))
        # gamma = (S_bar - (T/2.0)) / denom
        # return gamma.item()
        N = len(tree.get_leaves())
        heights = [n.get_farthest_leaf()[1] for n in tree.traverse() if not n.is_leaf()]
        heights.sort()
        diffs = np.diff([0.0] + heights)      # gives [bt[1], bt[2]-bt[1], ...]
        g = diffs[::-1]                       # same as rev(c(bt[1], diff(bt)))
        k = np.arange(2, 2+len(g))            # [2,3,...,N]
        T  = np.dot(g, k)                     # same as ST
        cum = np.cumsum(g * k)[:-1]
        S_bar = np.mean(cum)                  # same as stat
        denom = T * np.sqrt(1/(12*(N-2)))        # same as s
        gamma = (S_bar - T/2) / denom         # same as (stat - m)/s
        return gamma
    

    num_trials = 500
    a = 10
    b = 1
    l = 1 
    plotting_out_dir = "/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/tests/tmp/"

    Q = np.array([[-a, a], [b, -b]])

    for l in [1, 5, 10]:
        lam = l * np.ones(len(Q))
        gammas = []
        branch_lens = []
        skipped = 0
        for i in range(num_trials):
            seed = i * 100000 + i * l
            # tree = simulate_tree_gillespie(Q, lam, 0, T=1000, max_leaves=60, seed=seed)
            # tree = simulate_tree(Q, lam, 0, T=8/l, seed=seed)
            tree = simulate_tree_with_k_leaves(Q, lam, 0, k=20, seed=seed)

            gammas.append(gamma(tree))
            branch_lens += [n.dist for n in tree.traverse() if not n.is_root()]
        
        print(f"Skipped {skipped}")
        
        mean_gamma = sum(gammas) / len(gammas)
        print("Avg gamma val:       ", mean_gamma)
        plt.hist(gammas)
        plt.xlabel("Gamma statistic")
        plt.ylabel("Count")
        plt.title("Pure-birth distribution test (should be normal)")
        plt.savefig(f"{plotting_out_dir}/gamma_stat_test_t={num_trials}_l={l}.png")
        plt.clf()

        mean_branch_len = sum(branch_lens) / len(branch_lens)
        expected_branch_len = 1/(2*l)                           # See [2] for details
        print("Avg branch len:      ", mean_branch_len)
        print("Expected branch len: ", expected_branch_len)
        plt.hist(branch_lens, density=True)
        plt.xlabel("Branch lengths")
        plt.ylabel("Proportion")
        plt.title("Branch lengths")
        plt.savefig(f"{plotting_out_dir}/branch_lens_t={num_trials}_l={l}.png")
        plt.clf()

        assert np.isclose(mean_branch_len, expected_branch_len, rtol=1e-1)

    # Plot of num_trials normal distribution for reference
    z = np.random.normal(0, 1, size=num_trials)
    print("Avg z val:           ", z.mean())
    plt.hist(z)
    plt.xlabel("Standard normal")
    plt.ylabel("Count")
    plt.title("Histogram of standard normal samples")
    plt.savefig(f"{plotting_out_dir}/std_normal_{num_trials}.png")
    
    assert abs(z.mean()) < 0.1     # P(Fail) ~ 0
    assert abs(mean_gamma) < 0.1   # If implementation is correct, P(Fail) ~ 0



def test_tree_length():
    Q = np.array(
        [
            [-1.0, 1.0],
            [1.0, -1.0]
        ]
    )
    lam = np.array([4, 4])

    for i in range(2):
        for time in [0.5, 1, 1.5]:
            # tree = simulate_tree_gillespie(Q, lam, 0, T=time)
            tree = simulate_tree(Q, lam, 0, T=time, seed=int(i*9373*time))
            print(f"Time: {time}, Num leaves: {len(tree.get_leaves())}")

            # Check that the tree length equals the time
            dist_to_closest_leaf = tree.get_closest_leaf()[1] + tree.dist
            print("\tdist_to_closest_leaf:", dist_to_closest_leaf)
            dist_to_furthest_leaf = tree.get_farthest_leaf()[1] + tree.dist
            print("\tdist_to_furthest_leaf:", dist_to_furthest_leaf)
            assert np.isclose(dist_to_closest_leaf, time)
            assert np.isclose(dist_to_furthest_leaf, time)


def test_expected_number_of_leaves():
    num_trials = 10000    # NOTE: This needs to be quite large to approximate expeted val
    growth_rate = 2
    time = 1

    Q = np.array(
        [
            [-1.0, 0.25, 0.25, 0.25, 0.25],
            [0.25, -1.0, 0.25, 0.25, 0.25],
            [0.25, 0.25, -1.0, 0.25, 0.25],
            [0.25, 0.25, 0.25, -1.0, 0.25],
            [0.25, 0.25, 0.25, 0.25, -1.0]
        ]
    )
    lam = np.ones(len(Q)) * growth_rate

    expected_num_leaves = np.exp(growth_rate * time) #* 2    # NOTE: We need x2 factor if we start process at the first birth
    num_leaves = []

    for i in range(num_trials):
        if i % (num_trials//10) == 0:
            print(i)
        # tree = simulate_tree_gillespie(Q, lam, 0, T=time)
        tree = simulate_tree(Q, lam, 0, T=time, seed=i)
        num_leaves.append(len(tree.get_leaves()))

    avg_num_leaves = sum(num_leaves) / len(num_leaves)

    print("average number of leaves:  ", avg_num_leaves)
    print("expected number of leaves: ", expected_num_leaves)
    assert np.isclose(avg_num_leaves, expected_num_leaves, rtol=1e-1)

def test_irreducible():
    num_trials = 10
    growth_rate = 3
    time = 5
    Q = np.array(
        [
            [-growth_rate, growth_rate/2, growth_rate/2],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]
    )
    lam = np.array([growth_rate/2, 1e-8, 1e-8])

    for i in range(num_trials):
        # tree = simulate_tree_gillespie(Q, lam, 0, T=time)
        tree = simulate_tree(Q, lam, 0, T=time, seed=i)
        leaf_types = set([leaf.state for leaf in tree.get_leaves()])
        print(tree.get_ascii(attributes=["name", "dist"]))
        print("Num leaves:  ", len(tree.get_leaves()))
        print("Leaf labels: ", leaf_types)

        # Checking that
        # - all leaves are terminal types
        # - the terminal types never transition back to progenitor
        # - the root is a progenitor type
        terminal_types = [1, 2]
        progenitor_types = [0]
        assert all([leaf_type in terminal_types for leaf_type in leaf_types])
        for node in tree.traverse():
            assert node.state is not None
            if not node.is_root() and node.up.state in terminal_types:
                assert node.state not in progenitor_types

def test_fixed_num_leaves():
    totipotent_label = 4
    time = 1000
    growth_rate = 10
    Q = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, -8.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, -10.0, 4.0],
            [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, -10.0]
        ]
    )
    lam = np.array([growth_rate, growth_rate, growth_rate, growth_rate, \
                    growth_rate, growth_rate, growth_rate])

    for i in range(10):
        for num_leaves in [500]:
            tree = simulate_tree_gillespie(Q, lam, totipotent_label, T=time, max_leaves=num_leaves, seed=i)
            assert num_leaves == len(tree.get_leaves())

            # Check that the tree is ultrametric
            dist_to_closest_leaf = tree.get_closest_leaf()[1] + tree.dist
            dist_to_furthest_leaf = tree.get_farthest_leaf()[1] + tree.dist

            # leaf_types = set([leaf.state for leaf in tree.get_leaves()])
            counter = Counter()
            for leaf_node in tree.get_leaves():
                counter[leaf_node.state] += 1
            leaf_types = [(label, count) for label, count in counter.items()]
            leaf_types.sort(key=lambda el: el[0])
            print(f"closest: {dist_to_closest_leaf:2g} furthest: {dist_to_furthest_leaf:2g} \t leaves: {leaf_types}")
            assert np.isclose(dist_to_closest_leaf, dist_to_furthest_leaf)

def test_label_proportions():
    num_trials = 100000
    totipotent_label = 4
    time = 1
    growth_rate = 3
    Q = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -10.0, 5.0, 5.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, -10.0, 0.0],
            [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, -10.0]
        ]
    )
    lam = np.array([growth_rate, growth_rate, growth_rate, growth_rate, \
                    growth_rate, growth_rate, growth_rate])

    # NOTE: On average, all terminal types should have the same proportion
    counter = Counter()
    total_terminals = 0
    terminal_labels = [0, 1, 2, 3]
    for i in range(num_trials):
        for num_leaves in [100]:
            tree = simulate_tree(Q, lam, totipotent_label, T=time, seed=i)
            for leaf_node in tree.get_leaves():
                counter[leaf_node.state] += 1
                if leaf_node.state in terminal_labels:
                    total_terminals += 1
        if i % (num_trials//10) == 0:
            print(i)
            
    print(counter)
    proportion_terminals = {label: count / total_terminals \
                            for label, count in counter.items() if label in terminal_labels}
    print(proportion_terminals)
    assert np.isclose(counter[0]/total_terminals, 0.25, atol=1e-2)
    assert np.isclose(counter[1]/total_terminals, 0.25, atol=1e-2)
    assert np.isclose(counter[2]/total_terminals, 0.25, atol=1e-2)
    assert np.isclose(counter[3]/total_terminals, 0.25, atol=1e-2)
    


if __name__ == "__main__":
    main()
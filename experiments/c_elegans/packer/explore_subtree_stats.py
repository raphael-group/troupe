# The goal of this script is to explore the lineage tree and node labels to try
# to find a good coarse-graining of the types

import ete3
from collections import Counter

WORKING_DIR = "/Users/william_hs/Desktop/Projects/troupe/experiments/c_elegans/packer"
TREE_PATH = f"{WORKING_DIR}/raw_data/lineage_tree.nwk"
NODE_LABEL_PATH = f"{WORKING_DIR}/raw_data/tree_node_labels.txt"

TYPES_TO_REMOVE = [
    'XXX', 'T'
]

tree = ete3.Tree(TREE_PATH, format=3)

# print("=" * 60)
# print(tree.name)
# for child in tree.get_children():
#     print(child.name)
#     for grand_child in child.get_children():
#         print("\t", grand_child.name)
# print("=" * 60)

num_nodes = len(list(tree.traverse()))

name2label = {}
with open(NODE_LABEL_PATH, "r") as fp:
    for line in fp:
        arr = line.strip().split(' ')
        if len(arr) <= 1 or arr[1] in TYPES_TO_REMOVE:
            continue
        name = arr[0]
        label = arr[1]
        assert name not in name2label
        name2label[name] = label

labels = set(name2label.values())

### Label nodes in the tree

num_nodes_w_label = 0
for node in tree.traverse(strategy="postorder"):
    if node.name in name2label:
        label = name2label[node.name]
        num_nodes_w_label += 1
    else:
        label = None
    node.add_feature("label", label)

print(f"{num_nodes_w_label} / {num_nodes} = {num_nodes_w_label / num_nodes * 100}% nodes have labels")

### Annotate each node with the labels of its descendants

node2descendant_labels = {}
for node in tree.traverse(strategy="postorder"):
    descendant_label_counts = Counter() # include self-label
    if node.label is not None:
        descendant_label_counts[node.label] += 1
    for child in node.children:
        descendant_label_counts += node2descendant_labels[child]
    node2descendant_labels[node] = descendant_label_counts

### Sanity check

for node in tree.traverse(strategy="postorder"):
    # if node.name is not None and not node.is_leaf():
    #     print("\t", node2descendant_labels[node])
    if node.is_root():
        assert len(node2descendant_labels[node]) == len(labels)
    if node.is_leaf():
        assert len(node2descendant_labels[node]) == 1 or node.label == None


### Calculate the potency of each type

label2potency = {label: set() for label in labels}

for node in tree.traverse(strategy="postorder"):
    if node.label is None:
        continue
    desc_labels = node2descendant_labels[node].keys()
    label2potency[node.label] = label2potency[node.label].union(desc_labels)

terminal_labels = []
for i, label in enumerate(labels):
    if len(label2potency[label]) == 1:
        terminal_labels.append(label)
    # else:
    #     potency_text = ", ".join(label2potency[label])
    #     print("\t", label, "\t", potency_text)

print()
print(f"Dataset has {len(terminal_labels)} terminal types")
print("Counting all terminals across leaves...")

terminal2count = Counter()
for node in tree.get_leaves():
    if node.label in terminal_labels:
        terminal2count[node.label] += 1

print("\t", "leaf count", "  \t", "label")
for label, count in terminal2count.items():
    print("\t", count, "\t", label)

# Is there a hierarchy among the progenitor types?
# If so, you can group based on the observed potency?

label2terminal_potency = {}
for label, potency in label2potency.items():
    terminal_potency = {label for label in potency if label in terminal_labels}
    label2terminal_potency[label] = terminal_potency

print()
print("Mapping labels to terminal potencies")
for label, terminal_potency in label2terminal_potency.items():
    if label not in terminal_labels:
        print(label)
        potency_list = list(terminal_potency)
        if len(potency_list) == 0:
            print("\t Terminals:    None")
        else:
            potency_list.sort()
            print("\t Terminals:   ", ", ".join(potency_list))

        non_terminal_potency = [lab for lab in label2potency[label] if lab not in terminal_labels and lab != label]
        if len(non_terminal_potency) == 0:
            print("\t Progenitors: None")
        else:
            potency_list.sort()
            print("\t Progenitors:", ", ".join(non_terminal_potency))


# For each subtree (3 deep), how many terminal types are there in that subtree?

def get_terminals_in_subtree(tree):
    label2potency = {}
    for node in tree.traverse():
        if node.is_leaf():
            potency = set([node.label])
        else:
            potency = set([child.label for child in node.get_children()])
        if None in potency:
            potency.remove(None)
        if node.label not in label2potency:
            label2potency[node.label] = set()
        label2potency[node.label] = label2potency[node.label].union(potency)
    
    terminals = set([label for label, potency in label2potency.items() if len(potency) <= 1 and label is not None])
    return terminals


def get_observed_potency(node, terminals):
    leaves = node.get_leaves()
    num_terminal_leaves = 0
    potency = set()
    for leaf in leaves:
        if leaf.label in terminals:
            potency.add(leaf.label)
            num_terminal_leaves += 1
    return potency, num_terminal_leaves, len(leaves)

print()
print("All observed potencies and their leaf counts")
print("t_leaf\t leaf\t pot_len\t name\t potency")
for node in tree.traverse("levelorder"):
    if node.is_leaf():
        continue
    terminals = get_terminals_in_subtree(node)
    potency, num_term_leaves, num_leaves = get_observed_potency(node, terminals)
    potency_abrv = [state[0:3] if len(state) > 4 else state for state in potency]
    if num_term_leaves < 30:
        continue
    if len(potency) > 25 or len(potency) < 3:
        continue
    if num_term_leaves / num_leaves < 0.3:
        continue
    # print(terminals)
    print(num_term_leaves, "\t", num_leaves, "\t", len(potency), "\t", node.name, f"\t[{', '.join(potency_abrv)}]")
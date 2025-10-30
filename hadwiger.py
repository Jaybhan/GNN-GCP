import itertools
import networkx as nx
from wolframclient.language import wl          # Wolfram symbolic constructors
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt

def generate_graph_max_indep_set_2(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Go through all 3-node combinations
    for u, v, w in itertools.combinations(G.nodes, 3):
        # Check if the 3 nodes are mutually non-adjacent
        if not G.has_edge(u, v) and not G.has_edge(u, w) and not G.has_edge(v, w):
            # Randomly add one of the three possible edges to destroy the independence
            edge = random.choice([(u, v), (u, w), (v, w)])
            G.add_edge(*edge)

    return G

G = generate_graph_max_indep_set_2(8)
print(nx.to_numpy_array(G, dtype=int))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='black')


def adjacency_matrix_to_graph(adj_matrix):
    """Convert a (possibly asymmetric) 0/1 matrix to an undirected NetworkX graph."""
    n = len(adj_matrix)
    G = nx.Graph()
    G.add_nodes_from(range(n))                 # keep isolated vertices
    for i in range(n):
        for j in range(i + 1, n):              # look only above the diagonal
            if adj_matrix[i][j] or adj_matrix[j][i]:
                G.add_edge(i, j)
    return G


def is_connected_subgraph(G, nodes):
    return nx.is_connected(G.subgraph(nodes))


def is_complete_between(G, groups):
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if not any(G.has_edge(u, v) for u in groups[i] for v in groups[j]):
                return False
    return True


def all_partitions(collection, k):
    """Yield every way to split *collection* into exactly *k* non-empty blocks,
    disregarding order of the blocks."""
    if k == 1:
        yield [collection]
        return
    if len(collection) < k:
        return
    first = collection[0]
    for smaller in all_partitions(collection[1:], k - 1):
        yield [[first]] + smaller
    for partition in all_partitions(collection[1:], k):
        for i in range(len(partition)):
            yield partition[:i] + [partition[i] + [first]] + partition[i + 1:]


def has_k_minor(G, k):
    nodes = list(G.nodes)
    # try every subset of vertices with at least k elements
    for r in range(k, len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            sub_nodes = list(subset)
            for partition in all_partitions(sub_nodes, k):
                if all(is_connected_subgraph(G, part) for part in partition) and is_complete_between(G, partition):
                    print(partition)
                    return True
    return False


def hadwiger_number(adj_matrix):
    G = adjacency_matrix_to_graph(adj_matrix)
    for k in range(G.number_of_nodes(), 0, -1):
        if has_k_minor(G, k):
            return k
    return 1





print(hadwiger_number(nx.to_numpy_array(G, dtype=int)))
plt.title("Graph with Maximum Independent Set Size = 2")
plt.show()




# Start Wolfram session
session = WolframLanguageSession()

# Get only graphs that have a Hadwiger number defined
graph_names = session.evaluate(
    wlexpr('Select[GraphData[], Function[g, ValueQ[GraphData[g, "HadwigerNumber"]]]]'))

print(f"✅ Found {len(graph_names)} graphs with Hadwiger numbers.\n")

# Iterate through some (change to more if you want)
for name in graph_names:  # remove [:10] to run all (might be slow!)
    # Get Hadwiger number
    try:
        hadwiger = session.evaluate(wl.GraphData(name, "HadwigerNumber"))


        # Get adjacency matrix and convert to numpy
        adj = session.evaluate(wl.Normal(wl.AdjacencyMatrix(wl.GraphData(name))))
        adj_array = np.array(adj)
        if len(adj_array)<2 or len(adj_array[0])<2 or len(adj_array)>8:
            continue

        # Print info
        print(f"📌 {name}")
        print("   Adjacency Matrix:")
        print(adj_array)
        print(f"   Hadwiger Number: {hadwiger}")
        print(f"   Hadwiger Test: {hadwiger_number(adj_array)}")

        print("-" * 40)

        if hadwiger!=hadwiger_number(adj_array):
            print("I HAVE FAILED :C")
            break
    except:
        pass

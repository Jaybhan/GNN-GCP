from typing import Any
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import numpy as np
import os
from hadwiger import hadwiger_number, generate_graph_max_indep_set_2, generate_random_graph
import networkx as nx
import random
import matplotlib.pyplot as plt


#session = WolframLanguageSession()

output_dir = "hadwiger_random_graph_augmented_test"
os.makedirs(output_dir, exist_ok=True)

"""
graph_names = session.evaluate(
    wlexpr('Select[GraphData[], Function[g, ValueQ[GraphData[g, "HadwigerNumber"]]]]'))

print(f"✅ Found {len(graph_names)} graphs with Hadwiger numbers.\n")
"""
# Function to write the .graph file in GNN-GCP format
def write_graph_file(name, adj_matrix, hadwiger_number, out_path):
    n = adj_matrix.shape[0]

    with open(out_path, "w") as f:
        f.write(f"TYPE : {name}\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_DATA_FORMAT: EDGE_LIST\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")

        f.write("EDGE_DATA_SECTION\n")
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 1:
                    f.write(f"{i} {j}\n")
        f.write("-1\n")

        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row = "\t".join(str(int(adj_matrix[i][j])) for j in range(n))
            f.write(row + "\n")

        f.write("DIFF_EDGE\n")
        f.write("0 0\n")

        f.write("CHROM_NUMBER\n")
        f.write(f"{hadwiger_number}\n")

        f.write("EOF\n")


def split_edge(G, partition):
    u, v = random.choice(list(G.edges))
    new_node = max(G.nodes) + 1

    G.remove_edge(u, v)
    G.add_node(new_node)
    G.add_edge(u, new_node)
    G.add_edge(new_node, v)

    for block in partition:
        if u in block or v in block:
            block.append(new_node)
            break
    return G, partition


def add_nodes(G, partition):
    new_node = max(G.nodes) + 1
    G.add_node(new_node)

    target_block = random.choice(partition)
    attach_to = random.sample(target_block, k=random.randint(1, max(1, len(target_block)//2)))
    for v in attach_to:
        G.add_edge(v, new_node)

    target_block.append(new_node)
    return G, partition

def increase_had_by_1(G, had, partition):
    new_node = max(G.nodes) + 1
    G.add_node(new_node)
    for block in partition:
        G.add_edge(random.choice(block), new_node)
    for i in range(random.randint(0, len(G.nodes)-had)):
        v = random.choice(list(G.nodes))
        G.add_edge(v, new_node)
    partition.append([new_node])
    return G, partition


def main():
    count=0
    while True:
        G=generate_random_graph(11)
        print(count)
        G_adj=nx.to_numpy_array(G, dtype=int)
        had, partition=hadwiger_number(G_adj)
        if had==1:
            continue

        for i in range(random.randint(1, 10)):
            func=random.choice(["split_edge", "add_nodes", "increase_had"])
            if func=="split_edge":
                G, partition = split_edge(G, partition)
            if func=="add_nodes":
                G, partition = add_nodes(G, partition)
            if func=="increase_had":
                G, partition = increase_had_by_1(G, had, partition)
                had+=1
        """
        print(len(G.nodes))
        if len(G.nodes)<=11:
            test_had, test_had_nodes=hadwiger_number(nx.to_numpy_array(G, dtype=int))
            if test_had!=had:
                print(f"FAILURE! {test_had} != {had}")
                break
        else:
            print("too big")
            continue
        """


        file_name = f"graph_{count}"
        out_path = os.path.join(output_dir, file_name)
        write_graph_file("graph_{i}", G_adj, had, out_path)
        count+=1


        if count==100:
            print("Reached 100 graphs")
            break

if __name__ == "__main__":
    main()

"""
for name in graph_names:
    try:
        hadwiger = session.evaluate(wl.GraphData(name, "HadwigerNumber"))
        adj = session.evaluate(wl.Normal(wl.AdjacencyMatrix(wl.GraphData(name))))
        adj_array = np.array(adj)

        print(f"📌 {name} - Hadwiger Number: {hadwiger}")

        file_name = f"{name}.graph".replace(" ", "_")
        out_path = os.path.join(output_dir, file_name)

        if type(hadwiger) == int:
            write_graph_file(name, adj_array, hadwiger, out_path)
        else:
            print(f"⚠️ Failed for {name}: {hadwiger}")

    except Exception as e:
        print(f"⚠️ Failed for {name}: {e}")

# Done
session.terminate()
print("\n✅ All .graph files written to:", output_dir)
"""

"""
def main():
    count=0
    while True:
        G=generate_random_graph(12)
        G=nx.to_numpy_array(G, dtype=int)
        print(G)
        had=hadwiger_number(G)
        print(count)
        if had<100:

            file_name = f"graph_{count}"
            out_path = os.path.join(output_dir, file_name)

            write_graph_file("graph_{i}", G, had, out_path)
            count+=1
        if count==1000:
            print("Reached 1000 graphs")
            break
"""

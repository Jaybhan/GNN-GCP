from wolframclient.language import wl          # Wolfram symbolic constructors
from wolframclient.evaluation import WolframLanguageSession
"""
with WolframLanguageSession() as session:
    # 1. Build the graph entirely on the Python side
    g = wl.Graph([
        wl.UndirectedEdge(1, 2),
        wl.UndirectedEdge(2, 3),
        wl.UndirectedEdge(3, 1),
        wl.UndirectedEdge(3, 4)
    ])

    # 2. Send the HadwigerNumber expression to Wolfram **for evaluation**
    hadwiger = session.evaluate(wl.HadwigerNumber(g))

    # 3. Print the numeric result
    print("Hadwiger number of the graph:", hadwiger)


from wolframclient.language import wl
from wolframclient.evaluation import WolframLanguageSession

with WolframLanguageSession() as s:
    s.evaluate(wl.Needs("Combinatorica`"))          # ① load package

    g = s.evaluate(wl.FromUnorderedPairs([[1,2],     # ② build a Combinatorica graph
                                          [2,3],
                                          [3,1],
                                          [3,4]]))

    h = s.evaluate(wl.HadwigerNumber(g))            # ③ call the function
    print(h)                                        # -> 3
"""
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Start Wolfram session
session = WolframLanguageSession()

# Get only graphs that have a Hadwiger number defined
graph_names = session.evaluate(
    wlexpr('Select[GraphData[], Function[g, ValueQ[GraphData[g, "HadwigerNumber"]]]]'))

print(f"✅ Found {len(graph_names)} graphs with Hadwiger numbers.\n")

# Iterate through some (change to more if you want)
for name in graph_names[:2]:  # remove [:10] to run all (might be slow!)
    # Get Hadwiger number
    hadwiger = session.evaluate(wl.GraphData(name, "HadwigerNumber"))


    # Get adjacency matrix and convert to numpy
    adj = session.evaluate(wl.Normal(wl.AdjacencyMatrix(wl.GraphData(name))))
    adj_array = np.array(adj)

    # Print info
    print(f"📌 {name}")
    print(f"   Hadwiger Number: {hadwiger}")
    print("   Adjacency Matrix:")
    print(adj_array)
    print("-" * 40)

session.terminate()

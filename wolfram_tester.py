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
dictionary={}
for name in graph_names[:2]:  # remove [:10] to run all (might be slow!)
    # Get Hadwiger number
    hadwiger = session.evaluate(wl.GraphData(name, "HadwigerNumber"))

    if hadwiger in dictionary.keys():
        dictionary[hadwiger]+=1
    else:
        dictionary[hadwiger]=1

    # Get adjacency matrix and convert to numpy
    adj = session.evaluate(wl.Normal(wl.AdjacencyMatrix(wl.GraphData(name))))
    adj_array = np.array(adj)

    # Print info
    print(f"📌 {name}")
    print(f"   Hadwiger Number: {hadwiger}")
    print("   Adjacency Matrix:")
    print(adj_array)
    print("-" * 40)
print(dictionary)

session.terminate()

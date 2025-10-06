from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import numpy as np
import os

# Start Wolfram Language session
session = WolframLanguageSession()

# Directory to save .graph files
output_dir = "hadwiger_data"
os.makedirs(output_dir, exist_ok=True)

# Get all graphs with Hadwiger number defined
graph_names = session.evaluate(
    wlexpr('Select[GraphData[], Function[g, ValueQ[GraphData[g, "HadwigerNumber"]]]]'))

print(f"✅ Found {len(graph_names)} graphs with Hadwiger numbers.\n")

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

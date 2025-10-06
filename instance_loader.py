import os, sys
import random
import numpy as np
from functools import reduce

class InstanceLoader(object):

    def __init__(self,path):
        self.path = path
        self.filenames = [ path + '/' + x for x in os.listdir(path) ]
        random.shuffle(self.filenames)
        self.reset()
    #end

    def get_instances(self, n_instances):
        instances_generated = 0

        while instances_generated < n_instances:
            # Read graph from file
            Ma, chrom_number, diff_edge = read_graph(self.filenames[self.index])
            f = self.filenames[self.index]

            # Generate multiple color count examples for this graph
            # Try colors from 2 to chrom_number + 3
            for colors in range(2, chrom_number + 4):
                if instances_generated >= n_instances:
                    break

                cn_exists_val = 1 if colors >= chrom_number else 0
                yield Ma, colors, f, cn_exists_val
                instances_generated += 1

            if self.index + 1 < len(self.filenames):
                self.index += 1
            else:
                self.reset()
        #end
    #end

    def create_batch(instances):

        # n_instances: number of instances
        n_instances = len(instances)

        # n_vertices[i]: number of vertices in the i-th instance
        n_vertices  = np.array([ x[0].shape[0] for x in instances ])
        # n_edges[i]: number of edges in the i-th instance
        n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
        # n_colors[i]: number of colors in the i-th instance
        n_colors = np.array( [x[1] for x in instances])
        # cn_exists[i]: colorability target for the i-th instance
        cn_exists = np.array( [x[3] for x in instances])
        # total_vertices: total number of vertices among all instances
        total_vertices  = sum(n_vertices)
        # total_edges: total number of edges among all instances
        total_edges     = sum(n_edges)
        # total_colors: total number of colors among all instances
        total_colors = sum(n_colors)

        # Compute matrices M, MC
        # M is the adjacency matrix
        M              = np.zeros((total_vertices,total_vertices))
        # MC is a matrix connecting each problem nodes to its colors candidates
        MC = np.zeros((total_vertices, total_colors))

        for (i,(Ma,colors,f,cn_exists_val)) in enumerate(instances):
            # Get the number of vertices (n) and edges (m) in this graph
            n, m, c = n_vertices[i], n_edges[i], n_colors[i]
            # Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
            n_acc = sum(n_vertices[0:i])
            m_acc = sum(n_edges[0:i])
            c_acc = sum(n_colors[0:i])
            #Populate MC
            MC[n_acc:n_acc+n,c_acc:c_acc+c] = 1

            # Get the list of edges in this graph
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Populate M
            for e,(x,y) in enumerate(edges):
                if Ma[x,y] == 1:
                  M[n_acc+x,n_acc+y] = M[n_acc+y,n_acc+x] = 1
                #end if
            #end for
        #end for
        return M, n_colors, MC, cn_exists, n_vertices, n_edges, f
    #end

    def get_batches(self, batch_size):
        print(f"DEBUG: Total files: {len(self.filenames)}")
        print(f"DEBUG: Expected batches: {len(self.filenames) // batch_size}")
        for i in range( len(self.filenames) // batch_size ):
            print(f"DEBUG: Generating batch {i}")
            instances = list(self.get_instances(batch_size))
            print(f"DEBUG: Instances: {instances}")
            yield InstanceLoader.create_batch(instances)
        #end
    #end

    def get_test_batches(self, batch_size, total_instances):
        for i in range( total_instances ):
            instances = list(self.get_instances(batch_size))
            yield InstanceLoader.create_batch(instances)
        #end
    #end

    def reset(self):
        random.shuffle(self.filenames)
        self.index = 0
    #end
#end

def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)

        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end while

        # Parse diff edge
        while 'DIFF_EDGE' not in line: line = f.readline();
        diff_edge = [ int(x) for x in f.readline().split() ]

        # Parse target cost
        while 'CHROM_NUMBER' not in line: line = f.readline();
        chrom_number = int(f.readline().strip())

    #end
    return Ma,chrom_number,diff_edge
#end

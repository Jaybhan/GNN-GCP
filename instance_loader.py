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
        print("DEBUG: get_instances called for {} instances, current index={}".format(n_instances, self.index))
        instances_generated = 0

        while instances_generated < n_instances:
            print("DEBUG: instances_generated={}, reading file index={}".format(instances_generated, self.index))

            if self.index >= len(self.filenames):
                print("DEBUG: ERROR - index {} >= len(filenames) {}".format(self.index, len(self.filenames)))
                break

            print("DEBUG: Reading file: {}".format(self.filenames[self.index]))

            try:
                # Read graph from file
                Ma, chrom_number, diff_edge = read_graph(self.filenames[self.index])
                f = self.filenames[self.index]
                print("DEBUG: Successfully read file, chrom_number={}".format(chrom_number))
            except Exception as e:
                print("DEBUG: ERROR reading file {}: {}".format(self.filenames[self.index], e))
                # Skip this file and move to next
                if self.index + 1 < len(self.filenames):
                    self.index += 1
                else:
                    self.reset()
                continue

            # Generate multiple color count examples for this graph
            # Try colors from 2 to chrom_number + 3
            print("DEBUG: Generating color examples from 2 to {}".format(chrom_number + 3))
            for colors in range(2, chrom_number + 4):
                if instances_generated >= n_instances:
                    print("DEBUG: Reached target instances, breaking")
                    break

                cn_exists_val = 1 if colors >= chrom_number else 0
                print("DEBUG: Yielding instance {} with colors={}, cn_exists={}".format(instances_generated, colors, cn_exists_val))
                yield Ma, colors, f, cn_exists_val
                instances_generated += 1

            print("DEBUG: Moving to next file, current index={}".format(self.index))
            if self.index + 1 < len(self.filenames):
                self.index += 1
            else:
                print("DEBUG: Reached end of files, resetting")
                self.reset()
        #end
    #end

    def create_batch(instances):
        print("DEBUG: create_batch called with {} instances".format(len(instances)))

        # n_instances: number of instances
        n_instances = len(instances)
        print("DEBUG: n_instances = {}".format(n_instances))

        # n_vertices[i]: number of vertices in the i-th instance
        n_vertices  = np.array([ x[0].shape[0] for x in instances ])
        print("DEBUG: n_vertices = {}".format(n_vertices))

        # n_edges[i]: number of edges in the i-th instance
        n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
        print("DEBUG: n_edges = {}".format(n_edges))

        # n_colors[i]: number of colors in the i-th instance
        n_colors = np.array( [x[1] for x in instances])
        print("DEBUG: n_colors = {}".format(n_colors))

        # cn_exists[i]: colorability target for the i-th instance
        cn_exists = np.array( [x[3] for x in instances])
        print("DEBUG: cn_exists = {}".format(cn_exists))

        # total_vertices: total number of vertices among all instances
        total_vertices  = sum(n_vertices)
        print("DEBUG: total_vertices = {}".format(total_vertices))

        # total_edges: total number of edges among all instances
        total_edges     = sum(n_edges)
        print("DEBUG: total_edges = {}".format(total_edges))

        # total_colors: total number of colors among all instances
        total_colors = sum(n_colors)
        print("DEBUG: total_colors = {}".format(total_colors))

        print("DEBUG: Creating matrices M and MC...")
        # Compute matrices M, MC
        # M is the adjacency matrix
        M              = np.zeros((total_vertices,total_vertices))
        print("DEBUG: Created M matrix with shape {}".format(M.shape))

        # MC is a matrix connecting each problem nodes to its colors candidates
        MC = np.zeros((total_vertices, total_colors))
        print("DEBUG: Created MC matrix with shape {}".format(MC.shape))

        print("DEBUG: Starting matrix population loop...")
        for (i,(Ma,colors,f,cn_exists_val)) in enumerate(instances):
            print("DEBUG: Processing instance {} with shape {}".format(i, Ma.shape))

            # Get the number of vertices (n) and edges (m) in this graph
            n, m, c = n_vertices[i], n_edges[i], n_colors[i]
            # Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
            n_acc = sum(n_vertices[0:i])
            m_acc = sum(n_edges[0:i])
            c_acc = sum(n_colors[0:i])

            print("DEBUG: Instance {}: n={}, m={}, c={}, n_acc={}, c_acc={}".format(i, n, m, c, n_acc, c_acc))

            #Populate MC
            MC[n_acc:n_acc+n,c_acc:c_acc+c] = 1
            print("DEBUG: Populated MC for instance {}".format(i))

            # Get the list of edges in this graph
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))
            print("DEBUG: Instance {} has {} edges".format(i, len(edges)))

            # Populate M
            for e,(x,y) in enumerate(edges):
                if Ma[x,y] == 1:
                  M[n_acc+x,n_acc+y] = M[n_acc+y,n_acc+x] = 1
                #end if
            #end for
            print("DEBUG: Populated M for instance {}".format(i))
        #end for

        print("DEBUG: Matrix population complete, returning batch...")
        return M, n_colors, MC, cn_exists, n_vertices, n_edges, f
    #end

    def get_batches(self, batch_size):
        print("DEBUG: Total files: {}".format(len(self.filenames)))
        print("DEBUG: Expected batches: {}".format(len(self.filenames) // batch_size))
        for i in range( len(self.filenames) // batch_size ):
            print("DEBUG: Generating batch {}".format(i))
            instances = list(self.get_instances(batch_size))
            print("DEBUG: Instances: {}".format(instances))
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

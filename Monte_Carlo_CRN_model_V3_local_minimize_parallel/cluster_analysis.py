
# 2019-05-01  YJ
# given a neighbor list, e.g. [[1,2],[2,0], [0, 1]]  where 0 is connected to 1 and 2, 1 is connected to 0 and 2, 2 is connected to 0 and 1
# calculate the number of nth neighbor that is far from a central atom but can be reach by n step via a shortest path
# implement BSF search
# output a list of number of nth neighbor [] index means nth neighbor, 0th is the central atom itself.

# Python3 Program to print BFS traversal
# from a given source vertex. BFS(int s)
# traverses vertices reachable from s.


# This class represents a directed graph
# using adjacency list representation
# This code is adapted from Neelam Yadav

from collections import defaultdict
from basic_function import bondlen

class Graph:

    # Constructor
    def __init__(self):
        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # add all edge from neighbor list
    def add_all_Edge(self, neighborlist):
        for i in neighborlist:
            for j in neighborlist[i]:
                self.graph[i].append(j)

    # Function to print a BFS of graph
    def BFS(self, s):
        # Mark all the vertices as not visited
        visited = [False] * (len(self.graph))

        # Create a queue for BFS
        queue = []

        # Mark the source node as
        # visited and enqueue it
        queue.append(s)
        visited[s] = True

        while queue:
            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

    def gen_nth_neighbor(self, mysystem, frame, maximum_neighbor_order = 8):

        lx, ly, lz = mysystem[frame].L[0], mysystem[frame].L[1], mysystem[frame].L[2]
        res = [1.0]
        distance = [0.0]
        count_bond_to_next_order = [0.0]
        if bool(self.graph) is False:
            print("neighbor list is empty!")
            return res, distance
        for _ in range(1, maximum_neighbor_order + 1):
            res.append(0.0)
            distance.append(0.0)
            count_bond_to_next_order.append(0.0)

        visited = set()
        si = []
        # the following 4 lines just help debug
        for i in self.graph:
            si.append(i)
        si.sort()
        for n in si:
            # print("n=",n)
            visited.clear()
            # Mark all the vertices as not visited
            # Create a queue for BFS
            queue = []
            # Mark the source node as
            # visited and enqueue it
            queue.append(n)
            visited.add(n)
            order = 1
            size = 1

            next_level_node = set()

            while queue and order <= maximum_neighbor_order:
                # Dequeue a vertex from
                # queue and print it
                next_level_node.clear()
                count_bond = 0
                count = 0
                dist = 0.0
                for _ in range(size):
                    s = queue.pop(0)
                    for i in self.graph[s]:
                        if i in next_level_node:
                            count_bond += 1
                        if not i in visited:
                            count_bond += 1
                            queue.append(i)
                            visited.add(i)
                            next_level_node.add(i)
                            count += 1
                            dist += bondlen(mysystem[frame].myatom[n], mysystem[frame].myatom[i], lx, ly, lz)
                res[order] += count
                count_bond_to_next_order[order] += count_bond
                if count > 0:
                    distance[order] += dist / count
                size, order = count, order + 1
        print("there are:", len(self.graph), "central atoms")
        for n in range(1, maximum_neighbor_order + 1):
            res[n] /= len(self.graph)
            distance[n] /= len(self.graph)
            count_bond_to_next_order[n] /= len(self.graph)

        return res, distance, count_bond_to_next_order

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dwave.system import DWaveSampler, EmbeddingComposite
import neal
import greedy
G = nx.Graph()

points1 = [] #placement of lidars for one side
points2 = [] #placement of lidars for other side
points3 = [] #vertices to cover

for i in range(1, 6, 2):
    point = (i, 1)
    points1.append(point)

for i in range(1, 6, 2):
    point = (i, 4)
    points2.append(point)

for i in range(0, 7):
    for j in range(2, 4):
        point = (i, j)
        points3.append(point)

G.add_nodes_from(points1 + points2 + points3)

#edge that represents scope between lidar placement and vertices to cover
for node1 in points1:
  for j in range (node1[1]+1, node1[1]+3):
    if(j==node1[1]+1):
      for i in range(node1[0]-2, node1[0]+3):
        if (i,j) in points3:
          G.add_edge(tuple(node1), (i,j))
    if(j== node1[1]+2):
      for i in range(node1[0]-1, node1[0]+2):
        if (i,j) in points3:
          G.add_edge(tuple(node1), (i,j))
for node2 in points2:
  for j in range (node2[1]-2,node2[1]):
    if(j== node2[1]-2):
      for i in range(node2[0]-1, node2[0]+2):
        if (i,j) in points3:
          G.add_edge(tuple(node2), (i,j))
    if(j==node2[1]-1):
      for i in range(node2[0]-2, node2[0]+3):
        if (i,j) in points3:
          G.add_edge(tuple(node2), (i,j))


#Draw the graph

pos = {node: node for node in G.nodes()}
nx.draw(G, pos, with_labels=True)

plt.show()

B=1
Q = defaultdict(int)

#constraints favorisate every edge activation
for node in points3:
  degree = len(G.edges(node))
  for anode in G.neighbors(node):
    Q[(node, anode)] -= 1*B
    #minimisation function
    Q[(anode, anode)] += 1


sampler = neal.SimulatedAnnealingSampler()
#sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type__eq': 'pegasus'}))
#sampleset = sampler.sample(Q, num_reads=10)
sampleset = sampler.sample_qubo(Q,
                               chain_strength=1,
                               num_reads=50,
                               label='QAR-Lab',
                               return_embedding=True)
#solver = greedy.SteepestDescentSolver()
#solver = neal.SimulatedAnnealingSampler()
#sampleset = solver.sample_qubo(Q, num_reads=5)
print(sampleset)
#print("\nEmbedding found:\n", sampleset.info['embedding_context']['embedding'])

print("\nSampleset:")
print(sampleset)
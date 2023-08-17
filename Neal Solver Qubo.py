import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import neal
import greedy
from PuboOnGraph import Graph,listCovering,listLidar
G = nx.Graph()

points1 = [] #placement of lidars for one side
points2 = [] #placement of lidars for other side
points3 = [] #vertices to cover


#Draw the graph

pos = {node: node for node in G.nodes()}


B=1
# Q = defaultdict(int)

# #constraints favorisate every edge activation
# for node in points3:
#   degree = len(G.edges(node))
#   for anode in G.neighbors(node):
#     Q[(node, anode)] -= 1*B
#     #minimisation function
#     Q[(anode, anode)] += 1


sampler = neal.SimulatedAnnealingSampler()
solver = greedy.SteepestDescentSolver()
#sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type__eq': 'pegasus'}))
#sampleset = sampler.sample(Q, num_reads=10)
with open("matrice_QUBO_RR.npy", 'rb') as file:
    Q = np.load(file)
print("taille de la matrice =  ", len(Q[0]))
sampleset = sampler.sample_qubo(Q,
                               chain_strength=10,
                               num_reads=100000,
                               label='QAR-Lab',
                               return_embedding=True)

#solver = neal.SimulatedAnnealingSampler()
#sampleset = solver.sample_qubo(Q, num_reads=5)

#print("\nEmbedding found:\n", sampleset.info['embedding_context']['embedding'])

def showGraph():

    G = nx.Graph()
    points1 = [] #placement of lidars for one side
    points2 = [] #placement of lidars for other side
    points3 = [] #vertices to cover

    pointsL = listLidar
    points3 = listCovering
    
    print("\nSampleset:")
    print(sampleset.slice(0,10))
    S = []
    k = sampleset.first.sample
    for i,x in enumerate(pointsL):
      if k[i]:
        S.append((x[0], x[1]))

    G.add_nodes_from(points1 + points2 + points3)
    G = Graph

    pos = {node: node for node in G.nodes()} 
    nx.draw(G, pos, with_labels=False, node_size =40 )
    actE = []
    actN = []
    for node in S:
      for edge in G.edges(node):
        actE.append(edge)
      for node in G.neighbors(node):
        actN.append(node)
  
    pos = {node: node for node in G.nodes()}

    nx.draw(G, pos, with_labels=False,node_color = 'grey', node_size=40)
    nx.draw_networkx_nodes(G, pos, listLidar, node_color = 'purple', node_size= 40)
    nx.draw_networkx_nodes(G, pos, S, node_color='red', node_size=40)
    nx.draw_networkx_nodes(G, pos, actN, node_color='blue', node_size=40)
    nx.draw_networkx_nodes(G, pos, S, node_color='red', node_size=40)
    nx.draw_networkx_edges(G, pos, actE, edge_color='red', node_size=40)
    plt.show()
showGraph()

# for j in range (1,3,1):
#     n_temp_iter= 1000
#     listEnergy = []
#     listCoverNAct = []
#     listNbrLidarAct = []
#     for i in range (0, 10):
#         k = sampleset.first.sample
#         for i,x in enumerate(pointsL):
#           if k[i]:
#             S.append((x[0], x[1]))
#         listCoverN = []
#         for node in S:
#             for node2 in Graph.neighbors(node):
#                 listCoverN.append(node2)
                
#         listEnergy.extend(energies)
#         listCoverNAct.append(len(listCoverN))
#         listNbrLidarAct.append(len(S))

#     print("runtime = ",runtime)
        
#     fig,ax = plt.subplots(3)
#     ax[0].set_title("PUBO ran on LoopedGraph")
#     ax[0].set_ylabel("Energies")
#     ax[0].set_xticklabels(["SA 1000 iter 10"] )
#     ax[0].boxplot(listEnergy)
#     ax[1].set_ylabel("Nbr Activated Lidars")
#     ax[1].set_xticklabels(["SA 1000 iter 10"] )
#     ax[1].boxplot(listCoverNAct)
#     ax[2].set_ylabel("Nbr Activate Covering Nodes")
#     ax[2].set_xticklabels(["SA 1000 iter 10"] )
#     ax[2].boxplot(listNbrLidarAct)
#     plt.show()
    
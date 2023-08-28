import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import neal
import greedy
from PuboOnGraph import Graph, listCovering, listLidar, listEnergy as lEnerPUBO, listCoverNAct as lCovPUBO, listUnCoverAct as lUnCovPUBO, ListNbrLidarAct as lLidPUBO
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
    Q1 = np.load(file)
with open("matrice_QUBO_JS.npy", 'rb') as file:
    Q2 = np.load(file)
with open("matrice_QUBO_JR.npy", 'rb') as file:
    Q3 = np.load(file)
with open("matrice_QUBO_Equal.npy", 'rb') as file:
    Q4 = np.load(file)
with open("matrice_QUBO_Inequal.npy", 'rb') as file:
    Q5 = np.load(file)
print("taille de la matrice RR =  ", len(Q1[0]))
print("taille de la matrice JS =  ", len(Q2[0]))
print("taille de la matrice JR =  ", len(Q3[0]))
print("taille de la matrice Equal =  ", len(Q4[0]))
print("taille de la matrice Inequal =  ", len(Q5[0]))
listQUBO = [Q1, Q2, Q3, Q4, Q5]
def sampleset(Q):
  sampleset = sampler.sample_qubo(Q,
                    chain_strength=10,
                    num_reads = 1000,
                    label='QAR-Lab',
                    return_embedding=True)
  first_sample_energy = sampleset.record.energy[0]
  return first_sample_energy, list(sampleset.first.sample.values())

#solver = neal.SimulatedAnnealingSampler()
#sampleset = solver.sample_qubo(Q, num_reads=5)

#print("\nEmbedding found:\n", sampleset.info['embedding_context']['embedding'])

def showGraph():
    G = nx.Graph()
    pointsL = listLidar
    points3 = listCovering
    lEner = []
    lCov = []
    lUnCov = []
    lLid = []
    # for j in range (1,4,1):
    #   listEnergy = []
    #   listCoverNAct = []
    #   listNbrLidarAct = []
    #   for i in range (0, 50):
    #       k, f = sampleset(num_reads=100**j)
    #       S = []
    #       for i,x in enumerate(pointsL):
    #         if f[i]:
    #           S.append((x[0], x[1]))
    #       listCoverN = []
    #       for node in S:
    #         for node2 in Graph.neighbors(node):
    #           listCoverN.append(node2)
    #       listEnergy.append(k)
    #       listCoverNAct.append(len(listCoverN))
    #       listNbrLidarAct.append(len(S))
    #   lEner.append(listEnergy)
    #   lCov.append(listCoverNAct)
    #   lLid.append(listNbrLidarAct)
    for qubo in (listQUBO):
      listEnergy = []
      listCoverNAct = []
      listUnCoverAct = []
      listNbrLidarAct = []
      for iter in range (0, 10):
          k, f = sampleset(qubo)
          S = []
          for i,x in enumerate(pointsL):
            if f[i]:
              S.append((x[0], x[1]))
          listCoverN = []
          for node in S:
            for node2 in Graph.neighbors(node):
              listCoverN.append(node2)
          listEnergy.append(k)
          listCoverNAct.append(len(listCoverN))
          listUnCoverAct.append((len(listCovering) - len(listCoverN)))
          listNbrLidarAct.append(len(S))
      lEner.append(listEnergy)
      lCov.append(listCoverNAct)
      lUnCov.append(listUnCoverAct)
      lLid.append(listNbrLidarAct)
    lEner.append(lEnerPUBO)
    lCov.append(lCovPUBO)
    lUnCov.append(lUnCovPUBO)
    lLid.append(lLidPUBO)
    fig,ax = plt.subplots(4)
    ax[0].set_title("LoopedGraph")
    ax[0].set_ylabel("Energies")
    print(lEner)
    ax[0].boxplot(lEner, positions = [0, 1, 2, 3, 4, 5])
    ax[1].set_ylabel("Nbr Activated Lidars")
    #ax[1].set_xticklabels(["neal 100 iter 10", "neal 10000 iter 10", "neal 1000000 iter 10"] )
    ax[1].boxplot(lLid, positions = [0, 1, 2, 3, 4, 5])
    ax[2].set_ylabel("Nbr Activate Covering Nodes")
    ax[2].boxplot(lCov, positions = [0, 1, 2, 3, 4, 5])
    ax[3].set_ylabel("Nbr Unactivate Covering Nodes")
    ax[3].set_xticklabels(["QUBO #1", "QUBO #2", "QUBO #3", "QUBO #4", "QUBO #5", "PUBO"])
    ax[3].boxplot(lUnCov, positions = [0, 1, 2, 3, 4, 5])
    plt.show()
    
    # print("\nSampleset:")
    # print(sampleset.slice(0,10))
    
    # k = sampleset.first.sample
    # for i,x in enumerate(pointsL):
    #   if k[i]:
    #     S.append((x[0], x[1]))

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

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import neal
import greedy
from PuboOnGraph import Graph, listCovering, listLidar, listEnergy as lEnerPUBO, listCoverNAct as lCovPUBO, listUnCoverAct as lUnCovPUBO, ListNbrLidarAct as lLidPUBO
#from PuboOnSqArea import Graph, listCovering, listPositionLidar as listLidar, listEnergy as lEnerPUBO, listCoverNAct as lCovPUBO, listUnCoverAct as lUnCovPUBO, ListNbrLidarAct as lLidPUBO


points1 = [] #placement of lidars for one side
points2 = [] #placement of lidars for other side
points3 = [] #vertices to cover

B=1

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
listQUBO = [Q1, Q2, Q3, Q5, Q4]
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
    #lEner = []
    lCov = []
    lUnCov = []
    lLid = []
    percentTotCovN = []
    listDict = []
    _Dict = {}
    for qubo in (listQUBO):
      for node in points3:
        _Dict[node] = []
      #   print(_Dict)
      # print(len(qubo))
      #listEnergy = []
      listCoverNAct = []
      listUnCoverAct = []
      listNbrLidarAct = []
      percentCoveredNode=[]
      
      for iter in range (0, 10):
          k, f = sampleset(qubo)
          S = []
          for i,x in enumerate(pointsL):
            # print(len(pointsL))
            # print(i)
            if f[i]:
              S.append((x[0], x[1]))
          listCoverN = []
          for node in S:
            for node2 in Graph.neighbors(node):
              if node2 not in listCoverN:
                listCoverN.append(node2)
          #listEnergy.append(k)
          listCoverNAct.append(len(listCoverN))
          listUnCoverAct.append((len(listCovering) - len(listCoverN)))
          percentCoveredNode.append(len(listCoverN) / len(listCovering))
          listNbrLidarAct.append(len(S))
          _dict = {}
          for node in points3:
            _dict[node] = 0
          for node in points3:
            for lidar in Graph.neighbors(node):
              if lidar in S:
                _dict[node] += 1
              
          for node in points3:
            _Dict[node].append(_dict[node])
      listDict.append(_Dict)        
      #lEner.append(listEnergy)
      lCov.append(listCoverNAct)
      lUnCov.append(listUnCoverAct)
      lLid.append(listNbrLidarAct)
      percentTotCovN.append(percentCoveredNode)
    # print(listDict)
    #lEner.append(lEnerPUBO)
    lCov.append(lCovPUBO)
    lUnCov.append(lUnCovPUBO)
    lLid.append(lLidPUBO)
    print(lCovPUBO)
    print(len(lCov))
    percentTotCovN.append(percentCoveredNode)
    fig,ax = plt.subplots(2)
    print(len(lLid), len(percentTotCovN))
    ax[0].set_title("Random Graph")
    ax[0].set_ylabel("percentage of Covered Node")
    #print(lEner)
    ax[0].boxplot(percentTotCovN, positions = [0, 1, 2, 3, 4, 5])
    ax[1].set_ylabel("Nbr Activated Lidars")
    #ax[1].set_xticklabels(["neal 100 iter 10", "neal 10000 iter 10", "neal 1000000 iter 10"] )
    ax[1].boxplot(lLid, positions = [0, 1, 2, 3, 4, 5])
    # ax[2].set_ylabel("Nbr Activate Covering Nodes")
    # ax[2].boxplot(lCov, positions = [0, 1, 2, 3])
    # ax[3].set_ylabel("Nbr Unactivate Covering Nodes")
    # ax[3].set_xticklabels(["QUBO #1", "QUBO #2", "QUBO #3", "QUBO #4", "QUBO #5", "PUBO"])
    # ax[3].boxplot(lUnCov, positions = [0, 1, 2, 3])
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
    nx.draw(G, pos, with_labels=False, node_size =40)
    actE = []
    actN = []
    for node in S:
      for edge in G.edges(node):
        actE.append(edge)
      for node in G.neighbors(node):
        if node not in actN:
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
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import List,  Union, Callable
import random
import neal
from time import time




# Simulated Annealing for PUBO

def sa_solve(
    f: Callable,
    n: int,
    n_temp_iter: int = 100,
    n_iter: int = 1,
    temp: float = 50,
    warm_start: Union[List, np.ndarray] = None,
) -> (List, List, float):
    """
    Standard simulated annealing solver

    :param f: cost function
    :param n: problem instance size (i.e., length of solution bitstring)
    :param n_iter: number of runs
    :param n_temp_iter: number of mutations
    :param temp: starting temperature
    :param warm_start: warm start solution vector
    :return: solution with samples, energies and times
    """
    samples = []
    energies = []
    indices = list(range(0, n))

    # keep track of wallclock time
    start_time = time()

    for _ in range(n_iter):
        # define start vector
        if warm_start is None:
            x = np.array([0] * n) #start with the full graph
        else:
            x = np.array(warm_start)

        # evaluate start vectorx
        curr, curr_eval = x, f(x)
        best, best_eval = curr, curr_eval

        for i in range(n_temp_iter):
            # flip a random bit to generate a neighbor
            candidate = np.copy(curr)
            flip_pos = random.sample(indices, 1)
            candidate[flip_pos] = int(not candidate[flip_pos])

            # evaluate new vector
            candidate_eval = f(candidate)

            # keep best vector
            if candidate_eval <= best_eval:
                best, best_eval = candidate, candidate_eval

            # update temperature according to Metropolis–Hastings algorithm
            diff = candidate_eval - curr_eval
            t = temp / float(i + 1)

            metropolis_eval = math.exp(-diff / t)

            # base new mutations on new vector
            if diff <= 0 or random.random() < metropolis_eval:
                curr, curr_eval = candidate, candidate_eval

        samples.append(best.tolist())
        energies.append(best_eval)
    runtime = time() - start_time
    return samples, energies, runtime

# Running the PUBO

def runningSAPUBO(energyF, listLidar, listCovering, G):
    '''
    For each QUBO, run ten times the simulated annealing and return the number of lidars activated,
    the number of vertices covered, the number of vertices not covered and the percentage of vertices covered
    for each run
    
    Parameters: energyF the function to minimize
    Returns: listCoverNAct List of the covering node activate, 
    listUnCoverAct List of the covering node unactivate,  
    listNbrLidarAct list of number of lidar activate, 
    percentCoveredNode list of percentage of vertices covered
    '''
    listNbrLidarAct = []
    listCoverNAct = []
    listUnCoverAct = []
    percentCoveredNode = []
    n_temp_iter= 1000
    for i in range (0, 10):
        samples, energies, runtime = sa_solve(energyF, len(listLidar), n_temp_iter)
        S = []
        for i in range(len(samples[0])):
            if samples[0][i]:
                S.append(listLidar[i])
        listCoverN = []
        for node in S:
            for node2 in G.neighbors(node):
                if node2 not in listCoverN:
                    listCoverN.append(node2)
        listCoverNAct.append(len(listCoverN))
        listUnCoverAct.append((len(listCovering) - len(listCoverN)))
        percentCoveredNode.append(len(listCoverN) / len(listCovering))
        listNbrLidarAct.append(len(S))
    actE = []
    actN = []
    for node in S:
        for edge in G.edges(node):
            actE.append(edge)
        for node2 in G.neighbors(node):
            actN.append(node2)
    return listCoverNAct, listUnCoverAct, listNbrLidarAct, percentCoveredNode

# Simulating annealing for QUBO

def getSamplerQUBO():
    '''
    Get the sampler and the list of QUBO matrix
    
    Parameters: None
    Returns: sampler the sampler, listQUBO the list of QUBO matrix
    '''
    sampler = neal.SimulatedAnnealingSampler()
    #sampler = greedy.SteepestDescentSolver()
    with open("ressources/matrix/matrix_QUBO_RR.npy", 'rb') as file:
        Q1 = np.load(file)
    with open("ressources/matrix/matrix_QUBO_JS.npy", 'rb') as file:
        Q2 = np.load(file)
    with open("ressources/matrix/matrix_QUBO_JR.npy", 'rb') as file:
        Q3 = np.load(file)
    with open("ressources/matrix/matrix_QUBO_Equal.npy", 'rb') as file:
        Q4 = np.load(file)
    with open("ressources/matrix/matrix_QUBO_Inequal.npy", 'rb') as file:
        Q5 = np.load(file)
    listQUBO = [Q1, Q2, Q3, Q5, Q4]
    return sampler, listQUBO
    
def sampleset(Q, sampler):
    '''
    Run the simulated annealing for a QUBO matrix
    
    Parameters: Q the QUBO matrix, sampler the sampler
    Returns: first_sample_energy the energy of the first sample,
    list(sampleset.first.sample.values()) the list of the first sample
    '''
    sampleset = sampler.sample_qubo(Q,
                        chain_strength=10,
                        num_reads = 1000,
                        label='QAR-Lab',
                        return_embedding=True)
    first_sample_energy = sampleset.record.energy[0]
    return first_sample_energy, list(sampleset.first.sample.values())

def runningSAQUBO(sampler, listQUBO, listLidar, listCovering, G):
    '''
    Run ten times the simulated annealing for each QUBO matrix
    
    Parameters: sampler the sampler, listQUBO the list of QUBO matrix
    Returns: lCov list of the covering node activate,
    lUnCov list of the covering node unactivate,
    lLid list of number of lidar activate,
    PercentTotCovN list of percentage of vertices covered
    '''
    lCov = []
    lUnCov = []
    lLid = []
    percentTotCovN = []
    listDict = []
    _Dict = {}
    for qubo in (listQUBO):
      for node in listCovering:
        _Dict[node] = []
      listCoverNAct = []
      listUnCoverAct = []
      listNbrLidarAct = []
      percentCoveredNode=[]
      
      for iter in range (0, 10):
          k, f = sampleset(qubo, sampler)
          S = []
          for i,x in enumerate(listLidar):
            if f[i]:
              S.append((x[0], x[1]))
          listCoverN = []
          for node in S:
            for node2 in G.neighbors(node):
              if node2 not in listCoverN:
                listCoverN.append(node2)
          listCoverNAct.append(len(listCoverN))
          listUnCoverAct.append((len(listCovering) - len(listCoverN)))
          percentCoveredNode.append(len(listCoverN) / len(listCovering))
          listNbrLidarAct.append(len(S))
          _dict = {}
          for node in listCovering:
            _dict[node] = 0
          for node in listCovering:
            for lidar in G.neighbors(node):
              if lidar in S:
                _dict[node] += 1
              
          for node in listCovering:
            _Dict[node].append(_dict[node])
      listDict.append(_Dict)        
      lCov.append(listCoverNAct)
      lUnCov.append(listUnCoverAct)
      lLid.append(listNbrLidarAct)
      percentTotCovN.append(percentCoveredNode)
    return lCov, lUnCov, lLid, percentTotCovN
  
def showGraph( lCovPUBO, lUnCovPUBO, lLidPUBO, percentTotCovNP,lCov, lUnCov, lLid, percentTotCovN):
    '''
    Plot the boxplot of the number of lidars activated 
    and the percentage of vertices covered for each QUBO
    '''
    lCov.append(lCovPUBO)
    lUnCov.append(lUnCovPUBO)
    lLid.append(lLidPUBO)
    percentTotCovN.append(percentTotCovNP)
    fig,ax = plt.subplots(2)
    ax[0].set_title("Random Graph")
    ax[0].set_ylabel("percentage of Covered Node")
    ax[0].boxplot(percentTotCovN, positions = [0, 1, 2, 3, 4, 5])
    ax[1].set_ylabel("Nbr Activated Lidars")
    ax[1].boxplot(lLid, positions = [0, 1, 2, 3, 4, 5])
    ax[1].set_xticklabels(["RR", "JS", "JR", "In", "Eq", "PUBO"])
    plt.show()


    
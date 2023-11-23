from src.Graph import runGraph
from src.AllQUBOPUBO import runAllQUBOPUBO
from src.Annealer import runningSAPUBO, runningSAQUBO, getSamplerQUBO, showGraph

if __name__ == "__main__":
    '''
    Run the graph, the QUBO, the PUBO and the simulated annealing
    SHow the result with boxplot
    '''
    
    # Generate a random graph, some instantiation could not work, 
    # not hesitate to run again or change parameters of randomGraph()
    runGraph()
    #Generate the QUBO and the PUBO depending of the random graph
    energyF, listLidar, listCovering, G = runAllQUBOPUBO() 
    #Run the simulated annealing for the PUBO
    listCoverNActP, listUnCoverActP, ListNbrLidarActP, percentTotCovP = runningSAPUBO(energyF, listLidar, listCovering, G) 
    #Run the simulated annealing for the QUBO
    listCoverNActQ, listUnCoverActQ, ListNbrLidarActQ, percentTotCovNQ = runningSAQUBO(*getSamplerQUBO(),  listLidar, listCovering, G) 
    #Show the result with boxplot
    showGraph(listCoverNActP, listUnCoverActP, ListNbrLidarActP, percentTotCovP, 
              listCoverNActQ, listUnCoverActQ, ListNbrLidarActQ, percentTotCovNQ)
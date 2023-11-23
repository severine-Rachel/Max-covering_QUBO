import os
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.problems import QuadraticProgram
import networkx as nx
import matplotlib.pyplot as plt
import json
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import math

def createGraph():
    G = nx.Graph()
    return G

def importGraph():
    with open('ressources/graph.json') as mon_fichier:
        graph = json.load(mon_fichier)
    return graph

def generateGraph(graph, G):
    '''
    Generate the graph with the lidars and the vertices to cover
    
    Parameters: G (graph), graph (dict)
    Returns: pointsL a list of the possible lidar position, points3 a list of the vertices to cover
    '''
    pointsL = []
    points3 = []
    for i in graph['listLidar']:
        pointsL.append((i[0], i[1]))
    for i in graph['listCovering']:
        points3.append((i[0], i[1]))
    for i, j in graph['edge']:
        G.add_edge((i[0], i[1]),( j[0], j[1]))
    return pointsL, points3

def defineModel():
    m = Model(name='BMW') 
    return m

def QUBO_Equal(m, pointsL, points3, G):
    x = m.binary_var_dict(pointsL, name='x')
    m.objective_expr = sum(x[i] for i in pointsL)
    m.objective_sense = 'min'
    for node in points3:
        m.add_constraint(1 == sum(x[v] for v in G.neighbors(node)))
    m.export_as_lp(basename="ressources/lpFiles/Equal", path=os.path.abspath(""))

def QUBO_Inequal(m, pointsL, points3, G):
    x = m.binary_var_dict(pointsL, name='x')
    m.objective_expr = sum(x[i] for i in pointsL)
    m.objective_sense = 'min'
    for node in points3:
        m.add_constraint(1 <= sum(x[v] for v in G.neighbors(node)))
    m.export_as_lp(basename="ressources/lpFiles/Inequal", path=os.path.abspath(""))

def QUBO_JR(m, pointsL, points3, G):
    x = m.binary_var_dict(pointsL, name='x')
    m.objective_expr = sum(x[i] for i in pointsL)
    m.objective_sense = 'min'
    y = {node:m.binary_var_list(G.degree(node), name='z_'+str(node)) for node in points3}
    for node in points3:
        m.add_constraint(sum(y[node]) >= 1)
    for node in points3:
        m.add_constraint(sum(y[node]) == sum(x[v] for v in G.neighbors(node)))
    m.export_as_lp(basename="ressources/lpFiles/JR", path=os.path.abspath(""))

def QUBO_JS(m, pointsL, points3, G):
    x = m.binary_var_dict(pointsL, name='x')
    m.objective_expr = sum(x[i] for i in pointsL)
    m.objective_sense = 'min'
    ub = []
    for node in points3:
        ub.append(G.degree[node])
    y = m.integer_var_dict(points3, name='y',lb=1, ub=ub)
    for node in points3:
        m.add_constraint(y[node] == sum(x[v] for v in G.neighbors(node)))
    m.export_as_lp(basename="ressources/lpFiles/JS", path=os.path.abspath(""))

def QUBO_RR(m, pointsL, points3, G):
    x = m.binary_var_dict(pointsL, name='x')
    m.objective_expr = sum(x[i] for i in pointsL) #minimize the placement of lidars
    m.objective_sense = 'min'
    y = m.binary_var_dict(G.edges, name='y')
    for node in points3:
        var =[]
    for i in G.edges(node):
        if i in y:
                var.append(y[i])
        elif (i[1], i[0]) in y:
            var.append(y[i[1],i[0]])
        m.add_constraint(sum(var) >= 1)
    for node in pointsL:
        var =[]
        for i in G.edges(node):
            if i[0] == node:
                if i in y:
                    var.append(y[i]- x[node])
                elif (i[1], i[0]) in y:
                    var.append(y[i[1],i[0]]- x[node])
            m.add_constraint(sum(var) == 0)
    m.export_as_lp(basename="ressources/lpFiles/RR", path=os.path.abspath(""))
    
def Matrix():
    '''
    Create the matrix of all the QUBO problems
    
    Parameters: None (read the lp file of all QUBO)
    Returns: None (save all the matrix in a .npy file)
    '''
    QUBONames = ["RR","JS","JR","Equal","Inequal"]
    for index in range (0, 5):
        quadratic_program = QuadraticProgram()
        quadratic_program.read_from_lp_file('ressources/lpFiles/' +QUBONames[index]+'.lp')
        conv = QuadraticProgramToQubo()
        qp = conv.convert(quadratic_program)
        matrix = qp.objective.quadratic.coefficients.asformat("array")
        diagonal = qp.objective.linear.coefficients.asformat("array")
        for i in range (len(matrix)):
            for j in range(len(matrix[i])):
                if j == i:
                    matrix[i][j] += diagonal[0][j]
        matrix2 = np.zeros((int(math.sqrt(len(matrix))+1)**2,int(math.sqrt(len(matrix))+1)**2))
        matrix2[:len(matrix),:len(matrix)] = matrix
        np.save('ressources/matrix/matrix_QUBO_'+QUBONames[index]+'.npy',matrix2)

def PUBO(listPositionLidar, listCovering, Graph):
    '''
    Determine our PUBO formualtion depending of the relation 
    between the lidars and the vertices to cover (if at least 
    one lidar is in activate in the neightborhood of a vertex,
    the energy is minimize for this vertex, do exactly the same 
    for each vertex)
    
    Parameters: listPositionLidar (list), listCovering (list), Graph (graph)
    Returns: a function to execute with the simulated annealing
    '''
    def f(x):
        summation = 0
        L = 10
        summation = int(sum([x[i] for (i,v) in enumerate(listPositionLidar)]))
        for node in listCovering:
            product = int(math.prod([1 - x[i] for (i,v) in enumerate(listPositionLidar) if (v in Graph.neighbors(node))])) * L
            summation += product
        return summation
    return(f)

def runAllQUBOPUBO():
    '''
    Create the graph, generate the graph with the lidars and the vertices to cover,
    define the model, create the QUBO formulation, create the matrix of all the QUBO problems,
    determine the PUBO formulation
    
    Parameters: None
    Returns: energyF (energy), listLidar (list), listCovering (list), G (graph)
    '''
  
    G = createGraph()
    schemeGraph = importGraph()
    listLidar, listCovering = generateGraph(schemeGraph, G)
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, node_size=40)
    nx.draw_networkx_nodes(G, pos, listLidar, node_color = 'purple', node_size= 40)
    plt.show()
    m = defineModel()
    QUBO_Equal(m, listLidar, listCovering, G)
    QUBO_Inequal(m, listLidar, listCovering, G)
    QUBO_JR(m, listLidar, listCovering, G)
    QUBO_JS(m, listLidar, listCovering, G)
    QUBO_RR(m, listLidar, listCovering, G)
    Matrix()
    energyF = PUBO(listLidar, listCovering, G)
    return energyF, listLidar, listCovering, G


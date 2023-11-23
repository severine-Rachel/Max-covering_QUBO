import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import math
import numpy as np
from typing import Tuple, List,  Union, Callable, NewType
import random
import time
from time import time

Node = NewType("node",Tuple[float, float])
Edge = NewType("edge",Tuple[Node, Node])
Edge_Angle_List = NewType("Edge_Angle_List",List[Tuple[Edge, float]])

Gephi = nx.Graph()
Graph = nx.Graph()
listNodeLidar = []

def randomGraph():
    '''
    Create a random graph
    '''
    distH= random.uniform(3, 10)
    distV =  random.uniform(3,10)
    distmH =  random.uniform(3,distH)
    distmV =  random.uniform(3,distV)
    xH0 = round(-distH/2, 3)
    xH1 = round(distH/2, 3)
    yH = round(distV/2, 3)
    xV = xH1
    yV0 = round(-distV/2, 3)
    yV1 = yH
    xHm0 = xH1
    xHm1 = round(xH1-distmH, 3)
    yHm = yV0
    xVm = xH0
    yVm0 = yH
    yVm1 = round(yH-distmV, 3)
    
    if yVm1 != xHm1:
        xHn0 = xH0
        xHn1 = xHm1
        yHn = yVm1
        yVn0 = yHm
        yVn1 = yVm1
        xVn = xHm1
    lidarR = random.uniform(distH/2,8)
    Gephi = nx.Graph()
    obstacle = random.randrange(2)
    for index in range(0, obstacle):
        W = random.uniform(0.5,min(distH, distV)/2)
        xObs = random.uniform(xH0+W, xH1-W)
        yObs = random.uniform(yV0+W, yV1-W)
        Gephi.add_node((xObs,yObs), width= W,priority =0)
    
    Gephi.add_edges_from([
        ((xHm0,yHm), (xHm1,yHm),  {'lidarRange': lidarR, 'priority': 0, 'width': 0}),
        ((xVm,yVm0), (xVm,yVm1),  {'lidarRange': lidarR, 'priority': 0, 'width': 0}),
        ((xHn0,yHn), (xHn1,yHn),  {'lidarRange': lidarR, 'priority': 0, 'width': 0}),
        ((xVn,yVn0), (xVn,yVn1),  {'lidarRange': lidarR, 'priority': 0, 'width': 0}),
        ((xH0,yH), (xH1,yH),  {'lidarRange': lidarR, 'priority': 0, 'width': 0}),
        ((xV,yV0), (xV,yV1),  {'lidarRange': lidarR, 'priority': 0, 'width': 0})
    ])
    return Gephi

Gephi = randomGraph()

pos1 = {node: node for node in Gephi.nodes()}
nx.draw(Gephi, pos1, with_labels=False, node_size=10, edge_color = 'grey')
plt.xlim(-11, 11)
plt.ylim(-11, 11)
plt.xticks(range(-11, 11))
plt.yticks(range(-11, 11))
plt.gca().set_aspect('equal', adjustable='box')
#plt.show()

def createCircle(Gephi: nx.Graph):
    '''
    create circle area defining column obstacle 
    
    :param Gephi, our basic graph
    :return: list of edge representing circle 
    '''
    listNewEdge = []
    for edge in Gephi.edges():
        for node in edge:
            Gephi.nodes[node]["width"] = 0
            Gephi.nodes[node]["priority"] = 0
    for node2 in Gephi:
        node = Gephi.nodes[node2]
        # print(node)
        if node["priority"] == 0:
            if node["width"] > 0:
                step = math.pi/4
                theta = 0
                for index in range (0,8):
                    nodeX1 = round(node2[0]+ node["width"]* math.cos(theta), 3)
                    nodeY1 = round(node2[1]+ node["width"]* math.sin(theta), 3)
                    theta = theta + step
                    nodeX2 = round(node2[0]+ node["width"]* math.cos(theta), 3)
                    nodeY2 = round(node2[1]+ node["width"]* math.sin(theta), 3)
                    listNewEdge.append(((nodeX1, nodeY1),(nodeX2,nodeY2), {'lidarRange': 5, 'priority': 0, 'width': 0}))
    Gephi.add_edges_from(listNewEdge)
    
def HigherMonitoringZone(Gephi: nx.Graph):
    '''
    function that delimitate zone where we need more monitoring
    
    param the basic Graph gave by the user
    return: list of each zone representing by lower and upper bound of abscissa and ordinate with the priority
    '''
    listZone = []
    l = []
    blacklist = []
    for edge in Gephi.edges():
        for node in edge:
            Gephi.nodes[node]["width"] = 0
            Gephi.nodes[node]["priority"] = 0
    for edge in Gephi.edges():
        widthE = Gephi[edge[0]][edge[1]]["width"]
        priorityE = Gephi[edge[0]][edge[1]]["priority"]
        if widthE > 0:
            print(edge)
            dX = edge[1][0] - edge[0][0]
            dY = edge[1][1] - edge[0][1]
            dist= math.sqrt(dX**2 + dY**2)
            nbrN= dist//widthE
            if nbrN == 0:
                nbrN = 1
            alpha = dX / nbrN
            beta = dY / nbrN
            for k in range(0, int(nbrN)):
                    xAlpha = round(edge[0][0] + alpha*k,3)
                    yBeta = round(edge[0][1] + beta*k,3)
                    l.append((xAlpha,yBeta, widthE, priorityE))
            blacklist.append((edge[0], edge[1]))
    for node in l:
        Gephi.add_node((node[0],node[1]), width= node[2], priority =node[3])
    for edge in blacklist:
        Gephi.remove_edge(edge[0], edge[1])
    for node2 in Gephi:
        node = Gephi.nodes[node2]
        # print(node2)
        # print(node)
        if node["priority"] != 0:
            XUp = node2[0]+node["width"]
            XLow = node2[0]-node["width"]
            YUp = node2[1]+node["width"]
            YLow = node2[1]-node["width"]
            listZone.append((XUp, XLow,YUp, YLow, node["priority"]))
    return listZone

createCircle(Gephi)

listeEdgeLidar=[]
for edge in Gephi.edges:
    lidarRange = Gephi[edge[0]][edge[1]]["lidarRange"]
    if lidarRange > 0:
        listeEdgeLidar.append((edge, lidarRange))
# print("listeEdgeLidar ", listeEdgeLidar)

def LidarPositioningPossibilities(listPossibleLidarPositionEdges):
    '''
    Generator of nodes representing lidars position possibilities
    
    :param listPossibleLidarPositionEdges: list representing edge for lidars placement
    :return: list of all the position of the lidars regrouping in list for all edges
    '''
    listNodeLidar : List[List[Node]]
    l : List[Node]
    stepLidar = random.uniform(0.5, 3)
    listNodeLidar = []
    listPositionLidar = []
    for i in range (0, len(listPossibleLidarPositionEdges)):
            l = []
            lNodeLidar = []
            lidarBeam = listPossibleLidarPositionEdges[i][1]
            dX = listPossibleLidarPositionEdges[i][0][1][0] - listPossibleLidarPositionEdges[i][0][0][0]
            dY = listPossibleLidarPositionEdges[i][0][1][1] - listPossibleLidarPositionEdges[i][0][0][1]
            dist= math.sqrt(dX**2 + dY**2)
            nbrL = dist//stepLidar
            if nbrL == 0:
                nbrL = 1
            alpha = dX / nbrL
            beta = dY / nbrL
            # print(nbrL)
            # print(listPossibleLidarPositionEdges[i][0])
            # Calculate for each edge all the lidar position possibilities
            if listPossibleLidarPositionEdges[i][0][0] in listPositionLidar:
                for k in range(0, int(nbrL)):
                    #print("ici")
                    xAlpha = round(listPossibleLidarPositionEdges[i][0][1][0] - alpha*k, 3)
                    yBeta = round(listPossibleLidarPositionEdges[i][0][1][1] - beta*k, 3)
                    l.append((xAlpha , yBeta, lidarBeam))
                    lNodeLidar.append((xAlpha , yBeta))
            else:
                for k in range(0, int(nbrL)):
                    xAlpha = round(listPossibleLidarPositionEdges[i][0][0][0] + alpha*k,3)
                    yBeta = round(listPossibleLidarPositionEdges[i][0][0][1] + beta*k,3)
                    l.append((xAlpha, yBeta, lidarBeam ))
                    lNodeLidar.append((xAlpha , yBeta))
            listNodeLidar.extend(l)
            listPositionLidar.extend(lNodeLidar)
            #Graph.add_node((xAlpha, yBeta), lidarRange= lidarBeam)
    return listNodeLidar, listPositionLidar

listLidar, listPositionLidar = LidarPositioningPossibilities(listeEdgeLidar)
print(len(listPositionLidar))
for Lidar in listLidar:
    Graph.add_node((Lidar[0], Lidar[1]), lidarRange= Lidar[2])
# print("liste Lidar ",listLidar)
#Graph.add_nodes_from(listLidar)

def dist(A: Node, B: Node):
    return math.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)

def checking(node1, node2):
            check = True
            # ax + by + c = 0
            if node1[0]-node2[0] == 0: # case node1 node2 are vertical line 
                a1 = 1
                b1 = 0
                c1 = -node1[0]
            else:
                a1 = (node2[1]-node1[1])/(node1[0]-node2[0])
                b1 = 1
                c1 = - node1[1] - a1*node1[0]
            for edge in Gephi.edges:
                if edge[0][0]-edge[1][0] == 0:
                    a2 = 1
                    b2 = 0
                    c2 = -edge[0][0]
                else:
                    a2 = (edge[0][1]-edge[1][1])/(edge[1][0] - edge[0][0])
                    b2 = 1
                    c2 = - edge[0][1] - a2* edge[0][0]
                if a2*b1-a1*b2 != 0:
                    #print(m1*a2-m2*a)
                    x = round((c2*b1-c1*b2)/(a1*b2-a2*b1), 3)
                    y = round(-(c2*a1-c1*a2)/(a1*b2-a2*b1), 3)
                    if (min(node2[0],node1[0]) <= x <= max(node1[0], node2[0]) and min(node2[1],node1[1]) <= y <= max(node1[1], node2[1])) and (min(edge[0][0],edge[1][0]) <= x <= max(edge[0][0],edge[1][0]) and min(edge[0][1],edge[1][1]) <= y <= max(edge[0][1],edge[1][1]))  and (node1[0] != x or node1[1] != y) :
                        check = False
                        # print("The lidar",node1, "can not reach the point",node2, "because of the obstacle",edge, "avec le point d'intersection", x, y)
            return check
        
def newMonitorPoint(element: Node, stepStreet: float, blacklist: list, listNodeStreet : list):
    neighbors = [
        (element[0]+stepStreet, element[1]),
        (element[0]-stepStreet, element[1]),
        (element[0], element[1]+stepStreet),
        (element[0], element[1]-stepStreet),
        (element[0]+stepStreet, element[1]+stepStreet),
        (element[0]-stepStreet, element[1]-stepStreet),
        (element[0]-stepStreet, element[1]+stepStreet),
        (element[0]+stepStreet, element[1]-stepStreet),
        ]
    for neighbor in neighbors:
        if neighbor not in blacklist:
            if checking(element, neighbor):
                listNodeStreet.append(neighbor)
            else:
                blacklist.append(neighbor)
    return

def MonitoringArea(Gephi: nx.Graph):
    HMZ = HigherMonitoringZone(Gephi)
    stepStreet = random.uniform(0.5, 2)
    listToVisit = []
    blacklist = []
    listNodeStreet = []
    maxEdge = ((0,0),(0,0))
    for edge in Gephi.edges:
        if edge[0][1] == edge[1][1]:
            if edge[0][1] > maxEdge[0][1]:
                maxEdge = edge
    maxNode = max(maxEdge[0], maxEdge[1])
    listNewNode = []
    firstNode = (round((maxNode[0]-stepStreet), 3),round((maxNode[1]-stepStreet), 3))
    listNewNode.append(firstNode)
    print(firstNode)
    while listNewNode != []:
        listToVisit = listNewNode
        listNewNode = []
        for element in listToVisit:
            for index in HMZ:
                if index[1] <= element[0] <= index[0] and index[3] <= element[1] <= index[2]:
                    newMonitorPoint(element, stepStreet/index[4], blacklist, listNodeStreet)
            listNodeStreet.append(element)
            blacklist.append(element)
            neighbors = [
                
                (round(element[0]+stepStreet,3), element[1]),
                (round(element[0]-stepStreet,3), element[1]),
                (element[0], round(element[1]+stepStreet,3)),
                (element[0], round(element[1]-stepStreet,3)),
                ]
            for neighbor in neighbors:
                if neighbor not in blacklist and neighbor not in listToVisit and neighbor not in listNewNode:
                    if checking(element, neighbor):
                        listNewNode.append(neighbor)
                    else:
                        blacklist.append(neighbor)
    return listNodeStreet


def lidarJoint(listNodeStreet : List[Node], listLidar: List[Node]):
    listCoveringEdge = []
    for lidar in listLidar:
        for CoveringNode in listNodeStreet:
            distance = dist((lidar[0], lidar[1]), CoveringNode)
            if distance <= lidar[2]:
                if checking((lidar[0], lidar[1]), CoveringNode):
                    listCoveringEdge.append((((lidar[0], lidar[1])),(CoveringNode)))
    return listCoveringEdge

listCovering = MonitoringArea(Gephi)
listCoveringEdge = lidarJoint(listCovering, listLidar)
Graph.add_nodes_from(listCovering)
Graph.add_edges_from(listCoveringEdge)


# Formulation of the mathematical problem

def energyF(listPositionLidar, listCovering):
    def f(x):
        somme = 0
        L = 10
        somme = int(sum([x[i] for (i,v) in enumerate(listPositionLidar)]))
        for node in listCovering:
            #print(node)
            prod = int(math.prod([1 - x[i] for (i,v) in enumerate(listPositionLidar) if (v in Graph.neighbors(node))])) * L
            somme += prod
        return somme
    return(f)

# Simulated Annealing

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

def runningSA():
    listEnergy = []
    listNbrLidarAct = []
    listCoverNAct = []
    listUnCoverAct = []
    n_temp_iter= 1000
    for i in range (0, 1):
        samples, energies, runtime = sa_solve(energyF(listPositionLidar, listCovering), len(listPositionLidar), n_temp_iter)
        S = []
        for i in range(len(samples[0])):
            if samples[0][i]:
                S.append(listPositionLidar[i])
        listCoverN = []
        for node in S:
            for node2 in Graph.neighbors(node):
                listCoverN.append(node2)
                
        listEnergy.extend(energies)
        listCoverNAct.append(len(listCoverN))
        listUnCoverAct.append((len(listCovering) - len(listCoverN)))
        listNbrLidarAct.append(len(S))
    actE = []
    actN = []
    for node in S:
        for edge in Graph.edges(node):
            actE.append(edge)
        for node2 in Graph.neighbors(node):
            actN.append(node2)
    pos = {node: node for node in Graph.nodes()}
    nx.draw(Graph, pos, with_labels=False, node_size=40, node_color = 'grey')
    nx.draw_networkx_nodes(Graph, pos, listPositionLidar, node_color = 'purple', node_size= 40)
    nx.draw_networkx_nodes(Graph, pos, S, node_color='red', node_size=40)
    nx.draw_networkx_nodes(Graph, pos, actN, node_color='blue', node_size=40)
    nx.draw_networkx_edges(Graph, pos, actE, edge_color='blue')
    
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.xticks(range(-11, 11))
    plt.yticks(range(-11, 11))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    savefig('demo.png', transparent=True)
    return listEnergy, listCoverNAct, listUnCoverAct, listNbrLidarAct

listEnergy, listCoverNAct, listUnCoverAct, ListNbrLidarAct = runningSA()
# if __name__ == "__main__":
#     runningSA()

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import neal
import greedy
#from PuboOnGraph import Graph, listCovering, listLidar, listEnergy as lEnerPUBO, listCoverNAct as lCovPUBO, listUnCoverAct as lUnCovPUBO, ListNbrLidarAct as lLidPUBO
from PuboOnSqArea import Graph, listCovering, listPositionLidar as listLidar, listEnergy as lEnerPUBO, listCoverNAct as lCovPUBO, listUnCoverAct as lUnCovPUBO, ListNbrLidarAct as lLidPUBO
G = nx.Graph()

points1 = [] #placement of lidars for one side
points2 = [] #placement of lidars for other side
points3 = [] #vertices to cover

pointsL = listLidar
points3 = listCovering
G = Graph

#G.add_nodes_from(points1 + points2 + points3)

def showGraph(S):
    
    pos = {node: node for node in G.nodes()} 
    nx.draw(G, pos, with_labels=False, node_size=40)
    actE = []
    actN = []
    for node in S:
      for edge in G.edges(node):
        actE.append(edge)
      for node2 in G.neighbors(node):
        actN.append(node2)
    # Draw the graph
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=False,node_color = 'grey', node_size=40)
    nx.draw_networkx_nodes(G, pos, listLidar, node_color = 'purple', node_size= 40)
    nx.draw_networkx_nodes(G, pos, S, node_color='red', node_size=40)
    nx.draw_networkx_nodes(G, pos, actN, node_color='blue', node_size=40)
    nx.draw_networkx_nodes(G, pos, S, node_color='red', node_size=40)
    nx.draw_networkx_edges(G, pos, actE, edge_color='red', node_size=40)
    plt.show()
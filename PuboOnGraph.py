import networkx as nx
import matplotlib.pyplot as plt
import math as m
import numpy as np
from typing import Tuple, List,  Union, Callable, NewType
import random
from time import time

Node = NewType("node",Tuple[float, float])
Edge = NewType("edge",Tuple[Node, Node])
Edge_Angle_List = NewType("Edge_Angle_List",List[Tuple[Edge, float]])


Gephi = nx.Graph()

Gephi.add_edges_from([
    ((-8,16),(-10,-2), {'width':2}),
    ((5,10), (-8,16), {'width': 2}),
    ((5,10), (0,0), {'width': 2}),
    ((0,0), (15,-6), {'width': 2}),
    ((-10,-2), (0,0), {'width': 2})
    ])


def neighborGenerator(Gephi):
    """
    Generator of list of all edges link between two by two

    :param Gephi: Graph representing the intersection of the streets
    :return: List of adjacents edges with their angles
    """
    neighborEdge: List[Tuple[Edge, Edge, float, float, float]]
    neighborEdge = []
    
    for node in Gephi.nodes:
        if len(Gephi.edges(node)) > 1:
            l:Edge_Angle_List ;  listUR:Edge_Angle_List ; listUL:Edge_Angle_List ; listBL:Edge_Angle_List ; listBR:Edge_Angle_List ; bigList : Edge_Angle_List
            l, listUR, listUL, listBL, listBR, bigList = [],[],[],[],[],[]
            
            # convert angles of edges for each node depending of the orientation
            
            for E1 in Gephi.edges(node):
                second_node1 = set(E1).difference([node]).pop()
                radiusE1 = m.degrees(m.atan(abs(node[1] - second_node1[1])/abs(node[0] - second_node1[0])))
                if (node[0] > second_node1[0] and node[1] <= second_node1[1]):
                    radiusE1 = 180 - radiusE1
                    listUL.append((E1, radiusE1))
                elif (node[0] <= second_node1[0] and node[1] > second_node1[1]):
                    radiusE1 = 360 - radiusE1
                    listBR.append((E1, radiusE1))
                elif (node[0] > second_node1[0] and node[1] > second_node1[1]):
                    radiusE1 = 180 + radiusE1
                    listBL.append((E1, radiusE1))
                else:
                    listUR.append((E1, radiusE1))
                    
            # sort list by the degree from 0 to 360
            
            for i in [listUR, listUL, listBL, listBR]:
                l = sorted(i, key=lambda x: x[1])
                bigList.extend(l)
                l = []
            if len(bigList) > 2:
                for k in range(len(bigList)-1):
                    radiusE1E2 = abs(bigList[k][1] - bigList[k+1][1])
                    l.append((bigList[k][0],bigList[k+1][0],radiusE1E2, bigList[k][1], bigList[k+1][1]))
                radiusE1E2 = abs(bigList[0][1] + (360-bigList[-1][1]))
                l.append((bigList[0][0],bigList[-1][0],radiusE1E2, bigList[0][1], bigList[-1][1]))
            else:
                if (bigList[0] in listUR or bigList[0] in listBR) and (bigList[-1] in listUR or bigList[-1] in listBR) and not ((bigList[0] in listUR and bigList[-1] in listUR) or (bigList[0] in listBR and bigList[-1] in listBR)):
                    radiusE1E2 = abs(bigList[0][1] + (360-bigList[-1][1]))
                else:
                    radiusE1E2 = abs(bigList[0][1] - bigList[-1][1])
                l.append((bigList[0][0],bigList[-1][0],radiusE1E2, bigList[0][1], bigList[-1][1]))
            neighborEdge.extend(l)
            l = []
    # print("neighborEdge = ", neighborEdge)
    return(neighborEdge)

def intersectionGenerator(_list):
    """
    Generator of dictonary of all possible node that could represente the intersection of both streets 
    borders for each node depending of their neighbore node

    :param _list: List of each edge linked two by two 
    :return: dictionary for each node their borders node associated with their neightbore node 
    """
    _dict = {}
    for index in _list:
        set1 = set(index[0]) #recover the node
        set2 = set(index[1]) #recover the node
        commonNode = next(iter(set1.intersection(set2)))
        node1E1, node2E1 = index[0][0],index[0][1]
        node1E2, node2E2 = index[1][0],index[1][1]
        widthE1 = Gephi[node1E1][node2E1]["width"]
        widthE2 = Gephi[node1E2][node2E2]["width"]

        #Find perpendicular nodes to find the borders of the street 
        
        ptE1LX1 = node1E1[0] + widthE1*np.cos(m.radians(index[3]+90))
        ptE1LY1 = node1E1[1] + widthE1*np.sin(m.radians(index[3]+90))
        ptE1LX2 = node2E1[0] + widthE1*np.cos(m.radians(index[3]+90))
        ptE1LY2 = node2E1[1] + widthE1*np.sin(m.radians(index[3]+90))

        ptE2LX1 = node1E2[0] + widthE2*np.cos(m.radians(index[4]+90))
        ptE2LY1 = node1E2[1] + widthE2*np.sin(m.radians(index[4]+90))
        ptE2LX2 = node2E2[0] + widthE2*np.cos(m.radians(index[4]+90))
        ptE2LY2 = node2E2[1] + widthE2*np.sin(m.radians(index[4]+90))

        m1L = (ptE1LY2-ptE1LY1)/(ptE1LX2-ptE1LX1)
        p1L = ptE1LY1 - m1L * ptE1LX1
        m2L = (ptE2LY2-ptE2LY1)/(ptE2LX2-ptE2LX1)
        p2L = ptE2LY1 - m2L * ptE2LX1

        ptE1RX1 = node1E1[0] + widthE1*np.cos(m.radians(index[3]-90)) 
        ptE1RY1 = node1E1[1] + widthE1*np.sin(m.radians(index[3]-90))
        ptE1RX2 = node2E1[0] + widthE1*np.cos(m.radians(index[3]-90))
        ptE1RY2 = node2E1[1] + widthE1*np.sin(m.radians(index[3]-90))

        ptE2RX1 = node1E2[0] + widthE2*np.cos(m.radians(index[4]-90))
        ptE2RY1 = node1E2[1] + widthE2*np.sin(m.radians(index[4]-90))
        ptE2RX2 = node2E2[0] + widthE2*np.cos(m.radians(index[4]-90))
        ptE2RY2 = node2E2[1] + widthE2*np.sin(m.radians(index[4]-90))

        m1R = (ptE1RY2-ptE1RY1)/(ptE1RX2-ptE1RX1)
        p1R = ptE1RY1-m1R*ptE1RX1
        m2R = (ptE2RY2-ptE2RY1)/(ptE2RX2-ptE2RX1)
        p2R = ptE2RY1-m2R*ptE2RX1
        
        #Find intersection node between to edge that cross each other
        
        xR = (p1R-p2L)/(m2L-m1R)
        yR = m2L*(p1R-p2L)/(m2L-m1R)+p2L
        xL = (p1L-p2R)/(m2R-m1L)
        yL = m2R*(p1L-p2R)/(m2R-m1L)+p2R

        #Check the common node between two edge 
        nE1 = None
        if commonNode == node1E1:
            nE1 =  node2E1
        else:
            nE1 = node1E1
        nE2 = None
        if commonNode == node1E2:
            nE2 = node2E2
        else:
            nE2 = node1E2
        position = None
        
        # Find if all the nodes have superior axis comparing to the common node
        
        if (commonNode[0] < nE1[0] and commonNode[0] < nE2[0]):
            position = 'R'
        else :
            position = 'L'
        
        # Fill the dictionary depending of the type of the node 
        # the first number of the dict represent the degree
        # -1: node with one neightbore 
        
        if not nE1 in _dict:
            _dict[nE1] = [-1, ((ptE1RX2, ptE1RY2),(ptE1LX2, ptE1LY2), commonNode)]
        if not nE2 in _dict:
            _dict[nE2] = [-1, ((ptE2LX2, ptE2LY2),(ptE2RX2, ptE2RY2), commonNode)]

        if not commonNode in _dict:
            _dict[commonNode] = [2, ((xR, yR),(xL,yL),nE1, nE2, index[3], index[4]),position]
        else:
            if _dict[commonNode][0] == -1:
                _dict[commonNode] = [2, ((xR, yR),(xL,yL),nE1, nE2, index[3], index[4]), position]
            else:
                _dict[commonNode].append(((xR, yR),(xL,yL),nE1, nE2, index[3], index[4]))
                _dict[commonNode].extend(position)
                _dict[commonNode][0] += 1
        
    # Ajust the good degree for each node that have more than 2 neighbores
    
    for i in _dict.keys():
        if _dict[i][0] > 2:
            _dict[i][0] -= 1
    # print(_dict)
    return _dict

def graphBordersStreet(_dict):
    '''
    Generator of the borders of the streets 

    :param _dict: dictionary for each node their borders node associated with their neightbore node
    :return: list of the borders of the street where lidar can be placed in the form and the width
    [Edge0Border0Node0, Edge0Border0Node1, Edge0Border1Node0, Edge0Border1Node1, width, Edge1Border0Node0, ...]
    '''
    
    listPossibleLidarPositionEdges : List[Tuple[Node, Node, Node, Node], float]
    blacklist : List[Node]
        
    listPossibleLidarPositionEdges = []
    blacklist = []
    
    for n1 in Gephi.nodes:
        bordersFromN1(blacklist, n1, _dict, listPossibleLidarPositionEdges)
    return listPossibleLidarPositionEdges

# Find the edges representing the borders of the street depending on the first node N1

def bordersFromN1(blacklist, n1, _dict, listPossibleLidarPositionEdges):
    degreeN1 = _dict[n1][0]
    
    # Separate the program into two function depending of the degree of the first node
 
    if degreeN1 < 3:
        for n2 in Gephi.neighbors(n1):
            width = Gephi[n1][n2]["width"]
            if n2 not in blacklist:
                
                # Create an edge between the node and its neightbore using bordersFromN2
                bordersFromN2(n2, n1, _dict, listPossibleLidarPositionEdges)
                listPossibleLidarPositionEdges.append(width)
    else:
        for n2 in Gephi.neighbors(n1):
            width = Gephi[n1][n2]["width"]
            if n2 not in blacklist:
            
                #Create an edge between the node and its neightbore using bordersFromN2v2
                bordersFromN2v2(n2, n1, _dict, listPossibleLidarPositionEdges)
                listPossibleLidarPositionEdges.append(width)
    blacklist.append(n1)
    return width


def dist(A: Node, B: Node):
    return m.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)

# Find the edges representing the borders of the street depending on the second node N2 with N1 a simple angle

def bordersFromN2(n2, n1, _dict, listPossibleLidarPositionEdges):
    degreeN2 = _dict[n2][0]
    list1by1Border = []
    if degreeN2 < 3:
        A = dist(_dict[n1][1][0], _dict[n2][1][0]) + dist(_dict[n1][1][1],_dict[n2][1][1])
        B = dist(_dict[n1][1][1], _dict[n2][1][0]) + dist(_dict[n1][1][0],_dict[n2][1][1])
        if (A < B):
            listPossibleLidarPositionEdges.append((_dict[n1][1][0],_dict[n2][1][0],_dict[n1][1][1],_dict[n2][1][1]))
            
        else:
            listPossibleLidarPositionEdges.append((_dict[n1][1][1],_dict[n2][1][0],_dict[n1][1][0],_dict[n2][1][1]))
            
    else:
        for i in range(1, len(_dict[n2]), 2):
            if _dict[n2][i][2] == n1:
                if _dict[n2][i][4] < _dict[n2][i][5]:
                    if _dict[n2][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][1][1]),(_dict[n2][i][1])))
                    else:
                        list1by1Border.extend(((_dict[n1][1][0]),(_dict[n2][i][0])))
                else:
                    if _dict[n2][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][1][0]),(_dict[n2][i][1])))
                    else:
                        list1by1Border.extend(((_dict[n1][1][1]),(_dict[n2][i][0])))
                            
            if _dict[n2][i][3] == n1:
                if _dict[n2][i][4] > _dict[n2][i][5]:
                    if _dict[n2][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][1][1]),(_dict[n2][i][1])))
                    else:
                        list1by1Border.extend(((_dict[n1][1][0]),(_dict[n2][i][0])))
                else:
                    if _dict[n2][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][1][0]),(_dict[n2][i][1])))
                    else:
                        list1by1Border.extend(((_dict[n1][1][1]),(_dict[n2][i][0])))
        A = dist(list1by1Border[0], list1by1Border[1]) + dist(list1by1Border[2],list1by1Border[3])
        B = dist(list1by1Border[1], list1by1Border[2]) + dist(list1by1Border[0],list1by1Border[3])
        if (A < B):
                listPossibleLidarPositionEdges.append((list1by1Border[0], list1by1Border[1], list1by1Border[2],list1by1Border[3]))
                
        else:
                listPossibleLidarPositionEdges.append((list1by1Border[0], list1by1Border[3], list1by1Border[1],list1by1Border[2]))
                
    return listPossibleLidarPositionEdges

# Find the edges representing the borders of the street depending on the second node N2 with N1 an intersection

def bordersFromN2v2(n2, n1, _dict, listPossibleLidarPositionEdges):
    list1by1Border = []
    degreeN2 = _dict[n2][0]
    if degreeN2 < 3:
        for i in range(1, len(_dict[n1]), 2):
            if _dict[n1][i][2] == n2:
                if _dict[n1][i][4] < _dict[n1][i][5]:
                    if _dict[n1][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][i][1]),(_dict[n2][1][0])))
                    else:
                        list1by1Border.extend(((_dict[n1][i][0]),(_dict[n2][1][0])))
                else:
                    if _dict[n1][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][i][1]),(_dict[n2][1][1])))
                    else:
                        list1by1Border.extend(((_dict[n1][i][0]),(_dict[n2][1][0])))
            if _dict[n1][i][3] == n2:
                if _dict[n1][i][4] > _dict[n1][i][5]:
                    if _dict[n1][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][i][1]),(_dict[n2][1][1])))
                    else:
                        list1by1Border.extend(((_dict[n1][i][0]),(_dict[n2][1][0])))
                else:
                    if _dict[n1][i+1] == 'L':
                        list1by1Border.extend(((_dict[n1][i][1]),(_dict[n2][1][1])))
                    else:
                        list1by1Border.extend(((_dict[n1][i][0]),(_dict[n2][1][0])))
        A = dist(list1by1Border[0], list1by1Border[1]) + dist(list1by1Border[2],list1by1Border[3])
        B = dist(list1by1Border[1], list1by1Border[2]) + dist(list1by1Border[0],list1by1Border[3])
        if (A < B):
                listPossibleLidarPositionEdges.append((list1by1Border[0], list1by1Border[1], list1by1Border[2],list1by1Border[3]))

        else:
                listPossibleLidarPositionEdges.append((list1by1Border[0], list1by1Border[3], list1by1Border[1],list1by1Border[2]))
    return listPossibleLidarPositionEdges

def LidarPositioningPossibilities(listPossibleLidarPositionEdges):
    '''
    Generator of nodes representing lidars position possibilities

    :param listPossibleLidarPositionEdges: list representing edge for lidars placement with width every four nodes
    :return: list of all the position of the lidars regrouping in list for all edges
    '''
    listNodeLidar : List[List[Node]]
    l : List[Node]
    stepLidar = 2.5
    listNodeLidar = []
    for i in range (0, len(listPossibleLidarPositionEdges), 2):
        for j in range(0, len(listPossibleLidarPositionEdges[i]), 2):
            l = []
            
            dX = listPossibleLidarPositionEdges[i][j+1][0] - listPossibleLidarPositionEdges[i][j][0]
            dY = listPossibleLidarPositionEdges[i][j+1][1] - listPossibleLidarPositionEdges[i][j][1]
            dist= m.sqrt(dX**2 + dY**2)
            nbrL = dist//stepLidar
            alpha = dX / nbrL
            beta = dY / nbrL
        
            #Calculate for each edge all the lidar position possibilities
        
            for k in range(0, int(nbrL) ):
                xAlpha = listPossibleLidarPositionEdges[i][j][0] + alpha*k
                yBeta = listPossibleLidarPositionEdges[i][j][1] + beta*k
                l.append((xAlpha , yBeta))
            listNodeLidar.append(l)
            listNodeLidar.append(listPossibleLidarPositionEdges[i+1])
    return listNodeLidar


def StreetPositionCovering(listNodeLidar):
    """
    Generator of nodes representing the area to cover

    :param listNodeLidar: list of all the lidar grouping by edges
    :return: list of all the position of the lidars and covering node regrouping in list for all edges
    """
    
    listNodes: List[List[Node], str]
    listNodeCovering: List[Node]
    
    listNodes = []
    listNodeCovering = []
    
    #step of two : one side is enough to determine the orientation of the covering node
    
    for index in range(0, len(listNodeLidar)-1, 4):
        listNodeCovering = []
        listNodes.append(listNodeLidar[index] + listNodeLidar[index+2])
        AngleE = m.degrees(m.atan((listNodeLidar[index][0][1] - listNodeLidar[index][-1][1])/(listNodeLidar[index][0][0] - listNodeLidar[index][-1][0])))
        middleLidar = len(listNodeLidar[index]) // 2
        n1X = listNodeLidar[index][middleLidar][0] + 2*np.cos(m.radians(AngleE+90))
        n1Y = listNodeLidar[index][middleLidar][1] + 2*np.sin(m.radians(AngleE+90))
        n2X = listNodeLidar[index][middleLidar][0] + 2*np.cos(m.radians(AngleE-90))
        n2Y = listNodeLidar[index][middleLidar][1] + 2*np.sin(m.radians(AngleE-90))
        n1 = (n1X, n1Y)
        n2 = (n2X, n2Y)
        A = dist(n1,listNodeLidar[index+2][0]) + dist(n1,listNodeLidar[index+2][-1])
        B = dist(n2,listNodeLidar[index+2][0]) + dist(n2,listNodeLidar[index+2][-1])
        for node in listNodeLidar[index]:
            for percent in range (4,9,4):
                perc = listNodeLidar[index+1] *2 * percent*0.1
                if (A < B):
            
                    # all node of the covering street should have +90 depending of the lidar of listNodeLidar[index]
                    # all node of the covering street should have -90 depending of the lidar of listNodeLidar[index+2]
                    
                    n1X = node[0] + perc*np.cos(m.radians(AngleE+90))
                    n1Y = node[1] + perc*np.sin(m.radians(AngleE+90))
                    point = (n1X,n1Y)
                    listNodeCovering.append(point)
                else :
                    n1X = node[0] + perc*np.cos(m.radians(AngleE-90))
                    n1Y = node[1] + perc*np.sin(m.radians(AngleE-90))
                    point = (n1X,n1Y)
                    listNodeCovering.append(point)
        for node in listNodeLidar[index+2]:
            for percent in range (4,9,4):
                perc = listNodeLidar[index+1] *2 * percent*0.1
                if(A < B):
                    n1X = node[0] + perc*np.cos(m.radians(AngleE-90))
                    n1Y = node[1] + perc*np.sin(m.radians(AngleE-90))
                    point = (n1X,n1Y)
                    listNodeCovering.append(point)
                else:
                    n1X = node[0] + perc*np.cos(m.radians(AngleE+90))
                    n1Y = node[1] + perc*np.sin(m.radians(AngleE+90))
                    point = (n1X,n1Y)
                    listNodeCovering.append(point)
        listNodes.append(listNodeCovering)  
    return(listNodes)
      


def StreetEdge(listNodes,lidarBeamRange = 3):
    """
    Generator of edges between Lidars and Area to cover

    :param listNode: list of all the lidar and covering node grouping by edges
    :return: 
    """
    Graph = nx.Graph()
    listLidar = []
    listCovering = []
    for index in range (0, len(listNodes), 2):
        for nodeLidar in listNodes[index]:
            for nodeCovering in listNodes[index+1]:
                distLC = dist(nodeLidar, nodeCovering)
                if distLC <= lidarBeamRange:
                    #print("nodeLidar = ",nodeLidar,"nodeCovering", nodeCovering)
                    Graph.add_edge(nodeLidar, nodeCovering)
                    if nodeLidar not in listLidar:
                        listLidar.append(nodeLidar)
                    if nodeCovering not in listCovering:
                        listCovering.append(nodeCovering)
    #print(Graph.edges)
    # print(listCovering)
    # print(listLidar)
    return Graph,listLidar,listCovering
    


Graph,listLidar,listCovering = StreetEdge(StreetPositionCovering(LidarPositioningPossibilities(graphBordersStreet(intersectionGenerator(neighborGenerator(Gephi))))))

# Draw the graph before SA
def render():
    pos = {node: node for node in Graph.nodes}
    nx.draw(Graph,  pos, with_labels=False, node_color ='blue', node_size = 10)
    nx.draw_networkx_nodes(Graph, pos, listLidar, node_color='red', node_size= 10)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.xticks(range(-20, 20))
    plt.yticks(range(-20, 20))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Formulation of the mathematical problem

def energyF(listLidar, listCovering):
    def f(x):
        somme = 0    
        L = 10
        somme = int(sum([x[i] for (i,v) in enumerate(listLidar)]))
        for node in listCovering:
            #print(node)
            prod = int(m.prod([1 - x[i] for (i,v) in enumerate(listLidar) if (v in Graph.neighbors(node))])) * L
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

            metropolis_eval = m.exp(-diff / t)

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
    
    n_temp_iter= 1000
    for i in range (0, 10):
        samples, energies, runtime = sa_solve(energyF(listLidar, listCovering), len(listLidar), n_temp_iter)
        S = []
        for i in range(len(samples[0])):
            if samples[0][i]:
                S.append(listLidar[i])
        listCoverN = []
        for node in S:
            for node2 in Graph.neighbors(node):
                listCoverN.append(node2)
                
        listEnergy.extend(energies)
        listCoverNAct.append(len(listCoverN))
        listNbrLidarAct.append(len(S))

    print("runtime = ",runtime)
        
    fig,ax = plt.subplots(3)
    ax[0].set_title("PUBO ran on LoopedGraph")
    ax[0].set_ylabel("Energies")
    ax[0].set_xticklabels(["SA 1000 iter 10"] )
    ax[0].boxplot(listEnergy)
    ax[1].set_ylabel("Nbr Activated Lidars")
    ax[1].set_xticklabels(["SA 1000 iter 10"] )
    ax[1].boxplot(listCoverNAct)
    ax[2].set_ylabel("Nbr Activate Covering Nodes")
    ax[2].set_xticklabels(["SA 1000 iter 10"] )
    ax[2].boxplot(listNbrLidarAct)
    plt.show()
    
    
    # print(list(zip(samples, energies)))
    # print(S) 
    actE = []
    actN = []
    for node in S:
        for edge in Graph.edges(node):
            actE.append(edge)
        for node2 in Graph.neighbors(node):
            actN.append(node2)
    pos = {node: node for node in Graph.nodes()}
    nx.draw(Graph, pos, with_labels=False, node_size=40)
    nx.draw_networkx_nodes(Graph, pos, listLidar, node_color = 'purple', node_size= 40)
    nx.draw_networkx_nodes(Graph, pos, S, node_color='red', node_size=40)
    nx.draw_networkx_nodes(Graph, pos, actN, node_color='blue', node_size=40)
    nx.draw_networkx_edges(Graph, pos, actE, edge_color='red')
    
    plt.show()

# if __name__ == "__main__":
    # runningSA()
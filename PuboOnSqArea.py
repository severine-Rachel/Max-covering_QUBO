import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import math as m
import numpy as np
from typing import Tuple, List,  Union, Callable, NewType
import random
from time import time

Node = NewType("node",Tuple[float, float])
Edge = NewType("edge",Tuple[Node, Node])
Edge_Angle_List = NewType("Edge_Angle_List",List[Tuple[Edge, float]])

Gephi = nx.Graph()
listNodeLidar = []
Gephi.add_edges_from([
    ((0,-10), (0,10), {'width':20, 'area': 'L'}),
    ((-5,-10), (-5,-5), {'width':0, 'area': 'C'}),
    ((-5,-5), (-10,-5), {'width':0, 'area': 'C'}),
    ((-10,-5), (-10,5), {'width':0, 'area': 'C'}),
    ((-10,5), (5,5), {'width':0, 'area': 'C'}),
    ((5,5), (5,-10), {'width':0, 'area': 'C'}),
    ((5,-10), (-5,-10), {'width':0, 'area': 'C'})
    ])
for edge in Gephi.edges:
    if edge["area"] == 'L':
        if edge[0] not in listNodeLidar:
            listNodeLidar.append(edge[0])
        if edge[1] not in listNodeLidar:
            listNodeLidar.append(edge[1])
Gephi.add_nodes_from([
    ((5,5), {"width": 2})
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
    if neighborEdge == None:
        pass
    # print("neighborEdge = ", neighborEdge)
    return(neighborEdge)


def intersectionGenerator(neighborEdge):
    """
    Generator of dictonary of all possible node that could represente the intersection of both streets 
    borders for each node depending of their neighbore node

    :param neighborEdge: List of each edge linked two by two 
    :return: dictionary for each node their borders node associated with their neightbore node 
    """
    _dict = {}
    for index in neighborEdge:
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
        
        #Find intersection node between two edge that cross each other
        
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

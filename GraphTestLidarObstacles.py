import networkx as nx
import matplotlib.pyplot as plt
import math
G = nx.Graph()
G2 = nx.Graph()

G.add_edges_from([
    ((-10, -10),(-10,10)),
    ((-10,10), (10,10)),
    ((10,10), (10,-10)),
    ((10,-10), (-10,-10)),
    #((-10,-10),(10,10)),
    #((10,-10),(-10,10))
])
G.add_edge((-8, 5),(0,7))
listLidar = [(8,-5)]
G.add_nodes_from(listLidar)
listCovNode = [(-7, 8),(8,8),(2, 8)]
G.add_nodes_from(listCovNode)


for nodeL in listLidar:
    for nodeC in listCovNode:
        check = True
        # ax + by + c = 0
        if nodeC[0]-nodeL[0] == 0: # case nodeC nodeL are vertical line 
            a1 = 1
            b1 = 0
            c1 = -nodeL[0]
        else:
            a1 = (nodeC[1]-nodeL[1])/(nodeL[0]-nodeC[0])
            b1 = 1
            c1 = - nodeL[1] - a1*nodeL[0]
        for edge in G.edges:
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
                x = (c2*b1-c1*b2)/(a1*b2-a2*b1)
                y = -(c2*a1-c1*a2)/(a1*b2-a2*b1)
                if min(nodeC[0],nodeL[0]) < x < max(nodeL[0], nodeC[0]) and min(edge[0][0],edge[1][0]) < x < max(edge[0][0],edge[1][0]) :
                    check = False
                    print("The lidar",nodeL, "can not reach the point",nodeC, "because of the obstacle",edge)
                    
                    G2.add_node((x, y))
        if check == True:
            G.add_edge(nodeL, nodeC)
#def 
            
pos = {node: node for node in G.nodes()}
nx.draw(G, pos, with_labels=False, node_size=0)
nx.draw_networkx_nodes(G, pos, listLidar, node_color='red',  node_size=40, label='Lidar' )
nx.draw_networkx_nodes(G, pos, listCovNode, node_color='green',  node_size=40, label='Lidar' )
pos2 = {node: node for node in G2.nodes()}
nx.draw(G2, pos2, with_labels=False, node_color='black', node_size=40)
plt.show()
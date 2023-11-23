import os
from platform import node
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.problems import QuadraticProgram
import networkx as nx
from qiskit_optimization.converters import QuadraticProgramToQubo
import math
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()


points1 = [] #placement of lidars for one side
points2 = [] #placement of lidars for other side
points3 = [] #vertices to cover


for i in range(1, 4, 1):
    point = (i, 1)
    points1.append(point)

for i in range(1, 4, 1):
    point = (i, 4)
    points2.append(point)

for i in range(2, 3):
    for j in range(2, 4):
        point = (i, j)
        points3.append(point)

G.add_nodes_from(points1 + points2 + points3)


#edge that represents scope between lidar placement and vertices to cover
for node1 in points1:
  for j in range (node1[1]+1, node1[1]+3):
    if(j==node1[1]+1):
      for i in range(node1[0]-2, node1[0]+3):
        if (i,j) in points3:
          G.add_edge(tuple(node1), (i,j))
    if(j== node1[1]+2):
      for i in range(node1[0]-1, node1[0]+2):
        if (i,j) in points3:
          G.add_edge(tuple(node1), (i,j))
for node2 in points2:
  for j in range (node2[1]-2,node2[1]):
    if(j== node2[1]-2):
      for i in range(node2[0]-1, node2[0]+2):
        if (i,j) in points3:
          G.add_edge(tuple(node2), (i,j))
    if(j==node2[1]-1):
      for i in range(node2[0]-2, node2[0]+3):
        if (i,j) in points3:
          G.add_edge(tuple(node2), (i,j))

#Draw the graph

pos = {node: node for node in G.nodes()} 
nx.draw(G, pos,with_labels=True)

# plt.xlim(-1, 13)
# plt.ylim(-2, 7)

# plt.xticks(range(0, 13))
# plt.yticks(range(1, 7))

plt.show()

def showGraph(G, S):


    actE =[]
    for node in S:
      for edge in G.edges(node):
        actE.append(edge)
    # Draw the graph
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=True)

    nx.draw_networkx_nodes(G, pos, S, node_color='red')
    nx.draw_networkx_edges(G, pos, actE, edge_color='red')
    plt.show()


m = Model(name='BMW') #compare result log complexity graph generator function that compare quality of the result evaluating resumt (numbre of nodes & number of qubit)
#ten node how to compare how many qubit needs for alll approach and bard thing??? 3 items and x number qubit need quality of nulmber of acticvate node 
#do the pubo on simulating annnealing threatten sa function can be re-use 
pointsL = points1 + points2
x = m.binary_var_dict(pointsL, name='x') 
m.objective_expr = sum(x[i] for i in pointsL) #minimize the placement of lidars
m.objective_sense = 'min'

for node in points3:
    m.add_constraint(1 <= sum(x[v] for v in G.neighbors(node)))
print(m.prettyprint())

output = m.pprint_as_string()

m.export_as_lp(basename="BMW4", path=os.path.abspath(""))
sol_model = m.solve()
m.print_solution()
S = []
for (x, y) in sol_model.iter_var_values():
  sx = str(x)
  if sx[0] == 'x':
    S.append((int(sx[2]), int(sx[4])))
showGraph(G, S)

quadratic_program4 = QuadraticProgram()

quadratic_program4.read_from_lp_file(os.path.join(os.path.abspath(""), 'BMW4.lp'))

print("quadratic program 4 = ", quadratic_program4.prettyprint())
conv = QuadraticProgramToQubo()
qp4 = conv.convert(quadratic_program4)

print("qp4 = ", qp4)
print("matrice lenght : ", len(qp4.objective.quadratic.coefficients.asformat("array")))
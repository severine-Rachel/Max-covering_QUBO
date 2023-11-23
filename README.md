# BMW Sensor Positioning usecase

## Description
BMW usecase require a program to test differents QUBO formulation in order to solve a max covering optimization problem. The factory need to watch a specific zone with vehicles, some zone need more survey due to human passage. So to be sure the lidars will cover a specific zone (in a big zone that can not be 100% covered), some zone can add a priority factor which will add some point to a specific area. Each node which is not covered do not fit for the constraint and add value to the minimization function. Some point can be add to create obstacle. The lidars that avoid node behind obstacle despite the good range.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the needed libraries.
`pip install -r requirements.txt`

## Usage

Run the main.py program

It will plot an random graph created from src/Graph.py representing the wall of your place to watch with sometime a node circled by other nodes and edges in round representing an obstacle. (Note that sometime if the obstacle is too big, the programe can not place any node to cover and crash)

All the information of this graph are saved in a file named graph.json in the folder ressources 

The program plot a second graph with all correspondance between lidars possible positions (node in purple) and node to cover (node in cyan) from src/AllQUBOPUBO.py. Each edge represents the possible nodes to cover by each lidar.

The file allQUBOPUBO.py create lp files to describe the QUBO formulation and then create .npy file contained in /ressources respectively in /lpFiles and /matrix folders.

Finaly, the program plot a boxplot with the percentage of node covered and number of activated lidars from src/Annealer.py after running the simulated annealing (the running time depends of the size of the initial graph)

## File Legacy

In this folder there are all program that have been created furing this project. I describe the most important work

### All-QUBO-Compare.ipynb

Here is a calculation of performance for all QUBO, at the end, the program plot how QUBO formulation arrive to max covering depending of the number of activated lidars

### ComparisonComplexityQUBO.ipynb

Here is express all the step between QUBO formulation and matrix creation with boxplot that indicate the space complexity

### PuboOnGraph.py

Create a graph like street seen by far high, the informations needed is the nodes of the middle of each intersections with the width wanted of each street, the program will create a street view. Some problems persist because of the calculation of each wall in big intersection (more than three street connexion). Run a PUBO at the end.

### PuboOnSqArea.py

Basically the Graph.py file with PUBO on annealer running

## Contributing

Under the direction of Jonas Stein jonas.stein@ifi.lmu.de

## Authors and acknowledgment

Jonas Stein et Rachel Roux
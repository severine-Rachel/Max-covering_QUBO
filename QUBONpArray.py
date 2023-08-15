
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
#le qubo
conv = QuadraticProgramToQubo()
qp = conv.convert(quadratic_program)
print(qp.prettyprint())
#Binary variables = 138
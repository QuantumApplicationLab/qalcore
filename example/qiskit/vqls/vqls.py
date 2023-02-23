from qalcore.qiskit.vqls.vqls import VQLS
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.primitives import Estimator, Sampler 


A = np.random.rand(4, 4)
A = A + A.T

b = np.random.rand(4)

classical_solution = NumPyLinearSolver().solve(A, b / np.linalg.norm(b))

ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False)

estimator = Estimator()
sampler = Sampler()
vqls = VQLS(
    estimator,
    ansatz,
    COBYLA(maxiter=200, disp=True),
    sampler=sampler
)

opt= {"use_overlap_test": False,
      "use_local_cost_function": True}
res = vqls.solve(A, b, opt)


ref_solution = classical_solution.state / np.linalg.norm(classical_solution.state)
vqls_solution = np.real(Statevector(res.state).data)


plt.scatter(ref_solution, vqls_solution)
plt.plot([-1, 1], [-1, 1], "--")
plt.show()
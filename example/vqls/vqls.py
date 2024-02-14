from qalcore.qiskit.vqls import VQLS, VQLSLog
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms import optimizers as opt
from qiskit import Aer, BasicAer
import numpy as np

from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.primitives import Estimator, Sampler, BackendEstimator




A = np.random.rand(4, 4)
A = A + A.T

b = np.random.rand(4)

classical_solution = np.linalg.solve(A, b / np.linalg.norm(b))

ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False)



backend = BasicAer.get_backend("statevector_simulator")
# backend = BasicAer.get_backend("qasm_simulator")
options={"use_overlap_test": False, "use_local_cost_function": False}
estimator = BackendEstimator(backend)
estimator = Estimator()
sampler = Sampler()

vqls = VQLS(
    estimator,
    ansatz,
    opt.CG(maxiter=200),
    options=options,
    sampler=sampler
)


res = vqls.solve(A, b)


ref_solution = classical_solution / np.linalg.norm(classical_solution)
vqls_solution = np.real(Statevector(res.state).data)


plt.scatter(ref_solution, vqls_solution)
plt.plot([-1, 1], [-1, 1], "--")
plt.show()

plt.plot(log.values)
plt.ylabel('Cost Function')
plt.xlabel('Iterations')
plt.show()
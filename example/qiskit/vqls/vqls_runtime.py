from qalcore.qiskit.vqls import VQLS
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA, ADAM
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector

# from qiskit.primitives import Estimator, Sampler, BackendEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler, Session, Options



# define the problem
A = np.random.rand(4, 4)
A = A + A.T
b = np.random.rand(4)


# define ansatz
ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False)

opts = {"use_overlap_test": False,
        "use_local_cost_function": False}

# define the runtime
service = QiskitRuntimeService()
# backend = "ibmq_qasm_simulator"
# backend =  "simulator_statevector"
backend = "ibm_oslo"

with Session(service=service, backend=backend) as session:

    options = Options()
    # options.optimization_level = 3
    estimator = Estimator(session=session, options=options)

    # Sampler = Sampler(session=session, options=options)
    vqls = VQLS(
        estimator,
        ansatz,
        ADAM(maxiter=5, disp=True),
        options=opts
    )
    
    res = vqls.solve(A, b)


classical_solution = np.linalg.solve(A,b)

ref_solution = classical_solution/ np.linalg.norm(classical_solution)
vqls_solution = np.real(Statevector(res.state).data)


plt.scatter(ref_solution, vqls_solution)
plt.plot([-1, 1], [-1, 1], "--")
plt.show()
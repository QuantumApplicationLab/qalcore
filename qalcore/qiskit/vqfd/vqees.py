from qalcore.qiskit.vqls.variational_linear_solver import VariationalLinearSolver, VariationalLinearSolverResult
from qalcore.qiskit.vqfd.vqap import VQAP
import numpy as np
from qiskit.quantum_info import Statevector
from typing import List

class VQAP_IE(VQAP):

    def __init__(self, delta_x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_x = delta_x

    def process_probability_circuit_output(
        self,
        probability_circuit_output: List,
        return_norm: bool = False
    ) -> float:
        """Compute the final cost function from the sampled circuit values

        Args:
            probability_circuit_output (List): _description_
            weights (List): _description_

        Returns:
            float: _description_
        """
        if return_norm:
            numerator = probability_circuit_output[0]
        else:
            numerator = -0.5*probability_circuit_output[0]**2
            
        denominator = 3.0 + self.delta_x*(probability_circuit_output[1] + probability_circuit_output[2])      
        return numerator/denominator + self.tol

class VQEES():

    def __init__(self, solver: VQAP):
        self.solver = solver
        self.num_qubits = solver.num_qubits 

    def process_solution(self, res):
        vqap_solution = np.real(Statevector(res.state).data)
        return vqap_solution


    def solve(self, initial_condition: np.ndarray,
        tmax: float, dt: float, t0: float = 0.0):

        solution = []
        solution.append(initial_condition)
        time = np.arange(t0,tmax,dt)
        nt = len(time)

        for i in range(nt):
            
            # prepare the rhs
            source = solution[i]
            source /= np.linalg.norm(source)

            # solve the linear system
            sol = self.process_solution(self.solver.solve(source))

            # store the solution
            norm = np.linalg.norm(sol)
            solution.append(norm*sol)

        return solution
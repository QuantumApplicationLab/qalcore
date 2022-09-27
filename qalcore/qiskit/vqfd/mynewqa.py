from qiskit import QuantumCircuit
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm
from qalcore.qiskit.vqls.hadamard_test import HadammardTest
from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer
from typing import Optional, Union, List, Callable
from qalcore.qiskit.vqls.variational_linear_solver import (
    VariationalLinearSolver,
    VariationalLinearSolverResult,
)
import numpy as np

class VQLS(VariationalAlgorithm, VariationalLinearSolver):

    def __init__(self, 
                ansatz: Optional[QuantumCircuit] = None,
                optimizer: Optional[Union[Optimizer, Minimizer]] = None, ):
        ...

    def construct_circuits(self, A, b) -> List[QuantumCircuit]:
        ...

    def get_cost_evaluation_function(self, circuits) -> Callable:

    def solve(self, A: np.ndarray, b: np.ndarray) -> VariationalLinearSolverResult:

        # construct all required circuit 
        circuits = self.construct_circuits(A, b)

        # define a callable cost function
        cost_evaluation = self.get_cost_evaluation_function(circuits)

        # optimize
        opt_results = self.optimizer(fun=cost_evaluation, ... )

        # return the result
        return VariationalLinearSolverResult(opt_results)
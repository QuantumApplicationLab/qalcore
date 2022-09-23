from sympy import Symbol
from sympy.matrices import Matrix
import numpy as np
from qalcore.dwave.qubols.encodings import RealUnitQbitEncoding
from dwave.system import DWaveSampler , EmbeddingComposite
import neal
from dimod import ExactSolver

from .solution_vector import SolutionVector

class QUBOLS:

    def __init__(self, A, b):
        """Linear Solver using QUBO

        Args:
            A (np.matrix): matrix of the linear system
            b (np.array): rhs of the linear system
        """

        self.A = A 
        self.b = b
        self.size = A.shape[0]
        

    def solve(self, sampler = neal.SimulatedAnnealingSampler(), 
                    encoding=RealUnitQbitEncoding, nqbit=11, **kwargs):
        """Solve the linear system

        Args:
            sampler (_type_, optional): _description_. Defaults to neal.SimulatedAnnealingSampler().
            encoding (_type_, optional): _description_. Defaults to RealUnitQbitEncoding.
            nqbit (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """

        sol = SolutionVector(size=self.size, nqbit=nqbit, encoding=encoding)
        self.x = sol.create_polynom_vector()
        self.qubo_dict = self.create_qubo_matrix(self.x)

        self.sampleset = sampler.sample_qubo(self.qubo_dict,**kwargs)
        self.lowest_sol = self.sampleset.lowest()
        return sol.decode_solution(self.lowest_sol.record[0][0])

    def create_qubo_matrix(self, x):
        """Create the QUBO dictionary requried by dwave solvers
        to solve the linear system

        A x = b

        Args:
            Anp (np.array): matrix of the linear system
            bnp (np.array): righ hand side of the linear system
            x (sympy.Matrix): unknown

        Returns:
            _type_: _description_
        """
        A = Matrix(self.A)
        b = Matrix(self.b)

        polynom = x.T @ A.T @ A @ x - x.T @ A.T @ b - b.T@ A @ x + b.T @ b
        polynom = polynom[0]
        polynom = polynom.expand()
        polynom = polynom.as_ordered_terms()

        out = dict()

        for term in polynom:
            m = term.args
            if len(m) == 0:
                continue

            if len(m) == 2:
                varname = str(m[1])
                varname = varname.split("**")[0]
                key = (varname , varname)

            elif len(m) == 3:
                key = (str(m[1]),str(m[2]))

            if key not in out:
                out[key] = 0.0

            out[key] += m[0]

        return out

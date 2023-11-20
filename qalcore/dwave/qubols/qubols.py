from sympy import Symbol
from sympy.matrices import Matrix, SparseMatrix
import numpy as np
from qalcore.dwave.qubols.encodings import RealUnitQbitEncoding
from typing import Optional, Union, List, Callable, Dict, Tuple
from dwave.system import DWaveSampler , EmbeddingComposite
import neal
from dimod import ExactSolver
import scipy.sparse as spsp
from .solution_vector import SolutionVector

class QUBOLS:

    def __init__(self, options: Optional[Union[Dict, None]] = None):
        """Linear Solver using QUBO

        Args:
            options: dictionary of options for solving the linear system
        """

        self.default_solve_options = {
            "sampler": neal.SimulatedAnnealingSampler(),
            "encoding": RealUnitQbitEncoding,
            "num_qbits": 11,
            "num_reads": 100,
            "verbose": False
        }
        self.options = self._validate_solve_options(options)
        self.sampler = self.options.pop('sampler')

    def _validate_solve_options(self, options: Union[Dict, None]) -> Dict:
        """validate the options used for the solve methods

        Args:
            options (Union[Dict, None]): options
        """
        valid_keys = self.default_solve_options.keys()

        if options is None:
            options = self.default_solve_options

        else:
            for k in options.keys():
                if k not in valid_keys:
                    raise ValueError(
                        "Option {k} not recognized, valid keys are {valid_keys}"
                    )
            for k in valid_keys:
                if k not in options.keys():
                    options[k] = self.default_solve_options[k]

        return options


    def solve(self, 
              matrix: np.ndarray,
              vector: np.ndarray ):
        """Solve the linear system

        Args:
            sampler (_type_, optional): _description_. Defaults to neal.SimulatedAnnealingSampler().
            encoding (_type_, optional): _description_. Defaults to RealUnitQbitEncoding.
            nqbit (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """

        self.A = matrix 
        self.b = vector
        self.size = self.A.shape[0]
    
        sol = SolutionVector(size=self.size, 
                             nqbit=self.options['num_qbits'], 
                             encoding=self.options['encoding'])
        self.x = sol.create_polynom_vector()
        self.qubo_dict = self.create_qubo_matrix(self.x)

        self.sampleset = self.sampler.sample_qubo(self.qubo_dict, num_reads = self.options['num_reads'])
        self.lowest_sol = self.sampleset.lowest()
        return sol.decode_solution(self.lowest_sol.record[0][0])

    def create_qubo_matrix(self, x, prec=None):
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
        if isinstance(self.A, spsp.spmatrix):
            A = SparseMatrix(*self.A.shape, dict(self.A.todok().items()))
        else:
            A = Matrix(self.A)

        if isinstance(self.b, spsp.spmatrix):
            b = SparseMatrix(*self.b.shape, dict(self.b.todok().items()))
        else:
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

        if prec is None:
            return out

        elif prec is not None:
            nremoved = 0
            out_cpy = dict()
            for k, v in out.items():
                if np.abs(v)>prec:
                    out_cpy[k] = v
                else:
                    nremoved += 1
            print('Removed %d elements' %nremoved)
            return out_cpy


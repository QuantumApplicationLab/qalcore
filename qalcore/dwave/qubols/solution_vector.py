from sympy import Symbol
from sympy.matrices import Matrix
import numpy as np
from qalcore.dwave.utils.encodings import RealUnitQbitEncoding
from dwave.system import DWaveSampler , EmbeddingComposite
import neal
from dimod import ExactSolver

class SolutionVector(object):

    def __init__(self, size, nqbit, encoding, base_name = 'x'):
        """Encode the solution vector in a list of RealEncoded

        Args:
            size (int): number of unknonws in the vector (i.e. size of the system)
            nqbit (int): number of qbit required per unkown
            base_name (str, optional): base name of the unknowns Defaults to 'x'.
            only_positive (bool, optional):  Defaults to False.
        """
        self.size = size
        self.nqbit = nqbit
        self.base_name = base_name
        self.encoding = encoding
        self.encoded_reals = self.create_encoding()

    def create_encoding(self):
        """Create the eocnding for all the unknowns


        Returns:
            list[RealEncoded]:
        """
        encoded_reals = []
        for i in range(self.size):
            var_base_name = self.base_name + str(i+1)
            encoded_reals.append(self.encoding(self.nqbit, var_base_name))

        return encoded_reals

    def create_polynom_vector(self):
        """Create the list of polynom epxressions

        Returns:
            sympy.Matrix: matrix of polynomial expressions
        """
        pl = []
        for real in self.encoded_reals:
            pl.append(real.create_polynom())

        return Matrix(pl)

    def decode_solution(self, data):

        sol = []
        for i, real in enumerate(self.encoded_reals):
            local_data = data[i*self.nqbit:(i+1)*self.nqbit]
            sol.append(real.decode_polynom(local_data))
        return np.array(sol)
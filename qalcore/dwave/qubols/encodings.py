from sympy import Symbol
from sympy.matrices import Matrix
import numpy as np


class BaseQbitEncoding(object):

    def __init__(self, nqbit, var_base_name):
        """Encode a  single real number in a

        Args:
            nqbit (int): number of qbit required in the expansion
            var_base_name (str): base names of the different qbits
            only_positive (bool, optional): Defaults to False.
        """
        self.nqbit = nqbit
        self.var_base_name = var_base_name
        self.variables = self.create_variable()


    def create_variable(self):
        """Create all the variabes/qbits required for the expansion

        Returns:
            list: list of Symbol
        """
        variables = []
        for i in range(self.nqbit):
            variables.append(Symbol(self.var_base_name + '_%03d' %(i+1)))
        return variables

class EfficientEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0
        self.int_max = 2**(nqbit-1)-1

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = -2**(self.nqbit-1) * self.variables[0]
        for i in range(self.nqbit-1):
            out += 2**(i) * self.variables[i+1]
        return out/self.int_max
    
    def decode_polynom(self, data):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = -2**(self.nqbit-1) * data[0]
        for i in range(self.nqbit-1):
            out += 2**(i) * data[i+1]
        return out/self.int_max

class RealQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0
        for i in range(self.nqbit//2):
            out += 2**(i-self.base_exponent) * self.variables[i]
            out -= 2**(i-self.base_exponent) * self.variables[self.nqbit//2+i]
        return out

    def decode_polynom(self, data):
        out = 0.0
        for i in range(self.nqbit//2):
            out += 2**(i-self.base_exponent) * data[i]
            out -= 2**(i-self.base_exponent) * data[self.nqbit//2+i]
        return out

class RealUnitQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0
        self.int_max = None
        assert((nqbit-1)%2==0)

    def find_int_max(self):
        """Find the amx value of the encoding
        """
        i = 0
        self.int_max = 2**(i-self.base_exponent)

        for i in range(1, (self.nqbit-1)//2):
            self.int_max += 2**(i-self.base_exponent)


    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0

        self.find_int_max()

        i = 0
        out += 2**(i-self.base_exponent)/self.int_max * self.variables[i]

        for i in range(1, (self.nqbit-1)//2+1):

            out += 2**(i-self.base_exponent)/self.int_max * self.variables[i]
            out -= 2**(i-self.base_exponent)/self.int_max * self.variables[(self.nqbit-1)//2+i]
        return out

    def decode_polynom(self, data):
        out = 0.0
        
        if self.int_max is None:
            self.find_int_max()

        i=0
        out += 2**(i-self.base_exponent)/self.int_max * data[i]

        for i in range(1, (self.nqbit-1)//2+1):
            out += 2**(i-self.base_exponent)/self.int_max * data[i]
            out -= 2**(i-self.base_exponent)/self.int_max * data[(self.nqbit-1)//2+i]
        return out

class PositiveQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0
        for i in range(self.nqbit):
            out += 2**i * self.variables[i]
        return out

    def decode_polynom(self, data):
        out = 0.0
        for i in range(self.nqbit//2):
            out += 2**i * data[i]
            out -= 2**i * data[self.nqbit//2+i]
        return out
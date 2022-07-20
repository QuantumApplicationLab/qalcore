
import numpy as np 
import scipy.linalg as spla

class UnitaryDecomposition:
    # https://math.stackexchange.com/questions/1710247/every-matrix-can-be-written-as-a-sum-of-unitary-matrices/1710390#1710390
    
    def __init__(self, mat, thr=1E-6):
        self.mat = mat
        self.unit_coeffs = None
        self.unit_mats = None 
        self.thr = thr

    @staticmethod
    def get_real(x):
        """ Get the real part of x"""
        return 0.5 * (x + np.conjugate(x))

    @staticmethod
    def get_imag(x):
        """ Get the imaginary part of x """
        return 0.5/(1j) * (x - np.conjugate(x))

    @staticmethod
    def aux(x):
        """ Auxiliary function

        Performs a matrix operation that we'll need later
        """
        I = np.eye(len(x))
        x2 = np.linalg.matrix_power(x,2)
        return 1j*spla.sqrtm(I - x2)

    def decompose(self, check = False):

        """ Unitary decomposition

        Decomposes the complex normalized matrix X into four unitary matrices
        """

        # Normalize
        norm = np.linalg.norm(self.mat)
        X_n = self.mat / norm

        # Apply the algorithm as described in
        
        B = self.get_real(X_n)
        C = self.get_imag(X_n)

        ## Get the matrices
        UB = B + self.aux(B)
        UC = C + self.aux(C)
        VB = B - self.aux(B)
        VC = C - self.aux(C)

        ## Get the coefficients
        cb = norm * 0.5
        cc = cb * 1j

        if np.allclose(self.mat, self.mat.T.conj()):
            ## Return
            self.unit_coeffs = [cb,cb]
            self.unit_mats = [UB, VB]
        else:
            ## Return
            self.unit_coeffs = [cb,cb,cc,cc]
            self.unit_mats = [UB, VB, UC, VC]         

        # remove null matrices
        self.clean_matrices()

        if check:
            mat_recomp = self.recompose()
            assert(np.allclose(self.mat,mat_recomp))

    def normalize_coefficients(self):
        """Normalize the coefficients
        """

        sum_coeff = np.array(self.unit_coeffs).sum()
        self.unit_coeffs = [u/sum_coeff for u in self.unit_coeffs]

    def clean_matrices(self):
        """Remove matrices that are null
        """
        clean_mat = []
        clean_coeff = []
        for c,m in zip(self.unit_coeffs, self.unit_mats):
            if np.linalg.norm(m) > self.thr:
                clean_mat.append(m)
                clean_coeff.append(c)

        self.unit_coeffs = clean_coeff
        self.unit_mats = clean_mat


    def recompose(self):
        """ Rebuilds the original matrix from the decomposed one """
        recomp = np.zeros_like(self.unit_mats[0])
        for c, m in zip(self.unit_coeffs, self.unit_mats):
            recomp += c * m
        return recomp


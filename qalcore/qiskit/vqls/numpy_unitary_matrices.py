from collections import namedtuple
from types import SimpleNamespace
from typing import Optional, Union, List, Tuple, TypeVar, cast

import numpy as np
from numpy.testing import assert_
import numpy.typing as npt
import scipy.linalg as spla

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator


complex_t = TypeVar("complex_t", float, complex)
complex_arr_t = npt.NDArray[np.cdouble]


def auxilliary_matrix(x: Union[npt.NDArray[np.float_], complex_arr_t]) -> complex_arr_t:
    """Compute i * sqrt(I - x^2)

    Args:
        x (np.ndarray): input matrix

    Returns:
        np.ndarray: values of i * sqrt(I - x^2)
    """
    mat = np.eye(len(x)) - x @ x
    mat = cast(npt.NDArray[Union[np.float_, np.cdouble]], spla.sqrtm(mat))
    return 1.0j * mat


class Decomposition:
    """Decompose a matrix representing quantum circuits

    For the mathematical background for the decomposition, see the following
    math.sx answer: https://math.stackexchange.com/a/1710390

    """

    CircuitElement = namedtuple("CircuitElement", ["coeff", "circuit"])

    @classmethod
    def _as_complex(
        cls, num_or_arr: Union[complex_t, List[complex_t]]
    ) -> complex_arr_t:
        """Converts a number or a list of numbers to a complex array."""
        arr = num_or_arr if isinstance(num_or_arr, List) else [num_or_arr]
        return np.array(arr, dtype=np.cdouble)

    def __init__(
        self,
        matrix: Optional[npt.NDArray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
    ):
        """Decompose a matrix representing quantum circuits

        Parameters
        ----------
        matrix : npt.NDArray
            Array to decompose; only relevant in derived classes where
            `self.decompose_matrix()` has been implemented

        circuits : Union[QuantumCircuit, List[QuantumCircuit]]
            quantum circuits representing the matrix

        coefficients : Union[float, complex, List[float], List[complex]] (default: None)
            coefficients associated with the input quantum circuits; `None` is
            valid only for a circuit with 1 element

        """
        if matrix is not None:  # ignore circuits & coefficients
            self._matrix, self.num_qubits = self._validate_matrix(matrix)
            self._coefficients, self._matrices = self.decompose_matrix()
            self._circuits = self._create_circuits(self.matrices)
        elif circuits is not None:
            self._circuits: List[QuantumCircuit] = (
                circuits if isinstance(circuits, (list, tuple)) else [circuits]
            )
            assert_(
                isinstance(self._circuits[0], QuantumCircuit),
                f"{circuits}: invalid circuit",
            )
            if coefficients is None:
                if len(self._circuits) == 1:
                    self._coefficients = self._as_complex(1.0)
                else:
                    raise ValueError("coefficients mandatory for multiple circuits")
            else:
                self._coefficients = self._as_complex(coefficients)

            if len(self._circuits) != len(self._coefficients):
                raise ValueError("number of coefficients and circuits do not match")

            self.num_qubits: int = self._circuits[0].num_qubits
            if not all(map(lambda ct: ct.num_qubits == self.num_qubits, self.circuits)):
                _num_qubits = [ct.num_qubits for ct in self.circuits]
                raise ValueError(f"mismatched number of qubits: {_num_qubits}")

            self._matrices = [Operator(qc).data for qc in self.circuits]
            self._matrix = self.recompose()
        else:
            raise ValueError(
                f"inconsistent arguments: matrix={matrix}, coefficients={coefficients}, circuits={circuits}"
            )

        self.num_circuits = len(self._circuits)
        self.iiter = 0

    @classmethod
    def _compute_circuit_size(cls, matrix: npt.NDArray) -> int:
        """Compute the size of the circuit represented by the matrix."""
        return int(np.log2(matrix.shape[0]))

    @classmethod
    def _validate_matrix(cls, matrix: complex_arr_t) -> Tuple[complex_arr_t, int]:
        if len(matrix.shape) == 2 and matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Input matrix must be square: matrix.shape={matrix.shape}"
            )
        num_qubits = cls._compute_circuit_size(matrix)
        if num_qubits % 1 != 0:
            raise ValueError(
                f"Input matrix dimension is not a power of 2: {num_qubits}"
            )
        if not np.allclose(matrix, matrix.conj().T):  # FIXME: is this required?
            raise ValueError(f"Input matrix isn't Hermitian:\n{matrix}")
        return matrix, num_qubits

    def _create_circuits(self, unimatrices: List[np.ndarray]) -> List[QuantumCircuit]:
        """Construct the quantum circuits.

        Parameters
        ----------
        unimatrices : List[np.ndarray]
            list of unitary matrices of the decomposition.

        Returns
        -------
        List[QuantumCircuit]
            list of resulting quantum circuits.
        """

        def make_qc(mat: complex_arr_t) -> QuantumCircuit:
            qc = QuantumCircuit(self.num_qubits)
            qc.unitary(mat, qc.qubits)
            return qc

        return [make_qc(mat) for mat in unimatrices]

    @property
    def matrix(self) -> np.ndarray:
        """matrix of the decomposition"""
        return self._matrix

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """circuits of the decomposition"""
        return self._circuits

    @property
    def coefficients(self) -> complex_arr_t:
        """coefficients of the decomposition."""
        return self._coefficients

    @property
    def matrices(self) -> List[complex_arr_t]:
        """return the unitary matrices"""
        return self._matrices

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.num_circuits:
            out = self.CircuitElement(
                self._coefficients[self.iiter], self._circuits[self.iiter]
            )
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self._circuits)

    def __getitem__(self, index):
        return self.CircuitElement(self._coefficients[index], self._circuits[index])

    def recompose(self) -> complex_arr_t:
        """Rebuilds the original matrix from the decomposed one.

        Returns:
            np.ndarray: recomposed matrix
        """
        coeffs, matrices = self.coefficients, self.matrices
        return (coeffs.reshape(len(coeffs), 1, 1) * matrices).sum(axis=0)

    def decompose_matrix(self) -> Tuple[complex_arr_t, List[complex_arr_t]]:
        raise NotImplementedError(f"can't decompose in {self.__class__.__name__!r}")


class UnitaryDecomposition(Decomposition):
    def decompose_matrix(
        self,
    ) -> Tuple[complex_arr_t, List[complex_arr_t]]:
        """Decompose a generic numpy matrix into a sum of unitary matrices

        Returns:
            Tuple: list of coefficients and numpy matrix of the decompostion
        """

        # Normalize
        norm = np.linalg.norm(self._matrix)
        mat = self._matrix / norm

        mat_real = np.real(mat)
        mat_imag = np.imag(mat)

        coef_real = norm * 0.5
        coef_imag = coef_real * 1j

        ## Get the matrices
        unitary_matrices, unitary_coefficients = [], []
        if not np.allclose(mat_real, 0.0):
            aux_mat = auxilliary_matrix(mat_real)
            unitary_matrices += [mat_real + aux_mat, mat_real - aux_mat]
            unitary_coefficients += [coef_real] * 2

        if not np.allclose(mat_imag, 0.0):
            aux_mat = auxilliary_matrix(mat_imag)
            unitary_matrices += [mat_imag + aux_mat, mat_imag - aux_mat]
            unitary_coefficients += [coef_imag] * 2
        unit_coeffs = np.array(unitary_coefficients, dtype=np.cdouble)

        return unit_coeffs, unitary_matrices







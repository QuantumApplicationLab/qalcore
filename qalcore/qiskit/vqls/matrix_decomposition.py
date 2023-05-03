from collections import namedtuple
from itertools import product
from typing import Optional, Union, List, Tuple, TypeVar, cast

import numpy as np
from numpy.testing import assert_
import numpy.typing as npt
import scipy.linalg as spla

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli


complex_t = TypeVar("complex_t", float, complex)
complex_arr_t = npt.NDArray[np.cdouble]

class MatrixDecomposition:
    """Base class for the decomposition of a matrix in quantum circuits.
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
            self._coefficients, self._matrices, self._circuits = self.decompose_matrix()
            
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
        if not np.allclose(matrix, matrix.conj().T): 
            raise ValueError(f"Input matrix isn't symmetric:\n{matrix}")
        
        return matrix, num_qubits



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
        """
        Rebuilds the original matrix from the decomposed one.

        Returns
        -------
        np.ndarray
            The recomposed matrix.

        See Also
        --------
        decompose_matrix : Decompose a generic numpy matrix into a sum of unitary matrices.
        """
        coeffs, matrices = self.coefficients, self.matrices
        return (coeffs.reshape(len(coeffs), 1, 1) * matrices).sum(axis=0)

    def decompose_matrix(self) -> Tuple[complex_arr_t, List[complex_arr_t], List[QuantumCircuit]]:
        raise NotImplementedError(f"can't decompose in {self.__class__.__name__!r}")


class SymmetricDecomposition(MatrixDecomposition):
    """
    A class that represents the symmetric decomposition of a matrix.
    For the mathematical background for the decomposition, see the following
    math.sx answer: https://math.stackexchange.com/a/1710390

    Methods
    -------
    decompose_matrix() -> Tuple[complex_arr_t, List[complex_arr_t]]:
        Decompose a generic numpy matrix into a sum of unitary matrices.

    See Also
    --------
    MatrixDecomposition : A base class for matrix decompositions.
    recompose : Rebuilds the original matrix from the decomposed one.
    """

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


    @staticmethod
    def auxilliary_matrix(x: Union[npt.NDArray[np.float_], complex_arr_t]) -> complex_arr_t:
        """
        Returns the auxiliary matrix for the decomposition of size n.

        Parameters
        ----------
        x : np.ndarray
            original matrix.

        Returns
        -------
        np.ndarray
            The auxiliary matrix.

        Notes
        -----
        The auxiliary matrix is defined as : i * sqrt(I - x^2)

        """
        mat = np.eye(len(x)) - x @ x
        mat = cast(npt.NDArray[Union[np.float_, np.cdouble]], spla.sqrtm(mat))
        return 1.0j * mat

    def decompose_matrix(
        self,
    ) -> Tuple[complex_arr_t, List[complex_arr_t], List[QuantumCircuit]]:
        """
        Decompose a generic numpy matrix into a sum of unitary matrices.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to be decomposed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the list of coefficients and the numpy matrix of the decomposition.

        See Also
        --------
        recompose : Rebuilds the original matrix from the decomposed one.
        """
        # Normalize
        norm = np.linalg.norm(self._matrix)
        mat = self._matrix / norm

        mat_real = np.real(mat)
        mat_imag = np.imag(mat)

        coef_real = norm * 0.5
        coef_imag = coef_real * 1j

        # Get the matrices
        unitary_matrices, unitary_coefficients = [], []
        if not np.allclose(mat_real, 0.0):
            aux_mat = self.auxilliary_matrix(mat_real)
            unitary_matrices += [mat_real + aux_mat, mat_real - aux_mat]
            unitary_coefficients += [coef_real] * 2

        if not np.allclose(mat_imag, 0.0):
            aux_mat = self.auxilliary_matrix(mat_imag)
            unitary_matrices += [mat_imag + aux_mat, mat_imag - aux_mat]
            unitary_coefficients += [coef_imag] * 2
        unit_coeffs = np.array(unitary_coefficients, dtype=np.cdouble)

        # create the circuits
        circuits = self._create_circuits(unitary_matrices)

        return unit_coeffs, unitary_matrices, circuits


class PauliDecomposition(MatrixDecomposition):
    """
    A class that represents the Pauli decomposition of a matrix.

    Attributes
    ----------
    basis : str
        The basis of Pauli gates used for the decomposition.

    Methods
    -------
    decompose_matrix() -> Tuple[complex_arr_t, List[complex_arr_t]]:
        Decompose a matrix into a sum of Pauli strings.

    See Also
    --------
    MatrixDecomposition : A base class for matrix decompositions.
    """

    basis = "IXYZ"

    @staticmethod
    def _create_circuit(pauli_string: str) -> QuantumCircuit:
        """creates a quantum circuit for a given pauli string

        Args:
            pauli_string (str): the input pauli string

        Returns:
            QuantumCircuit: quantum circuit for the string 
        """
        num_qubit = len(pauli_string)
        qc = QuantumCircuit(num_qubit)
        for iqbit, gate in enumerate(pauli_string[::-1]):
            qc.__getattribute__(gate.lower())(iqbit)
        return qc

    def decompose_matrix(self) -> Tuple[complex_arr_t, List[complex_arr_t], List[QuantumCircuit]]:
        """Decompose a generic numpy matrix into a sum of Pauli strings.

        Returns:
            Tuple[complex_arr_t, List[complex_arr_t]]: 
                A tuple containing the list of coefficients and the numpy matrix of the decomposition.
        """

        prefactor = 1.0 / (2**self.num_qubits)
        unit_mats, coeffs, circuits = [], [], []

        for pauli_gates in product(self.basis, repeat=self.num_qubits):

            pauli_string = "".join(pauli_gates)
            pauli_op = Pauli(pauli_string)
            pauli_matrix = pauli_op.to_matrix()
            coef: complex_arr_t = np.trace(pauli_matrix @ self.matrix)

            if coef * np.conj(coef) != 0:
                coeffs.append(prefactor * coef)
                unit_mats.append(pauli_matrix)
                circuits.append(self._create_circuit(pauli_string))

        return np.array(coeffs, dtype=np.cdouble), unit_mats, circuits

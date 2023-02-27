from types import SimpleNamespace
from typing import Optional, Union, List, Tuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg as spla

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator


complex_t = TypeVar("complex_t", float, complex)


class Decomposition:
    r"""Compute the unitary decomposition of a general matrix
    See:
        https://math.stackexchange.com/questions/1710247/every-matrix-can-be-written-as-a-sum-of-unitary-matrices/1710390#1710390
    """

    @classmethod
    def _as_complex(
        cls, num_or_arr: Union[complex_t, List[complex_t]]
    ) -> npt.NDArray[np.cdouble]:
        arr = num_or_arr if isinstance(num_or_arr, List) else [num_or_arr]
        return np.array(arr, dtype=np.cdouble)

    @classmethod
    def _compute_circuit_size(cls, matrix: npt.NDArray) -> int:
        return int(np.log2(matrix.shape[0]))

    @classmethod
    def _validate_matrix(
        cls, matrix: npt.NDArray[np.cdouble]
    ) -> Tuple[npt.NDArray[np.cdouble], int]:
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

    def __init__(
        self,
        matrix: Optional[np.ndarray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
        check_decomposition: Optional[bool] = True,
        normalize_coefficients: Optional[bool] = True,
    ):
        """Unitary decomposition

        Args:
            matrix (Optional[np.ndarray], optional): input matrix to be transformed.
            circuit (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional): quantum circuit(s) representing the matrix.
            coefficients (Optional[Union[float, complex, List[float], List[complex]]], optional): coefficients of associated with the input quantum circuits.
            check_decomposition (Optional[bool], optional): Check if the decomposition matches the input matrix. Defaults to True.
            normalize_coefficients (Optional[bool], optional): normalize the coefficients of the decomposition. Defaults to True.

        - matrix: NDArray[np.cdouble]
        - coefficients: NDArray[np.cdouble]
        - circuits: List[QuantumCircuit]
        - unitary_matrices: List[NDArray[np.cdouble]]

        """

        if matrix is not None:  # ignore circuits & coefficients
            self._matrix, self.num_qubits = self._validate_matrix(matrix)
            self._coefficients, self._unitary_matrices = self.decompose_matrix(
                check=check_decomposition, normalize_coefficients=normalize_coefficients
            )

            self._circuits = self.create_circuits(self.unitary_matrices)
        elif circuits is not None:
            self._circuits: List[QuantumCircuit] = (
                circuits if isinstance(circuits, (list, tuple)) else [circuits]
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
            if not all(map(lambda i: i == self.num_qubits, self._circuits)):
                raise ValueError(f"mismatched number of qubits: {self._circuits}")

            self._unitary_matrices = [Operator(qc).data for qc in self._circuits]

            self._matrix = self.recompose(self.coefficients, self.circuits)
        else:
            raise ValueError(
                f"inconsistent arguments: matrix={matrix}, coefficients={coefficients}, circuits={circuits}"
            )

        self.num_circuits = len(self._circuits)
        self.iiter = 0

    @property
    def matrix(self) -> np.ndarray:
        """matrix of the decomposition"""
        return self._matrix

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """circuits of the decomposition"""
        return self._circuits

    @property
    def coefficients(self) -> npt.NDArray[np.cdouble]:
        """coefficients of the decomposition."""
        return self._coefficients

    @property
    def unitary_matrices(self) -> List[npt.NDArray[np.cdouble]]:
        """return the unitary matrices"""
        return self._unitary_matrices

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.num_circuits:
            out = SimpleNamespace(
                coeff=self._coefficients[self.iiter], circuit=self._circuits[self.iiter]
            )
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self._circuits)

    def __getitem__(self, index):
        return SimpleNamespace(
            coeff=self._coefficients[index], circuit=self._circuits[index]
        )

    def recompose(
        self,
        unit_coeffs: npt.NDArray[np.cdouble],
        unit_mats: List[npt.NDArray[np.cdouble]],
    ) -> npt.NDArray[np.cdouble]:
        """Rebuilds the original matrix from the decomposed one.

        Args:
            unit_coeffs (List[float]): coefficients of the decomposition
            unit_mats (List[np.ndarray]): matrices of the decomposition

        Returns:
            np.ndarray: recomposed matrix
        """
        return (unit_coeffs.reshape(len(unit_coeffs), 1, 1) * unit_mats).sum(axis=0)

    def create_circuits(self, unit_mats: List[np.ndarray]) -> List[QuantumCircuit]:
        """Contstruct the quantum circuits.

        Args:
            unit_mats (List[np.ndarray]): list of unitary matrices of the decomposition.

        Returns:
            List[QuantumCircuit]: list of resulting quantum circuits.
        """

        def make_qc(mat: npt.NDArray[np.cdouble]) -> QuantumCircuit:
            qc = QuantumCircuit(self.num_qubits)
            qc.unitary(mat, qc.qubits)
            return qc

        return [make_qc(mat) for mat in unit_mats]

    def normalize_coefficients(
        self, unit_coeffs: npt.NDArray[np.cdouble]
    ) -> npt.NDArray[np.cdouble]:
        """Normalize the coefficients

        Args:
            unit_coeffs (npt.NDArray[np.cdouble]): list of coefficients

        Returns:
            npt.NDArray[np.cdouble]: List of normalized coefficients
        """
        return unit_coeffs / unit_coeffs.sum()

    @classmethod
    def get_auxilliary_matrix(
        cls, x: npt.NDArray[Union[np.float_, np.cdouble]]
    ) -> npt.NDArray[np.cdouble]:
        """Compute i * sqrt(I - x^2)

        Args:
            x (np.ndarray): input matrix

        Returns:
            np.ndarray: values of i * sqrt(I - x^2)
        """
        mat = np.eye(len(x)) - x @ x
        mat = cast(npt.NDArray[Union[np.float_, np.cdouble]], spla.sqrtm(mat))
        return 1.0j * mat

    def decompose_matrix(
        self,
        check: Optional[bool] = False,
        normalize_coefficients: Optional[bool] = False,
    ) -> Tuple[npt.NDArray[np.cdouble], List[npt.NDArray[np.cdouble]]]:
        raise NotImplementedError(f"can't decompose in {self.__class__.__name__!r}")


class UnitaryDecomposition(Decomposition):
    def decompose_matrix(
        self,
        check: Optional[bool] = False,
        normalize_coefficients: Optional[bool] = False,
    ) -> Tuple[npt.NDArray[np.cdouble], List[npt.NDArray[np.cdouble]]]:
        """Decompose a generic numpy matrix into a sum of unitary matrices

        Args:
            check (Optional[bool], optional): _description_. Defaults to False.
            normalize_coefficients (Optional[bool], optional): _description_. Defaults to False.

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
            aux_mat = self.get_auxilliary_matrix(mat_real)
            unitary_matrices += [mat_real + aux_mat, mat_real - aux_mat]
            unitary_coefficients += [coef_real] * 2

        if not np.allclose(mat_imag, 0.0):
            aux_mat = self.get_auxilliary_matrix(mat_imag)
            unitary_matrices += [mat_imag + aux_mat, mat_imag - aux_mat]
            unitary_coefficients += [coef_imag] * 2
        unit_coeffs = np.array(unitary_coefficients, dtype=np.cdouble)

        if check:
            mat_recomp = self.recompose(unit_coeffs, unitary_matrices)
            assert np.allclose(self._matrix, mat_recomp)

        if normalize_coefficients:
            unit_coeffs = self.normalize_coefficients(unit_coeffs)

        return unit_coeffs, unitary_matrices







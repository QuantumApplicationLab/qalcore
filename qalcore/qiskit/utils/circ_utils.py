from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit import execute, Aer
import numpy as np


def vector_creation(desired_vector, nqbit, decimals=6):
    """Get the unitary matrix required to create a given vector

    .. math:
        U |0\rangle \rightarrow |v\rangle

    Args:
        desired_vector (np.array): the vector we want to create
        nqbit (int): number of qbit required
        decimals (int, optional): precision desired

    Returns:
        Tuple(float, np.array): the norm and matrix 
    """

    # normalize the vector
    norm = np.linalg.norm(desired_vector)
    desired_vector = np.asarray(desired_vector).flatten()
    desired_vector /= norm

    # create the circuit
    qc = QuantumCircuit(nqbit) 
    qc.initialize(desired_vector, list(range(nqbit))) 
    usimulator=Aer.get_backend('unitary_simulator')

    # execture the circuit
    job = execute(qc, usimulator)

    # get the unitary of the circuit
    umatrix = job.result().get_unitary(qc,decimals=decimals) 
    return norm, umatrix

def get_controlled_matrix(matrices, auxiliary, qubits):
    """Create the matrix corresponding to the controlled version of the operator

    Args:
        matrices (_type_): _description_
        auxiliary (_type_): _description_
        qubits (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert(auxiliary==0)
    nqbit = len(qubits)+1
    ctrl_matrices = []
    for m in matrices:
        cmat = np.eye(2**nqbit).astype('complex')
        cmat[1::2,1::2] = m
        ctrl_matrices.append(cmat)

    return ctrl_matrices


def apply_controlled_gate(circ, mat, auxiliary, qubits, name=None):
    """Apply a controlled operation on the circuit

    Args:
        circ (_type_): _description_
        mat (_type_): _description_
        auxiliary (_type_): _description_
        qubits (_type_): _description_
        name (_type_, optional): _description_. Defaults to None.
    """
    qc1 = QuantumCircuit(len(qubits)+1, name=name)
    op = Operator(mat)
    qc1.append(op, [auxiliary] + qubits)
    circ.append(qc1, [auxiliary] + qubits)
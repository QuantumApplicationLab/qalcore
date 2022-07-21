from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend
from qiskit.quantum_info.operators import Operator
from qiskit import execute, Aer
from qiskit import transpile, assemble
import numpy as np
from typing import List


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
    return norm, np.array(umatrix)

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


def unitarymatrix2circuit(A, backend, name=None):
    """Create the circuit associated with the backend

    Args:
        A (np.array): matrix to be transformed
        backend (qiskit.backend): backend to be used
    """
    nqbit = int(np.ceil(np.log2(A.shape[0])))
    config = backend.configuration()
    qc = QuantumCircuit(nqbit, name=name)
    qc.unitary(A, list(range(nqbit)))
    return transpile(qc, basis_gates=config.basis_gates, output_name=name)

def get_circuit_state_vector(circ: QuantumCircuit, 
                             backend: Backend, 
                             decimals: int =100):
    """Get the state vector of a give circuit after execution on the backend

    Args:
        circ (QuantumCircuit): circuit to get the statevector of
        backend (backend): backend
        decimals (int, optional): Numbers of decimals we need. Defaults to 100.
    """

    circ.save_statevector()
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)
    result = job.result()

    return result.get_statevector(circ, decimals=decimals)
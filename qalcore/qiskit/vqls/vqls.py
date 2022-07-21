# Variational Quantum Linear Solver
# Ref : 
# Tutorial :

import numpy as np
from scipy.optimize import minimize

import qiskit 
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit import Aer, transpile, assemble
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


from qalcore.utils.unitary_decomposition import UnitaryDecomposition
from qalcore.qiskit.utils.circ_utils import vector_creation, get_controlled_matrix, unitarymatrix2circuit
from qalcore.qiskit.utils.circ_utils import get_circuit_state_vector
from qalcore.qiskit.utils.circuit.hadammard_test import HadammardTest
from qalcore.qiskit.utils.circuit.special_hadammard_test import SpecialHadammardTest
from types import SimpleNamespace

class VQLS:

    def __init__(self, A, b):
        """Variational Quantum Linear Solver 

        .. math
            A x = b

        Args:
            A (np.array): matrix of the linear system
            b (np.array): rhs of the system
        """

        self.A = A
        self.b = b 
        self.backend = None
        self.system_size = self.A.shape[0]
        self.nqbit = int(np.log2(self.system_size))

    def solve(self, ansatz=None, niter=100, 
              backend=Aer.get_backend('aer_simulator'), verbose=False):

        self.backend = backend
        self.niter = niter
        self.iiter = 0 
        self.verbose=verbose

        # decompose the A matrix as a sum of unitary matrices
        unitdecomp_A = UnitaryDecomposition(self.A)
        unitdecomp_A.decompose(check=True)
        unitdecomp_A.normalize_coefficients()

        # get the circuit associated with the A matrices
        self.Acirc, self.Aconjcirc = self._get_A_circuit(unitdecomp_A)

        # get the matrix of the cricuit needed to create the rhs
        norm, Ub_mat = vector_creation(self.b, self.nqbit, decimals=16)

        # get the circuit needed to create the Ub_mat
        self.Ubconjcirc = unitarymatrix2circuit(Ub_mat.transpose(), self.backend)

        # variational ansatz
        if ansatz is None:
            self.ansatz = RealAmplitudes(self.nqbit, entanglement='linear', reps=2, insert_barriers=False)
        else:
            assert(type(ansatz)==qiskit.QuantumCircuit)
            self.ansatz = ansatz

        # initial value of the parameters
        x0 = np.random.rand(self.ansatz.num_parameters_settable)

        return minimize(self._cost, 
                        x0 = x0,
                        args=(self,),
                        method='COBYLA',
                        options={'maxiter':niter, 'disp':False})   


    def _get_A_circuit(self, umats):
        """Creates the circuit associated with the A matrices
        """
        acirc, aconjcirc = [], []
        for c, mat in zip(umats.unit_coeffs, umats.unit_mats):
            qc = unitarymatrix2circuit(mat, self.backend)
            acirc.append(SimpleNamespace(coeff=c, circuit=qc))

            qc = unitarymatrix2circuit(mat.transpose(), self.backend)
            aconjcirc.append(SimpleNamespace(coeff=c, circuit=qc))

        return acirc, aconjcirc

    @staticmethod
    def _cost(parameters, *args):
        """Computes the cost of the optimization

        Args:
            parameters (_type_): _description_

        Returns:
            _type_: _description_
        """

        (self,) = args
        self.iiter += 1
        hdmr_sum = self._compute_hadammard_sum(parameters)
        spec_hdmr_sum = self._compute_special_hadammard_sum(parameters)

        cost = 1.0 - (spec_hdmr_sum/hdmr_sum)

        if self.verbose:
            print('iteration %d/%d, cost %f' %(self.iiter, self.niter, cost))

        return cost.real

    def _assign_parameters(self, parameters):
        """Assign the parameter values to the circuit

        Args:
            parameters (_type_): _description_
        """
        
        bind_dict = {}
        for i, key in enumerate(self.ansatz.parameters):
            bind_dict[key] = parameters[i]
        return self.ansatz.assign_parameters(bind_dict, inplace=False)

    def _compute_hadammard_sum(self, parameters):
        """Compute the Hadammard sum

        .. math:

            \langle \Phi | \Phi \rangle \ = \ \displaystyle\sum_{m} \displaystyle\sum_{n} c_m^{*} c_n \langle 0 | V(k)^{\dagger} A_m^{\dagger} A_n V(k) |0\rangle

        Args:
            parameters (_type_): _description_

        Returns:
            _type_: _description_
        """

        hdmr_sum = 0.0 + 0.0j

        for circ_i in self.Acirc:
            for circ_j in self.Aconjcirc:

                prefac = circ_i.coeff.conj() * circ_j.coeff
                beta_ij = 0.0 + 0.0j

                for compute_imaginary_part in [False, True]:

                    local_ansatz = self._assign_parameters(parameters)

                    hdmr_circ = HadammardTest(
                        ansatz=local_ansatz,
                        operators=[circ_i.circuit, circ_j.circuit],
                        num_qubits=self.nqbit+1,
                        imaginary=compute_imaginary_part 
                    )

                    qctl = QuantumRegister(self.nqbit+1)
                    qc = ClassicalRegister(self.nqbit+1)
                    circ = QuantumCircuit(qctl, qc)

                    circ.compose(hdmr_circ, 
                                 qubits=list(range(self.nqbit+1)), 
                                 inplace=True)

                    state_vector = np.array(get_circuit_state_vector(circ, self.backend))
                    proba = 1.0 - 2.0 * (state_vector[::2]*state_vector[::2].conj()).sum()
                   
                    if compute_imaginary_part:
                        beta_ij += 1.0j * proba 
                    else:
                        beta_ij += proba 

                hdmr_sum += prefac * beta_ij

        return hdmr_sum.real


    def _compute_special_hadammard_sum(self, parameters):
        """Compute the special haddamard sum

        Args:
            parameters (_type_): _description_

        Returns:
            _type_: _description_
        """
        spec_hdmr_sum = 0.0 + 0.0j

        for circ_i in self.Acirc:
            for circ_j in self.Acirc:

                prefac = circ_i.coeff * np.conjugate(circ_j.coeff)
                mult_factor = 1.0
                gamma_ij = 0.0 + 0.0j

                for extra in [0,1]:

                    term = 0.0 + 0.0j

                    for compute_imaginary_part in [False, True]:

                        if extra == 0:
                            ops = [circ_i.circuit, self.Ubconjcirc]
                        else:
                            ops = [circ_j.circuit, self.Ubconjcirc]

                        local_ansatz = self._assign_parameters(parameters)

                        spec_hdmr_circ = SpecialHadammardTest(
                            ansatz=local_ansatz,
                            operators=ops,
                            num_qubits=self.nqbit+1,
                            imaginary=compute_imaginary_part 
                        )

                        qctl = QuantumRegister(self.nqbit+1)
                        qc = ClassicalRegister(self.nqbit+1)
                        circ = QuantumCircuit(qctl, qc)
                        circ.compose(spec_hdmr_circ, 
                                     qubits=list(range(self.nqbit+1)),
                                     inplace=True)

                        state_vector = np.array(get_circuit_state_vector(circ, self.backend))
                        proba = 1.0 - 2.0 * (state_vector[1::2]*state_vector[1::2].conj()).sum()

                        if compute_imaginary_part:
                            term += 1.0j * proba 
                        else:
                            term += proba
                        
                        mult_factor *= proba

                    if extra == 0:
                        gamma_ij += term
                    else:
                        gamma_ij *= term.conj()
                
                spec_hdmr_sum += prefac * gamma_ij

        return spec_hdmr_sum.real
                    
from sympy import Symbol
from sympy.matrices import Matrix
import numpy as np
from qalcore.dwave.qubols.encodings import RealUnitQbitEncoding
from typing import Optional, Union, List, Callable, Dict, Tuple
from dwave.system import DWaveSampler , EmbeddingComposite
import neal
from dimod import ExactSolver
from functools import partial
from dwave.embedding.chain_strength import uniform_torque_compensation
import dwave_networkx as dnx 
from minorminer import find_embedding 
from dwave.embedding import embed_qubo, majority_vote, chain_break_frequency
from .solution_vector import SolutionVector
from .qubols import QUBOLS 

class EmbeddedQUBOLS(QUBOLS):

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
            "verbose": False,
            "chain_strength": None,
            "target_graph": dnx.chimera_graph(16),
            "threshold": None
        }
        self.options = self._validate_solve_options(options)
        self.sampler = self.options.pop('sampler')
        self.target_graph = self.options.pop("target_graph")


    def create_embedded_qubo_dict(self):
        """Embed the qubo dictionary on a target graph 
        """

        # find the embedding
        embedding = find_embedding(self.qubo_dict, self.target_graph)

        # embed the qubo 
        embedded_qubo_dict = embed_qubo(self.qubo_dict, embedding, self.target_graph, 
                                        chain_strength=self.options["chain_strength"])

        # convert to linear indexes
        idx_translate = {}
        count = 0
        for k, v in embedding.items():
            for idx in v:
                idx_translate[idx] = count
                count += 1

        # translate the embedding
        chains = []
        for k, v in embedding.items():
            new_idx = [idx_translate[idx] for idx in v]
            embedding[k] = new_idx
            chains.append(new_idx)


        embedded_qubo_dict_tr = {}
        for k,v in embedded_qubo_dict.items():
            embedded_qubo_dict_tr [ (idx_translate[k[0]], idx_translate[k[1]]) ] = v

        return (embedded_qubo_dict_tr,
                embedding, 
                chains)


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
        self.qubo_dict = self.create_qubo_matrix(self.x, prec=self.options["threshold"])

        self.embedded_qubo_dict, self.embedding, self.chains = self.create_embedded_qubo_dict()

        self.sampleset = self.sampler.sample_qubo(self.embedded_qubo_dict, num_reads = self.options['num_reads'])
        lowest_sol = self.sampleset.lowest()
        self.chain_break = chain_break_frequency(self.sampleset.record.sample, self.embedding)
        lowest_sol, _ = majority_vote(lowest_sol.record[0][0], self.chains)
        
        return sol.decode_solution(lowest_sol[0])



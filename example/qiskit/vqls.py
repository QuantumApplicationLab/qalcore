from qalcore.qiskit.vqls.vqls import VQLS 
import numpy as np 

A = np.random.rand(8,8)
A = A + A.T 

b = np.random.rand(8,1)

vqls = VQLS(A,b)
vqls.solve(verbose=True)
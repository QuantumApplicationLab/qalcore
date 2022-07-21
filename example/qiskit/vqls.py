from qalcore.qiskit.vqls.vqls import VQLS 
import numpy as np 
import matplotlib.pyplot as plt

# symmetric matrix
A = np.random.rand(8,8)
A = A + A.T 

# rhs
b = np.random.rand(8,1)

# classical solution
csol = np.linalg.solve(A,b)
norm = np.linalg.norm(csol)
csol /= norm
csol = np.asarray(csol).flatten().real


# quantum solution
vqls = VQLS(A,b)
qsol = vqls.solve(verbose=True,niter=100)

# plot the result
plt.scatter(csol,qsol.real)
plt.plot([-1,1],[-1,1],'--')
plt.show()
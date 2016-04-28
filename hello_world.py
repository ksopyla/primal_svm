# Welcome to your new Python project
# To start, simply click on "Run file" icon
print("Hello, World!")


import numpy as np
from scipy import sparse
from scipy.sparse.linalg import dsolve

mtx = sparse.spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5)
mtx.todense()

rhs = np.array([1, 2, 3, 4, 5], dtype=np.float32)

mtx1 = mtx.astype(np.float32)
x = dsolve.spsolve(mtx1, rhs, use_umfpack=False)
print(x)  

print("Error: %s" % (mtx1 * x - rhs)) 

mtx2 = mtx.astype(np.float64)
x = dsolve.spsolve(mtx2, rhs, use_umfpack=True)
print(x)  

print("Error: %s" % (mtx2 * x - rhs)) 
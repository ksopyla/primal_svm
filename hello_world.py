import numpy as np
import linearSVM as lsvm



X = np.array([[0.5, 0.3], [1, 0.8], [1, 1.4], [0.6, 0.9]])
Y = np.array([-1, -1, 1, 1])
svm = lsvm.PrimalSVM()

#svm.fit(X,Y)
svm._solve_Newton(X,Y)
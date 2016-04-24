import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg

from sklearn.base import BaseEstimator, ClassifierMixin

#class PrimalSVM(BaseEstimator, ClassifierMixin):
class PrimalSVM():
    '''
    Solves linear SVM in primal, with use of Newton or Conjugate Gradient
    '''
    
    def __init__(self, l2reg=1.0, newton_iter=20 ):
        self.l2reg = l2reg
        self.newton_iter = newton_iter
    
    def fit(self, X, y, method=0):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        self : object
            Returns self.
        """
        
        pass
    
    def _solve_Newton(self):
        """
        Solve the primal SVM problem with Newton method
        """
    def _obj_func(self,w,X,Y,out):
        
        l2reg = self.l2reg
        w0 = w
        w0[-1]=0
        obj = np.sum(out**2)/2+l2reg*w0.dot(w0)/2
        
        return obj;
    
    def predict(self, X):
        pass
    
    
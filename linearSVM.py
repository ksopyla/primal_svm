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
        
        [n,d] = X.shape
        w = np.zeros(d+1)
        
        
        if method==0:
            self._solve_Newton()
        else:
            self._solve_CG()
          
        
        
    
    def _solve_Newton(self):
        """
        Solve the primal SVM problem with Newton method
        """
        pass
        
    def _solve_CG(self):
        pass
   
        
    def _obj_func(self,w,X,Y,out):
        """
        Computes primal value end gradient
        Parameters
        ----------
        w : {array-like} - hyperplane normal vector
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Y : array-like, shape = [n_samples]
            Target vector relative to X
        out: loss function values
        Returns
        -------
        (obj,grad) : tuple, obj - function value, grad - gradient
            
        """
        
        l2reg = self.l2reg
        w0 = w
        w0[-1]=0
        obj = np.sum(out**2)/2+l2reg*w0.dot(w0)/2
        
        grad = l2reg*w0 - np.append( [np.dot(out*Y,X)], [np.sum(out*Y)])
        
        return (obj,grad)
    
    def predict(self, X):
        pass
    
    
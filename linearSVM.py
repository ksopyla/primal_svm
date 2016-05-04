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
        
        self._prec = 1e-6
        
        self.coef_ = None
        self.support_vectors = None
    
    def fit(self, X, Y, method=0):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Y : array-like, shape = [n_samples]
            Target vector relative to X
        method: 0 - Newton method (with full Hessian computation), 1 - Conjugate Gradient method
        Returns
        -------
        self : object
            Returns self.
        """
        
        self._X = X
        self._Y =Y
        
        if method==0:
            self._solve_Newton(X,Y)
        else:
            self._solve_CG()
          
        
        return self
        
    
    def _solve_Newton(self,X,Y):
        """
        Solve the primal SVM problem with Newton method
        """
        [n,d] = X.shape
        
        #we add one last component, which is b (bias)
        self.w = np.zeros(d+1)
        #helper variable for storing 1-Y*(np.dot(X,w))
        self.out = np.ones(n,1)
        
        l= self.l2reg
        #the number of alg. iteration
        iter=0
        
        while True:
            iter=iter+1
            if iter > self.newton_iter:
                print("Maximum {0} of Newton stpes reached, change newton_iter parameter or try larger lambda".format(iter))
                break
            
            obj, grad = self._obj_func(self.w,X,Y,out)
            
            #np.where retunrs a tuple, we take the first dim
            sv = np.where( self.out>0)[0]
           
            hess = self._compute_hessian(sv)
            
            #compute step vector step = -hess\grad
            # %timeit np.linalg.lstsq(hess,grad)
            # %timeit np.linalg.solve(hess,grad) - is faster 4x
            
            step = np.linalg.solve(hess,grad)
            
            t, out = self._line_search(w,step,out)
            
            self.w = self.w+t*step
            
            if step.dot(grad) < self._prec* obj:
                break;

        
        
    def _solve_CG(self):
        pass
   
   
    def _compute_hessian(self, sv):
        """
        Computes the full hessina matrix of svm problem
        hess = lambda*diag([1...1,0])+ [[Xsv'*Xsv sum(Xsv,1)']; [sum(Xsv) length(
        
        Parameters
        sv - array like, list of support vector indices
        ----------
        """
        
        #grab the support vectors
        Xsv = self._X[sv,:]
        
        [n,d] = self._X.shape
      
        #reserve memory for hessian
        hess = np.zeros((d+1,d+1))
        #first compute the second part with dot products between x_i
        hess[0:-1,0:-1]= Xsv.T.dot(Xsv)
        hess[-1,0:-1] = Xsv.sum(axis=0)
        hess[0:-1, -1] = Xsv.sum(axis=0)
        hess[-1,-1] = len(sv)
        
        #then add the first part with lambda
        hess= hess+ self.l2reg* np.diag(np.append(np.ones((d,)), 0))
        
        
        return hess
        
        
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
        
    def _line_search(self, w, d, out):
        """
        Performs line search for optimal w in direction d
        Parameters
        ----------
        w : {array-like}  - hyperplane normal vector
        d : {array-like}  - vector along which we seek optimal sollution
        
        """
        
        Xd = self._X.dot(d[0:-1])+d[-1]
        wd = self.l2reg * w[0:-1].dot(d[0:-1])
        dd = self.l2reg * d[0:-1].dot(d[0:-1])
        
        Y = self._Y
        
        
        t=0
        iter=0
        out2=out
        
        #we do only max 1000 iteration, it should be enough
        while iter<1000:
            out2 = out -t*(Y*Xd)
            sv = np.where( out>0)[0]
            
            #gradient along the line
            g = wd + t*dd - (out2[sv]*Y[sv]).dot(Xd[sv])
            # second derivative along the line
            h = dd + Xd[sv].dot(Xd[sv])
            # 1D Newton step
            t=t-g/h
            
            if g**2/h < 1e-10:
                break
            iter=iter+1
            
        return t, out2
    
    def predict(self, X):
        pass
    
    
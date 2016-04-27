import unittest
import numpy as np
from scipy import sparse as sp
from linearSVM import *



class PrimalSVMTests(unittest.TestCase):
    
    
    def setUp(self):
        self.X = np.array([ [0.5, 0.3], [1 , 0.8], [1,1.4], [0.6, 0.9]])
        self.Y = np.array([-1,-1,1,1])
        self.svm = PrimalSVM()
    
    def test_obj_value_points_correctly_class_close_to_hyperplane(self):
        
        bias = 0.0
        w= np.array([-1,1, bias])
        l = self.svm.l2reg
        X = np.array([ [0.5, 0.3], [1 , 0.8], [1,1.4], [0.6, 0.9]])
        Y = np.array([-1,-1,1,1])
        
        #compute loss for all X -> 1-yi*(xi*w+b)
        out = np.fmax(0, 1-Y*(X.dot(w[0:-1])+w[-1]))
        
        expectedObj=2.0650000000000004
        
        result,_ = self.svm._obj_func(w,X,Y,out)
        
        self.assertAlmostEqual(expectedObj, result)
        
    def test_obj_value_points_correctly_class_far_from_hyperplane(self):
        
        bias = 0.0

        w= np.array([-1,1, bias])
        l = self.svm.l2reg
        X = np.array([ [5, 0.3], [1 , -0.8], [1,6], [-0.6, 3]])
        Y = np.array([-1,-1,1,1])
        
        #compute loss for all X -> 1-yi*(xi*w+b)
        out = np.fmax(0, 1-Y*(X.dot(w[0:-1])+w[-1]))
        
        expectedObj=1.0
        
        result,_  = self.svm._obj_func(w,X,Y,out)
        
        self.assertAlmostEqual(expectedObj, result)
    
    def test_obj_grad_points(self):
        
        bias = 0.0
        w= np.array([-1,1, bias])
        l = self.svm.l2reg
        X = np.array([ [0.5, 0.3], [1 , 0.8], [1,1.4], [0.6, 0.9]])
        Y = np.array([-1,-1,1,1])
        
        #compute loss for all X -> 1-yi*(xi*w+b)
        out = np.fmax(0, 1-Y*(X.dot(w[0:-1])+w[-1]))
        
        expected=np.array([-1.18,  1.59, -0.3 ])
        
        (obj, grad) = self.svm._obj_func(w,X,Y,out)

        np.testing.assert_array_almost_equal(expected,grad) #,'Gradients are not equal')

    
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(PrimalSVMTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
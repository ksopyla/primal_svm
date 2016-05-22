import unittest
import numpy as np
from scipy import sparse as sp
from linearSVM import *


class PrimalSVMTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0.5, 0.3], [1, 0.8], [1, 1.4], [0.6, 0.9]])
        self.Y = np.array([-1, -1, 1, 1])
        self.svm = PrimalSVM()
        self.svm._X = self.X
        self.svm._Y = self.Y

    def test_obj_value_points_correctly_class_close_to_hyperplane(self):
        bias = 0.0
        w = np.array([-1, 1, bias])
        l = self.svm.l2reg
        X = np.array([[0.5, 0.3], [1, 0.8], [1, 1.4], [0.6, 0.9]])
        Y = np.array([-1, -1, 1, 1])

        # compute loss for all X -> 1-yi*(xi*w+b)
        out = np.fmax(0, 1 - Y * (X.dot(w[0:-1]) + w[-1]))

        expectedObj = 2.0650000000000004

        result, _ = self.svm._obj_func(w, X, Y, out)

        self.assertAlmostEqual(expectedObj, result)

    def test_obj_value_points_correctly_class_far_from_hyperplane(self):
        bias = 0.0

        w = np.array([-1, 1, bias])
        l = self.svm.l2reg
        X = np.array([[5, 0.3], [1, -0.8], [1, 6], [-0.6, 3]])
        Y = np.array([-1, -1, 1, 1])

        # compute loss for all X -> 1-yi*(xi*w+b)
        out = np.fmax(0, 1 - Y * (X.dot(w[0:-1]) + w[-1]))

        expectedObj = 1.0

        result, _ = self.svm._obj_func(w, X, Y, out)

        self.assertAlmostEqual(expectedObj, result)

    def test_obj_grad_points(self):
        bias = 0.0
        w = np.array([-1, 1, bias])
        l = self.svm.l2reg
        X = np.array([[0.5, 0.3], [1, 0.8], [1, 1.4], [0.6, 0.9]])
        Y = np.array([-1, -1, 1, 1])

        # compute loss for all X -> 1-yi*(xi*w+b)
        out = np.fmax(0, 1 - Y * (X.dot(w[0:-1]) + w[-1]))

        expected = np.array([-0.82, 0.41, 0.3])

        (obj, grad) = self.svm._obj_func(w, X, Y, out)

        np.testing.assert_array_almost_equal(expected, grad)
        
    def test_compute_hessian(self):
        bias = 0.0
        w = np.array([-1, 1, bias])
        l = self.svm.l2reg
        X = np.array([[0.5, 0.3], [1, 0.8], [1, 1.4], [0.6, 0.9]])
        
        expected = np.array([[3.61, 2.89 , 3.1], [2.89, 4.5 , 3.4], [3.1, 3.4, 4]])

        hess = self.svm._compute_hessian(np.array([0,1,2,3]))

        np.testing.assert_array_almost_equal(expected, hess)
        
        
    def test_compute_linesearch_w_eq_zero(self):
        
        bias = 0.0
        w = np.array([0, 0, bias])
        l = self.svm.l2reg
        
        
        out = np.array([1,1,1,1])
        
        #values computed with use of matlab implementation
        d= np.array([ -0.07716665,  0.75756366, -0.58412495])
        expected_out = np.array([0.60456081, 0.94475931, 0.60070248, 0.94861765])
        expected_t=1


        t, out2 = self.svm._line_search(w,d,out)

        self.assertAlmostEqual(expected_t, t)
        
        np.testing.assert_array_almost_equal(expected_out, out2)
        
    


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(PrimalSVMTests)
    unittest.TextTestRunner(verbosity=2).run(suite)

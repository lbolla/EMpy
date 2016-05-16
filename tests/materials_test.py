# pylint: disable=no-self-use
from unittest import TestCase

import random
import math

from numpy import array
from numpy import pi
from numpy.testing import assert_almost_equal

import EMpy.materials as mat


class MaterialsTest(TestCase):

    def test_RefractiveIndex(self):
        ''':: testing the EMpy.materials.RefractiveIndex class constructors'''
        
        # test const:
        test_rix = 1.50
        a = mat.RefractiveIndex( n0_const = test_rix )
        self.assertEqual(  a.get_rix(1.0)[0]  ,  array([ test_rix ])  )
        
        # test poly:
        test_poly = [1,1]   # n(wl) = 1*wl + 1
        test_rix = 2.0      # n(1) = 1*1 + 1 = 2
        a = mat.RefractiveIndex( n0_poly = test_poly )
        assert_almost_equal(  a.get_rix(1.0)[0]  ,  array([ test_rix ])  )
        
        # test smcoeffs:
        test_poly = [1]*6   
        ''' 6-coeffs:
            n(wls) =  1. +
            B1 * wls ** 2 / (wls ** 2 - C1) +
            B2 * wls ** 2 / (wls ** 2 - C2) +
            B3 * wls ** 2 / (wls ** 2 - C3)
        '''
        test_rix = 1.0536712127723509e-08
        a = mat.RefractiveIndex( n0_smcoeffs = test_poly )
        assert_almost_equal(  a.get_rix(0.5)[0]  ,  array([ test_rix ])  )
        
        # test func:
        test_rix = 1.50
        test_poly = lambda x: 0.0*x + test_rix      # returns a constant
        a = mat.RefractiveIndex( n0_func = test_poly )
        assert_almost_equal(  a.get_rix([1.0,1.5])[0]  ,  array([ test_rix ])  )
        
        # test known:
        test_rix = 1.50
        test_wl = 1.0
        test_dict = {}
        test_dict[test_wl] = test_rix
        self.assertEqual(  a.get_rix(test_wl)[0]  , array([ test_rix ])  )
        
    #end test_RefractiveIndex()
#end class(MaterialsTest)

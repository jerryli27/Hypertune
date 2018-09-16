"""Unit test for hyper_tune"""
import unittest

import hyper_tune
import hyper_tune_test_helper

class TestDataUtilMethods(unittest.TestCase):
    def test_main(self):
        ret = hyper_tune.main('hyper_tune_test_helper', params_and_constraints='[("x",-5,5),("y",-5,5),]',
                              init_points=5, n_iter=25, output_file_name='')
        self.assertAlmostEqual(ret['max_val'], hyper_tune_test_helper.expression(hyper_tune_test_helper.MAX_X_VAL), delta=0.1)
        self.assertAlmostEqual(ret['max_params']['x'], hyper_tune_test_helper.MAX_X_VAL, delta=0.1)


if __name__ == '__main__':
    unittest.main()
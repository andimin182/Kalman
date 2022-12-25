import unittest
from kf import KF
import numpy as np

class TestKF(unittest.TestCase):
    def test_construct_with_x_and_v(self):
        x = 0.2
        v = 5
        aVariance = 1.2

        kf = KF(initialX0=x, initialV0=v, accel_var=aVariance)

        # Tests
        self.assertAlmostEqual(kf.position,x)
        self.assertAlmostEqual(kf.velocity,v)

    def test_call_predict_method(self):
        x = 0.2
        v = 5
        aVariance = 1.2

        kf = KF(initialX0=x, initialV0=v, accel_var=aVariance)
        kf.predict(dt=0.1)
  
    def test_after_calling_predict_method_x_and_P_right_shapes(self):
        x = 0.2
        v = 5
        aVariance = 1.2

        kf = KF(initialX0=x, initialV0=v, accel_var=aVariance)
        kf.predict(dt=0.1)

        # Tests
        self.assertEqual(kf.mean.shape, (2, ))
        self.assertAlmostEqual(kf.covariance.shape,(2,2))

    def test_after_calling_predict_uncertainty_increases(self):
        x = 0.2
        v = 5
        aVariance = 1.2

        kf = KF(initialX0=x, initialV0=v, accel_var=aVariance)
        covBefore = np.linalg.det(kf.covariance)
        kf.predict(dt=0.1)
        covAfter = np.linalg.det(kf.covariance)

        # Tests
        self.assertGreater(covAfter, covBefore)
    def test_after_calling_update_uncertainty_decreases(self):
        x = 0.2
        v = 5
        aVariance = 1.2

        kf = KF(initialX0=x, initialV0=v, accel_var=aVariance)
        covBefore = np.linalg.det(kf.covariance)
        kf.update(stateMeas=0.1, measVar=0.1)
        covAfter = np.linalg.det(kf.covariance)

        # Tests
        self.assertLess(covAfter, covBefore)

       

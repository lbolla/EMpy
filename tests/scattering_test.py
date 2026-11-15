import unittest

import numpy

import EMpy.scattering as S


class ScatteringTest(unittest.TestCase):
    def test_stack_shapes(self):
        x = numpy.arange(4)
        y = numpy.arange(3)
        z = numpy.arange(2)
        expected = S.stack(x, y, z)
        self.assertEqual(expected.shape, (3, 24))
        numpy.testing.assert_array_equal(
            expected, S.stack(x.reshape(1, -1), y.reshape(-1, 1), list(z))
        )

    def test_kottler_validation_and_dS_shapes(self):
        P = numpy.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        Q = numpy.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        J = numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        M = numpy.zeros_like(J)
        dS_1d = numpy.array([0.1, 0.2])
        dS_2d = dS_1d.reshape(1, -1)

        f1 = S.currentsScatteringKottler(P, J, M, Q, dS_1d, f=1e9)
        f2 = S.currentsScatteringKottler(P, J, M, Q, dS_2d, f=1e9)
        numpy.testing.assert_allclose(f1.E, f2.E)
        numpy.testing.assert_allclose(f1.H, f2.H)
        self.assertTrue(numpy.all(numpy.isfinite(f1.E)))
        self.assertTrue(numpy.all(numpy.isfinite(f1.H)))

        with self.assertRaises(ValueError):
            S.currentsScatteringKottler(numpy.zeros((2, 2)), J, M, Q, dS_1d, f=1e9)
        with self.assertRaises(ValueError):
            S.currentsScatteringKottler(P, J, M, Q, numpy.array([0.1]), f=1e9)

    def test_kottler_self_term_r_zero(self):
        Q = numpy.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        P = Q.copy()
        J = numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        M = numpy.zeros_like(J)
        dS = numpy.array([0.1, 0.1])

        field = S.currentsScatteringKottler(P, J, M, Q, dS, f=1e9)
        self.assertTrue(numpy.all(numpy.isfinite(field.E)))
        self.assertTrue(numpy.all(numpy.isfinite(field.H)))

        field_single = S.currentsScatteringKottler(
            Q[:, :1], J[:, :1], M[:, :1], Q[:, :1], dS[:1], f=1e9
        )
        numpy.testing.assert_array_equal(field_single.E, numpy.zeros((3, 1)))
        numpy.testing.assert_array_equal(field_single.H, numpy.zeros((3, 1)))


if __name__ == "__main__":
    unittest.main()

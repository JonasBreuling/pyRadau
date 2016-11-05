#!/usr/bin/env python

import numpy as np
import unittest

from pyradau13 import radau13

class TestIntegration(unittest.TestCase):
    def _pendulum_rhs(self, t, x):
        return [ x[1], -x[0] ]

    def test_exception_from_callback(self):
        class _TestException(Exception):
            pass
        def _rhs(t, x):
            raise _TestException()
        def _dense_cb(told, t, x, cont):
            raise _TestException()

        self.assertRaises(_TestException, lambda: radau13(_rhs, 0, 1))
        self.assertRaises(_TestException,
                          lambda: radau13(lambda t, x: 1, 0, 1, dense_callback=_dense_cb))

    def test_intentional_abort(self):
        """radau13 should not raise if dense callback returns True, but it should
        if an exception occurs there."""
        def _rhs(t, x):
            return 1
        def _dense_cb(told, t, x, cont):
            return True
        def _dense_cb_err(told, t, x, cont):
            return 1/0
        self.assertEqual(radau13(_rhs, 0, 1, dense_callback=_dense_cb), 0)
        self.assertRaises(ZeroDivisionError, lambda: radau13(_rhs, 0, 1, dense_callback=_dense_cb_err))

    def test_exp(self):
        self.assertAlmostEqual(float(radau13(lambda t, x: x, 1, 1)), np.exp(1))

    def test_quad(self):
        self.assertAlmostEqual(float(radau13(lambda t, x: t, 0, 1)), .5)

    def test_pendulum(self):
        for i in np.linspace(1, 10, 50):
            self.assertAlmostEqual(float(radau13(self._pendulum_rhs, [ 1, 0 ], i)[0]), np.cos(i))

    def test_dense_feedback(self):
        _x = []
        _t = []
        def _dense_cb(told, t, x, cont):
            _x.append(x[0])
            _t.append(t)
        radau13(self._pendulum_rhs, [1, 0], 10, dense_callback=_dense_cb)

        self.assertTrue(len(_x) > 1)
        for t, x in zip(_t, _x):
            self.assertAlmostEqual(x, np.cos(t))

    def test_multiple_time_levels(self):
        X = np.linspace(0.1, 1, 100)
        y = radau13(lambda t, x: x, 1, X)
        self.assertAlmostEqual(max(abs(np.exp(X) - y)), 0)

    def test_continous_output(self):
        def _dense_cb(told, t, x, cont):
            tavg = (told + t) / 2
            self.assertAlmostEqual(cont(tavg), np.exp(tavg))
        radau13(lambda t, x: x, 1, 1, dense_callback=_dense_cb)

    def test_memory_usage(self):
        try:
            import resource
            import gc
        except ImportError:
            return

        def _test():
            state = np.ones(100)
            radau13(lambda t, x: x, state, 10)
        _test()
        usage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for j in range(100):
            gc.collect()
            _test()
        usage_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.assertGreater(usage_before, usage_after - 100)

    def test_extra_args(self):
        optional_args = [ x for x in radau13.__doc__.split("\n\n") if
                         x.startswith("Further optional")][0].split(":")[1].split()
        for arg in optional_args:
            try:
                radau13(lambda t, x: x, 1, 1, **{arg: 0})
            except Exception as e:
                raise RuntimeError("Passing %s failed: %s" % (arg, e))
        radau13(lambda t, x: x, 1, 1, **{x: 0 for x in optional_args})

    def test_jacobian(self):
        def _jac_test_rhs(t, x):
            return x
        def _jac_test_jac(t, x):
            return 1

        self.assertAlmostEqual(np.linalg.norm(radau13(_jac_test_rhs, [1, 0], 10) -
            radau13(_jac_test_rhs, [1, 0], 10, jacobian_fn=_jac_test_jac)), 0)

    def test_mass_matrix(self):
        """ Test problem:

          d x_1 / dt = x_2
                   0 = x_1 - x_2 / 2
        """
        mass_matrix = np.array([ [ 1, 0 ], [ 0, 0 ] ])
        rhs = lambda t, x: [ x[1], x[0] - x[1] / 2 ]
        retval = radau13(rhs, [ 1, 2 ], 10, mass_matrix=mass_matrix)[0]
        self.assertAlmostEqual(retval, np.exp(2 * 10), 2)

    def test_failure(self):
        "This problem *is* unsolvable"
        with self.assertRaises(RuntimeError):
            radau13(lambda t, x: x**2, 1, 1)


if __name__ == '__main__':
    unittest.main()

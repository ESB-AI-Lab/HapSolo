"""Tests for hapsolo.cost_formula — safe AST-based cost formula compiler."""

import math
import unittest

import numpy as np

from hapsolo.cost_formula import (
    compile_cost_formula,
    validate_formula,
    DEFAULT_FORMULA,
    ALLOWED_VARIABLES,
)
from hapsolo.scoring import cost_function


class TestValidation(unittest.TestCase):

    def test_default_formula_valid(self):
        validate_formula(DEFAULT_FORMULA)

    def test_simple_expressions(self):
        for expr in ['D + M', '(D + M) / S', 'D ** 2 / S', 'M', 'S + D + F + M']:
            validate_formula(expr)

    def test_with_functions(self):
        for expr in ['sqrt(D)', 'log(S + 1)', 'exp(-M)', 'abs(D - M)', 'log2(S + 1)']:
            validate_formula(expr)

    def test_with_theta_vars(self):
        validate_formula('theta_s * S + theta_d * D')

    def test_with_C_and_n(self):
        validate_formula('(n - C) / S')

    def test_ternary(self):
        validate_formula('D / S if S > 0 else 99999')

    def test_reject_empty(self):
        with self.assertRaises(ValueError):
            validate_formula('')
        with self.assertRaises(ValueError):
            validate_formula('   ')

    def test_reject_import(self):
        with self.assertRaises((ValueError, SyntaxError)):
            validate_formula("__import__('os')")

    def test_reject_attribute_access(self):
        with self.assertRaises(ValueError):
            validate_formula('S.__class__')

    def test_reject_unknown_variable(self):
        with self.assertRaises(ValueError):
            validate_formula('X + Y')

    def test_reject_unknown_function(self):
        with self.assertRaises(ValueError):
            validate_formula('eval(S)')
        with self.assertRaises(ValueError):
            validate_formula('print(S)')
        with self.assertRaises(ValueError):
            validate_formula('open("file")')

    def test_reject_lambda(self):
        with self.assertRaises((ValueError, SyntaxError)):
            validate_formula('lambda: S')

    def test_reject_comprehension(self):
        with self.assertRaises((ValueError, SyntaxError)):
            validate_formula('[x for x in range(10)]')

    def test_reject_string_constant(self):
        with self.assertRaises(ValueError):
            validate_formula('"hello" + S')

    def test_case_sensitive(self):
        with self.assertRaises(ValueError):
            validate_formula('s + d')


class TestCompileAndEvaluate(unittest.TestCase):

    def test_default_formula_matches_cost_function(self):
        fn = compile_cost_formula(DEFAULT_FORMULA)
        S, D, F, M = 100, 20, 5, 10
        n = S + D + F + M
        expected = cost_function(M, S, D, F, n, 1.0, 1.0, 0.0, 1.0)
        result = fn(S=S, D=D, F=F, M=M, C=S+D, n=n,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertAlmostEqual(result, expected, places=10)

    def test_custom_quadratic(self):
        fn = compile_cost_formula('(D**2 + M) / S')
        result = fn(S=100, D=20, F=5, M=10, C=120, n=135,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertAlmostEqual(result, (400 + 10) / 100)

    def test_custom_with_log(self):
        fn = compile_cost_formula('log(D + 1) / S')
        result = fn(S=100, D=20, F=0, M=0, C=120, n=120,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertAlmostEqual(result, math.log(21) / 100)

    def test_C_equals_S_plus_D(self):
        fn = compile_cost_formula('C')
        result = fn(S=60, D=40, F=5, M=10, C=100, n=115,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertEqual(result, 100)

    def test_n_variable(self):
        fn = compile_cost_formula('n')
        result = fn(S=60, D=40, F=5, M=10, C=100, n=115,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertEqual(result, 115)

    def test_theta_weights_accessible(self):
        fn = compile_cost_formula('theta_d * D / (theta_s * S)')
        result = fn(S=100, D=20, F=0, M=0, C=120, n=120,
                    theta_s=2.0, theta_d=3.0, theta_f=0.0, theta_m=1.0)
        self.assertAlmostEqual(result, (3.0 * 20) / (2.0 * 100))

    def test_literal_constants(self):
        fn = compile_cost_formula('0.5 * D + 2 * M')
        result = fn(S=100, D=20, F=0, M=10, C=120, n=130,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertAlmostEqual(result, 0.5 * 20 + 2 * 10)

    def test_deeply_nested_parens(self):
        fn = compile_cost_formula('((((S))))')
        result = fn(S=42, D=0, F=0, M=0, C=42, n=42,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertEqual(result, 42)

    def test_formula_ignoring_some_vars(self):
        fn = compile_cost_formula('M / S')
        result = fn(S=100, D=999, F=999, M=10, C=1099, n=2098,
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        self.assertAlmostEqual(result, 0.1)


class TestCostFunctionIntegration(unittest.TestCase):

    def test_formula_fn_overrides_default(self):
        fn = compile_cost_formula('(D + M) / S')
        result = cost_function(10, 100, 20, 5, 135,
                               theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0,
                               formula_fn=fn)
        self.assertAlmostEqual(result, (20 + 10) / 100)

    def test_formula_fn_none_uses_default(self):
        result_default = cost_function(10, 100, 20, 5, 135,
                                        theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        result_none = cost_function(10, 100, 20, 5, 135,
                                     theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0,
                                     formula_fn=None)
        self.assertEqual(result_default, result_none)


class TestNumpyArrayCompat(unittest.TestCase):

    def test_element_wise_on_numpy(self):
        fn = compile_cost_formula('(D + M) / S')
        S = np.array([100, 200, 50], dtype=np.float64)
        D = np.array([20, 10, 30], dtype=np.float64)
        F = np.array([5, 2, 8], dtype=np.float64)
        M = np.array([10, 5, 15], dtype=np.float64)
        result = fn(S=S, D=D, F=F, M=M, C=S+D, n=np.float64(300),
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        expected = (D + M) / S
        np.testing.assert_allclose(result, expected)

    def test_sqrt_on_numpy(self):
        fn = compile_cost_formula('sqrt(D)')
        D = np.array([4.0, 9.0, 16.0])
        result = fn(S=np.ones(3), D=D, F=np.zeros(3), M=np.zeros(3),
                    C=np.ones(3)+D, n=np.float64(100),
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))

    def test_default_formula_on_numpy(self):
        fn = compile_cost_formula(DEFAULT_FORMULA)
        S = np.array([100.0, 200.0])
        D = np.array([20.0, 10.0])
        F = np.array([5.0, 2.0])
        M = np.array([10.0, 5.0])
        result = fn(S=S, D=D, F=F, M=M, C=S+D, n=np.float64(300),
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        expected = (0.0 * F + 1.0 * D + 1.0 * M) / (1.0 * S)
        np.testing.assert_allclose(result, expected)


class TestGPUCompat(unittest.TestCase):

    def setUp(self):
        try:
            import cupy as cp
            self.cp = cp
        except ImportError:
            self.skipTest('cupy not available')

    def test_element_wise_on_cupy(self):
        cp = self.cp
        fn = compile_cost_formula('(D + M) / S')
        S = cp.array([100, 200, 50], dtype=cp.float64)
        D = cp.array([20, 10, 30], dtype=cp.float64)
        F = cp.array([5, 2, 8], dtype=cp.float64)
        M = cp.array([10, 5, 15], dtype=cp.float64)
        result = fn(S=S, D=D, F=F, M=M, C=S+D, n=cp.float64(300),
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        expected = (D + M) / S
        cp.testing.assert_allclose(result, expected)

    def test_sqrt_on_cupy(self):
        cp = self.cp
        fn = compile_cost_formula('sqrt(D)')
        D = cp.array([4.0, 9.0, 16.0])
        result = fn(S=cp.ones(3), D=D, F=cp.zeros(3), M=cp.zeros(3),
                    C=cp.ones(3)+D, n=cp.float64(100),
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        cp.testing.assert_allclose(result, cp.array([2.0, 3.0, 4.0]))

    def test_default_formula_on_cupy(self):
        cp = self.cp
        fn = compile_cost_formula(DEFAULT_FORMULA)
        S = cp.array([100.0, 200.0])
        D = cp.array([20.0, 10.0])
        F = cp.array([5.0, 2.0])
        M = cp.array([10.0, 5.0])
        result = fn(S=S, D=D, F=F, M=M, C=S+D, n=cp.float64(300),
                    theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0)
        expected = (0.0 * F + 1.0 * D + 1.0 * M) / (1.0 * S)
        cp.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()

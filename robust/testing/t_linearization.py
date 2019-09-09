from __future__ import division
from builtins import zip
from builtins import range
import numpy as np
from gpkit import Variable, Model
import scipy.optimize as op

import unittest
from gpkit.tests.helpers import run_tests

from robust.linearize_twoterm_posynomials import LinearizeTwoTermPosynomials

np.warnings.filterwarnings('ignore')

def convex_function(x):
    return np.log(1 + np.exp(x))

class TestLinearization(unittest.TestCase):
    def test_tangent_point_func(self):
        for _ in range(100):
            eps = np.random.rand() * 0.2
            x_old = - np.random.rand() * np.log(np.exp(0.2) - 1)
            y_old = convex_function(x_old) - eps
            x_tangent = op.newton(LinearizeTwoTermPosynomials.tangent_point_func, x_old + 1, args=(x_old, eps))
            y_tangent = convex_function(x_tangent)

            self.assertGreater(x_tangent, x_old)

            def tangent_line(x):
                return (y_old - y_tangent) * (x - x_tangent) / (x_old - x_tangent) + y_tangent

            self.assertLess(convex_function(x_tangent) - tangent_line(x_tangent), 0.0001)

            trial_points = list(np.arange(0, 20, 0.01))
            function_points = [convex_function(i) for i in trial_points]
            tangent_points = [tangent_line(i) for i in trial_points]
            difference = [a - b for a, b in zip(function_points, tangent_points)]

            self.assertTrue(all(i >= 0 for i in difference))

        return


    def test_intersection_point_function(self):
        for _ in range(100):
            eps = np.random.rand() * 0.2
            x_old = - np.random.rand() * np.log(np.exp(0.2) - 1)
            y_old = convex_function(x_old) - eps

            x_tangent = op.newton(LinearizeTwoTermPosynomials.tangent_point_func, x_old + 1, args=(x_old, eps))
            y_tangent = convex_function(x_tangent)

            tangent_slope = (y_old - y_tangent) / (x_old - x_tangent)
            tangent_intercept = - (y_old - y_tangent) * x_tangent / (x_old - x_tangent) + y_tangent

            x_intersection = op.newton(LinearizeTwoTermPosynomials.intersection_point_func,
                                       x_tangent + 1, args=(tangent_slope, tangent_intercept, eps))
            self.assertGreater(x_intersection, x_tangent)

            diff = convex_function(x_intersection) - eps - tangent_slope * x_intersection - tangent_intercept

            self.assertLess(diff, 0.0001)
        return


    def test_iterate_two_term_posynomial_linearization_coeff(self):
        # Passes...
        for _ in range(10):

            eps = np.random.rand() * np.log(2)
            r = int(np.ceil(np.random.rand()*18)) + 1

            number_of_actual_r, a, b, x_tangent, x_intersection = LinearizeTwoTermPosynomials.iterate_linearization_coeff(r, eps)

            def piece_wise_linear_function(x):
                evaluations = [a_i*x + b_i for a_i, b_i in zip(a, b)]
                evaluations.insert(0, 0)
                evaluations.append(x)
                return max(evaluations)

            # raise Exception("ejer")
            tangent_difference = [convex_function(k) - piece_wise_linear_function(k) for k in x_tangent]
            intersection_difference = [convex_function(k) - piece_wise_linear_function(k) for k in x_intersection]

            self.assertTrue(all(i <= 0.00001 for i in tangent_difference))
            self.assertTrue(all(i - eps <= 0.00001 for i in intersection_difference))

    def test_two_term_posynomial_linearization_coeff(self):
        for r in range(3, 21):

            slopes, intercepts, x_tangent, x_intersection, eps = LinearizeTwoTermPosynomials.\
                linearization_coeff(r)

            self.assertTrue(all(np.abs(slopes[i] + slopes[r - i - 3] - 1) <= 0.0001 for i in range(0, r-2)))
            self.assertTrue(all(np.abs(intercepts[i] - intercepts[r - i - 3]) <= 0.001 for i in range(0, r-2)))

            def piece_wise_linear_function(x):
                evaluations = [a_i*x + b_i for a_i, b_i in zip(slopes, intercepts)]
                evaluations.insert(0, 0)
                evaluations.append(x)
                return max(evaluations)

            tangent_difference = [convex_function(k) - piece_wise_linear_function(k) for k in x_tangent]
            intersection_difference = [convex_function(k) - piece_wise_linear_function(k) for k in x_intersection]

            self.assertTrue(all(i <= 0.001 for i in tangent_difference))
            self.assertTrue(all(np.abs(i - eps) <= 0.001 for i in intersection_difference))

    def test_linearize_two_term_posynomial(self):
        for _ in range(20):
            tol = np.random.rand()*0.001
            number_of_gp_variables = int(np.random.rand()*20) + 1

            m_1 = np.random.rand()*10
            m_2 = np.random.rand()*10
            subs = {}
            for i in range(number_of_gp_variables):
                x = Variable('x_%s' % i)
                m_1 *= x**(np.random.rand()*10 - 5)
                m_2 *= x**(np.random.rand()*10 - 5)
                subs[x.key.name] = np.random.rand()*100

            p = m_1 + m_2

            r = int(np.random.rand()*18) + 2
            linearized_p = LinearizeTwoTermPosynomials(p)
            no_data_upper, no_data_lower, data = linearized_p.linearize(1, r)

            self.assertEqual(len(data), r)
            self.assertEqual(len(no_data_upper), 1)
            self.assertEqual(len(no_data_lower), 1)

            subs_p = p.sub(subs)
            m = Model(Variable("w_1"), data, subs)
            sol = m.solve(verbosity=0)
            w_1 = sol['cost']

            upper_pos = no_data_upper[0].as_posyslt1()
            lower_pos = no_data_lower[0].as_posyslt1()

            upper_over_lower = upper_pos[0]/lower_pos[0]
            eps = np.log(upper_over_lower.cs[0])

            self.assertTrue(np.log(subs_p.cs[0]) - np.log(lower_pos[0].sub({"w_1": w_1}).cs[0]) <= eps + tol)
            self.assertTrue(np.log(upper_pos[0].sub({"w_1": w_1}).cs[0]) - np.log(subs_p.cs[0]) <= eps + tol)

TESTS = [TestLinearization]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

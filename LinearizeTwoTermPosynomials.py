import numpy as np
import scipy.optimize as op
from gpkit import Variable, Monomial, Posynomial


class LinearizeTwoTermPosynomials:
    """
    Linearizes two term posynomials
    """

    p = Posynomial()

    def __init__(self, p):
        self.p = p

    @staticmethod
    def tangent_point_func(k, x, eps):
        """
        the function used to calculate the tangent point
        :param k: the variable
        :param x: the old x
        :param eps: the error
        :return: the function
        """
        return np.log(1 + np.exp(x)) - eps - np.log(1 + np.exp(k)) - np.exp(k) * (x - k) / (1 + np.exp(k))

    @staticmethod
    def intersection_point_func(x, a, b, eps):
        """
        the function used to calculate the intersection point
        :param x: the variable
        :param a: the slope
        :param b: the intercept
        :param eps: the error
        :return: the slopes, intercepts, and intersection points
        """
        return a * x + b - np.log(1 + np.exp(x)) + eps

    @staticmethod
    def iterate_two_term_exp_linearization_coeff(r, eps):
        """
        Finds the appropriate r, slope, and intercept for a given eps
        :param r: the number of PWL functions
        :param eps: error
        :return: the slope, intercept, and new x
        """
        a, b = [], []

        x_first = np.log(np.exp(eps) - 1)
        x_new = x_first
        for i in xrange(r - 2):
            x_old = x_new
            x_tangent = op.newton(LinearizeTwoTermPosynomials.tangent_point_func, x_old + 1, args=(x_old, eps))
            a.append(np.exp(x_tangent) / (1 + np.exp(x_tangent)))
            b.append(-np.exp(x_tangent) * x_tangent / (1 + np.exp(x_tangent)) + np.log(1 + np.exp(x_tangent)))
            x_new = op.newton(LinearizeTwoTermPosynomials.intersection_point_func,
                              x_tangent + 1, args=(a[i], b[i], eps))
        return a, b, x_new

    @staticmethod
    def two_term_exp_linearization_coeff(r, tol):
        """
        Finds the appropriate r, slopes, and intercepts for a given tolerance
        :param r: the number of PWL functions
        :param tol: tolerance
        :return: the slope, intercept, and new x
        """

        a = []
        b = []
        eps = 0

        eps_min = 0
        eps_max = np.log(2)
        delta = 100
        # print('two_term_exp_linearization: before looping')
        while delta > tol:
            eps = (eps_max + eps_min) / 2
            x_final_theoretical = -np.log(np.exp(eps) - 1)
            # print('two_term_exp_linearization: before iterating')
            try:
                (a, b, x_final_actual) = \
                    LinearizeTwoTermPosynomials.iterate_two_term_exp_linearization_coeff(r, eps)
            except:
                x_final_actual = x_final_theoretical + 2 * tol
            # print('two_term_exp_linearization: after iterating')
            if x_final_actual < x_final_theoretical:
                eps_min = eps
            else:
                eps_max = eps
            delta = np.abs(x_final_actual - x_final_theoretical)
            # print('two_term_exp_linearization: eps ', eps)
        # print('two_term_exp_linearization: end looping')
        return a, b, eps

    def linearize_two_term_exp(self, m, r, tol):
        # print('linearize_two_term_exp: begin with r = ', r)
        if len(self.p.exps) != 2:
            raise Exception('The Posynomial is not a two term posynomial')
        # print('linearize_two_term_exp: before finding coeff')
        (a, b, eps) = LinearizeTwoTermPosynomials.two_term_exp_linearization_coeff(r, tol)
        # print('linearize_two_term_exp: after finding coeff')
        data_constraints = []
        w = Variable('w_%s' % m)
        no_data_constraints_upper = [w * np.exp(eps) <= 1]
        no_data_constraints_lower = [w <= 1]
        first_monomial = Monomial(self.p.exps[0], self.p.cs[0])
        second_monomial = Monomial(self.p.exps[1], self.p.cs[1])
        data_constraints += [first_monomial <= w]
        for i in xrange(r - 2):
            data_constraints += [first_monomial ** a[r - 3 - i] *
                                 second_monomial ** a[i] * np.exp(b[i]) <= w]
        data_constraints += [second_monomial <= w]
        # print('linearize_two_term_exp: end!!')
        return no_data_constraints_upper, no_data_constraints_lower, data_constraints

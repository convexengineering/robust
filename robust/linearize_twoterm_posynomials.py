from __future__ import division
from builtins import range
from builtins import object
import numpy as np
import scipy.optimize as op
import os
from gpkit import Variable, Monomial, Posynomial


class LinearizeTwoTermPosynomials(object):
    """
    Linearizes two term posynomials
    """

    def __init__(self, p):
        self.p = p

    @staticmethod
    def tangent_point_func(k, x, eps):
        """
        the function used to calculate the tangent points
        :param k: the point of tangency
        :param x: the old intersection point
        :param eps: the error
        :return: the equation of the tangent line
        """
        # warnings.simplefilter("ignore"):  # this is making things slower
        return np.log(1 + np.exp(x)) - eps - np.log(1 + np.exp(k)) - np.exp(k) * (x - k) / (1 + np.exp(k))

    @staticmethod
    def intersection_point_func(x, a, b, eps):
        """
        the function used to calculate the intersection points
        :param x: the break point to be solved for
        :param a: the slope of the tangent line
        :param b: the intercept of the tangent line
        :param eps: the linearization error
        :return: the break point equation
        """
        return a * x + b - np.log(1 + np.exp(x)) + eps

    @staticmethod
    def iterate_two_term_posynomial_linearization_coeff(r, eps):
        """
        Finds the appropriate slopes, intercepts, tangency points, and intersection points for a given linearization
        error and number of piecewise linear sections
        :param r: the number of piecewise linear sections
        :param eps: linearization error
        :return: the slopes, intercepts, tangency points, and intersection points
        """
        if r < 2:
            raise Exception('The number of piece-wise sections should two or larger')

        a, b = [], []

        x_intersection = []
        x_tangent = []
        x_intersection.append(np.log(np.exp(eps) - 1))

        i = 1
        while i < r - 1:
            x_old = x_intersection[i - 1]
            try:
                tangent_point = op.newton(LinearizeTwoTermPosynomials.tangent_point_func, x_old + 1, args=(x_old, eps))
                slope = np.exp(tangent_point) / (1 + np.exp(tangent_point))
                intercept = -np.exp(tangent_point) * tangent_point / (1 + np.exp(tangent_point)) + np.log(
                    1 + np.exp(tangent_point))
                intersection_point = op.newton(LinearizeTwoTermPosynomials.intersection_point_func,
                                               tangent_point + 1, args=(slope, intercept, eps))
            except RuntimeError:
                return i, a, b, x_tangent, x_intersection

            x_tangent.append(tangent_point)
            a.append(slope)
            b.append(intercept)
            x_intersection.append(intersection_point)

            i += 1
        return r, a, b, x_tangent, x_intersection

    @staticmethod
    def compute_two_term_posynomial_linearization_coeff(r, tol):
        """
        Calculates the slopes, intercepts, tangency points, intersection points, and linearization error for a given
        number of piecewise-linear sections

        :param r: the number of piecewise-linear sections
        :param tol: tolerance of the linearization parameters
        :return: slopes, intercepts, tangency points, intersection points, and linearization error
        """
        a = None
        b = None
        x_tangent = None
        x_intersection = None

        eps = None
        eps_min = 0
        eps_max = np.log(2)
        delta = 100
        delta_old = 200

        while delta > tol and delta != delta_old:
            delta_old = delta
            eps = (eps_max + eps_min) / 2
            x_final_theoretical = -np.log(np.exp(eps) - 1)
            number_of_actual_r, a, b, x_tangent, x_intersection = \
                LinearizeTwoTermPosynomials.iterate_two_term_posynomial_linearization_coeff(r, eps)

            x_final_actual = x_intersection[-1]

            if x_final_actual > x_final_theoretical or number_of_actual_r < r:
                eps_max = eps
            else:
                eps_min = eps
            delta = np.abs(x_final_actual - x_final_theoretical)
        return a, b, x_tangent, x_intersection, eps

    @staticmethod
    def two_term_posynomial_linearization_coeff(r):
        """
        Reads the slopes, intercepts, tangency points, intersection points, and linearization error for a given number
        of piecewise-linear sections from a text file

        :param r: the number of piecewise-linear sections
        :return: slopes, intercepts, tangency points, intersection points, and linearization error
        """
        if r < 2:
            raise Exception('The number of piece-wise sections should two or larger')

        if r < 100:
            linearization_data_file = open(os.path.dirname(__file__) + "/data/linearization_data.txt", "r")
            for _ in range(r-2):
                linearization_data_file.readline()
            line = linearization_data_file.readline()
            data = line.split(": ")
            slopes = data[0].split(", ")[0:-1]
            slopes = [float(item) for item in slopes]
            intercepts = data[1].split(", ")[0:-1]
            intercepts = [float(item) for item in intercepts]
            x_tangent = data[2].split(", ")[0:-1]
            x_tangent = [float(item) for item in x_tangent]
            x_intersection = data[3].split(", ")[0:-1]
            x_intersection = [float(item) for item in x_intersection]
            eps = float(data[4])
            linearization_data_file.close()

            return slopes, intercepts, x_tangent, x_intersection, eps
        else:
            return LinearizeTwoTermPosynomials.compute_two_term_posynomial_linearization_coeff(r, 2*np.finfo(float).eps)

    def linearize_two_term_posynomial(self, m, r):
        """
        Approximates a two term posynomial constraint by upper and lower piecewise-linear constraints

        :param m: the index of the constraint
        :param r: the number of piecewise-linear sections
        :return: the deprived of data upper and lower constraints and the common data containing constraints
        """
        if r < 2:
            raise Exception('The number of piece-wise sections should be two or larger')

        if len(self.p.exps) > 2:
            raise Exception('The posynomial is larger than a two term posynomial')

        if len(self.p.exps) < 2:
            return [], [], [self.p <= 1]

        a, b, _, _, eps = LinearizeTwoTermPosynomials.two_term_posynomial_linearization_coeff(r)

        data_constraints = []

        w = Variable('w_%s' % m)
        no_data_constraints_upper = [w * np.exp(eps) <= 1]
        no_data_constraints_lower = [w <= 1]

        first_monomial, second_monomial = self.p.chop()
        data_constraints += [first_monomial <= w]

        for i in range(r - 2):
            data_constraints += [first_monomial ** a[r - 3 - i] *
                                 second_monomial ** a[i] * np.exp(b[i]) <= w]

        data_constraints += [second_monomial <= w]
        return no_data_constraints_upper, no_data_constraints_lower, data_constraints

if __name__ == '__main__':
    pass

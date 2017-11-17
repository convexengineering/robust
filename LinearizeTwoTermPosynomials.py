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
        the function used to calculate the tangent points
        :param k: the variable
        :param x: the old x
        :param eps: the error
        :return: the function
        """
        # warnings.simplefilter("ignore"): this is making things slower
        return np.log(1 + np.exp(x)) - eps - np.log(1 + np.exp(k)) - np.exp(k) * (x - k) / (1 + np.exp(k))

    @staticmethod
    def intersection_point_func(x, a, b, eps):
        """
        the function used to calculate the intersection points
        :param x: the variable
        :param a: the slope
        :param b: the intercept
        :param eps: the error
        :return: the slopes, intercepts, and intersection points
        """
        return a * x + b - np.log(1 + np.exp(x)) + eps

    @staticmethod
    def iterate_two_term_posynomial_linearization_coeff(r, eps):
        """
        Finds the appropriate r, slope, and intercept for a given eps
        :param r: the number of PWL functions
        :param eps: error
        :return: the slope, intercept, and new x
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
            except:
                return i, a, b, x_tangent, x_intersection

            x_tangent.append(tangent_point)
            a.append(slope)
            b.append(intercept)
            x_intersection.append(intersection_point)

            i += 1
        return r, a, b, x_tangent, x_intersection

    @staticmethod
    def compute_two_term_posynomial_linearization_coeff(r, tol):
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
        Finds the appropriate r, slopes, and intercepts for a given tolerance
        :param r: the number of PWL functions
        :return: the slope, intercept, and new x
        """
        if r < 2:
            raise Exception('The number of piece-wise sections should two or larger')

        if r < 100:
            linearization_data_file = open("data/linearization_data.txt", "r")
            for _ in xrange(r-2):
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
        Approximates a two term posynomial constraint by upper and lower piece-wise linear constraints
        :param m: the index of the constraint
        :param r: the number of linear functions used for approximation
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

        first_monomial = Monomial(self.p.exps[0], self.p.cs[0])
        second_monomial = Monomial(self.p.exps[1], self.p.cs[1])
        data_constraints += [first_monomial <= w]

        for i in xrange(r - 2):
            data_constraints += [first_monomial ** a[r - 3 - i] *
                                 second_monomial ** a[i] * np.exp(b[i]) <= w]

        data_constraints += [second_monomial <= w]

        return no_data_constraints_upper, no_data_constraints_lower, data_constraints

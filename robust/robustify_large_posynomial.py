from __future__ import absolute_import
from __future__ import division
from builtins import map
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
from gpkit import Variable, Monomial, SignomialsEnabled
from gpkit.nomials import NomialMap

from .robust_gp_tools import RobustGPTools


class RobustifyLargePosynomial(object):

    def __init__(self, p, type_of_uncertainty_set, setting):
        self.p = p
        self.type_of_uncertainty_set = type_of_uncertainty_set
        self.setting = setting

    @staticmethod
    def merge_mesh_grid(array, n):
        """
        A method used in perturbation_function method, allows easy computation of the
        output at the regression points
        :param array: The multidimensional array we need to make simpler (1D)
        :param n: The total number of interesting points
        :return: The simplified array
        """
        if n == 1:
            return [array]
        else:
            output = []
            for i in range(len(array)):
                output = output + RobustifyLargePosynomial. \
                    merge_mesh_grid(array[i], n / (len(array) + 0.0))
            return output

    @staticmethod
    def perturbation_function(perturbation_vector, type_of_uncertainty_set, number_of_regression_points):
        """
        A method used to do the linear regression
        :param type_of_uncertainty_set: the type of uncertainty set
        :param perturbation_vector: A list representing the perturbation associated
        with each uncertain parameter
        :param number_of_regression_points: The number of regression points
        per dimension
        :return: the regression coefficients and intercept
        """
        dim = len(perturbation_vector)
        result, input_list = [], []
        if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'one norm' or dim == 1:
            if dim == 1:
                x = [np.linspace(-1, 1, number_of_regression_points)]
            else:
                x = np.meshgrid(*[np.linspace(-1, 1, number_of_regression_points)] * dim)

            for _ in range(number_of_regression_points ** dim):
                input_list.append([])

            for i in range(dim):
                temp = RobustifyLargePosynomial.merge_mesh_grid(x[i], number_of_regression_points ** dim)
                for j in range(number_of_regression_points ** dim):
                    input_list[j].append(temp[j])

        else:
            theta_mesh_grid = np.meshgrid(*[np.linspace(0, 2*np.pi - 2*np.pi/number_of_regression_points,
                                                        number_of_regression_points)] * (dim - 1))
            thetas_list = []
            for _ in range(number_of_regression_points ** (dim - 1)):
                thetas_list.append([])
            for i in range(dim - 1):
                temp = RobustifyLargePosynomial.merge_mesh_grid(theta_mesh_grid[i], number_of_regression_points ** (dim - 1))
                for j in range(number_of_regression_points ** (dim - 1)):
                    thetas_list[j].append(temp[j])
            for i in range(number_of_regression_points ** (dim - 1)):
                an_input_list = []
                for j in range(dim):
                    product = 1
                    for k in range(j):
                        product *= np.cos(thetas_list[i][k])
                    if j != dim - 1:
                        product *= np.sin(thetas_list[i][j])
                    an_input_list.append(product)
                input_list.append(an_input_list)

        num_of_inputs = len(input_list)
        for i in range(num_of_inputs):
            output = 1
            for j in range(dim):
                if perturbation_vector[j] != 0:
                    output = output * perturbation_vector[j] ** input_list[i][j]
            result.append(output)
        max_index, max_value, min_index, min_value = None, -np.inf, None, np.inf
        for i, element in enumerate(result):
            if element < min_value:
                min_value = element
                min_index = i
            if element >= max_value:
                max_value = element
                max_index = i
        tol = float(0)
        the_index = -1
        while tol <= 1e-4:
            the_index += 1
            tol = abs(input_list[min_index][the_index] - input_list[max_index][the_index])

        capital_a = []
        b = []
        y_m_i = input_list[min_index][the_index] - input_list[max_index][the_index]
        back_count = 0
        for k in range(num_of_inputs):
            if k != max_index and k != min_index:
                capital_a.append([])
                y_ratio = (input_list[k][the_index] - input_list[max_index][the_index])/y_m_i
                b.append(result[k] + max_value*(y_ratio - 1) - min_value*y_ratio)
                for l in range(dim):
                    if l != the_index:
                        y_k_l = input_list[k][l] - input_list[max_index][l]
                        y_m_l = input_list[min_index][l] - input_list[max_index][l]
                        capital_a[k-back_count].append(y_k_l - y_m_l*y_ratio)
            else:
                back_count += 1

        capital_a_trans = list(map(list, list(zip(*capital_a))))
        capital_b = np.dot(capital_a_trans, capital_a)
        r_h_s = np.dot(capital_a_trans, b)

        try:
            solution = list(np.linalg.solve(capital_b, r_h_s))
            l1 = 0
            l2 = 0
            the_sum = 0
            while l1 < dim - 1:
                if l2 != the_index:
                    y_m_l = input_list[min_index][l2] - input_list[max_index][l2]
                    the_sum += solution[l1]*y_m_l
                    l1 += 1
                l2 += 1
            a_i = (min_value - max_value - the_sum)/y_m_i
            coeff = solution[0:the_index] + [a_i] + solution[the_index:len(solution)]
            the_sum = 0
            for l in range(dim):
                the_sum += coeff[l]*input_list[max_index][l]
            intercept = max_value - the_sum
        except np.linalg.LinAlgError:
            coeff = [(min_value - max_value)/y_m_i]
            intercept = max_value - coeff[0]*input_list[max_index][the_index]
        return coeff, intercept

    def linearize_perturbations(self, p_uncertain_vars, number_of_regression_points):
        """
        A method used to linearize uncertain exponential functions
        :param p_uncertain_vars: the uncertain variables in the posynomial
        :param number_of_regression_points: The number of regression points per dimension
        :return: The linear regression of all the exponential functions, and the mean vector
        """
        etas = []
        mean_vector = [] #TODO: remove mean vector, full of ones
        coeff, intercept = [], []

        for i in range(len(p_uncertain_vars)):
            etas.append(RobustGPTools.generate_etas(p_uncertain_vars[i]))

        perturbation_matrix = []
        for i in range(len(self.p.exps)):
            only_uncertain_vars_monomial_exps = RobustGPTools.\
                only_uncertain_vars_monomial(self.p.exps[i])
            perturbation_matrix.append([])
            mon_uncertain_vars = [var for var in only_uncertain_vars_monomial_exps
                                  if RobustGPTools.is_directly_uncertain(var)]
            mean = 1
            for j, var in enumerate(p_uncertain_vars):
                if var.key in mon_uncertain_vars:
                    perturbation_matrix[i].append(np.exp(only_uncertain_vars_monomial_exps.get(var.key) * etas[j]))
                else:
                    perturbation_matrix[i].append(0)
            coeff.append([])
            intercept.append([])
            coeff[i], intercept[i] = RobustifyLargePosynomial. \
                perturbation_function(perturbation_matrix[i], self.type_of_uncertainty_set, number_of_regression_points)
            mean_vector.append(mean)

        return coeff, intercept, mean_vector

    def no_coefficient_monomials(self):
        """
        separates the monomials in a posynomial into a list of monomials
        with no coefficients
        :return: The list of monomials
        """
        monmaps = [NomialMap({exp: 1.}) for exp, c in self.p.hmap.items()]
        for monmap in monmaps:
            monmap.units = self.p.hmap.units
        mons = [Monomial(monmap) for monmap in monmaps]
        for mony in mons:
            mony.pof = self.p.pof
        return mons

    @staticmethod
    def generate_robust_constraints(gamma, type_of_uncertainty_set,
                                    monomials, perturbation_matrix,
                                    intercept, mean_vector, enable_sp, m):
        """
        :param gamma: controls the size of the uncertainty set
        :param type_of_uncertainty_set: box, elliptical, or one norm
        :param monomials: the list of monomials
        :param perturbation_matrix: the matrix of perturbations
        :param intercept: the list of intercepts
        :param mean_vector: the list of means
        :param enable_sp: whether or not we prefer sp solutions
        :param m: the index
        :return: the robust set of constraints
        """
        constraints = []
        s_main = Variable("s_%s" % m)
        if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'one norm':

            constraints += [sum([a * b for a, b in
                                 zip([a * b for a, b in
                                      zip(mean_vector, intercept)], monomials)]) + gamma * s_main <= 1]
        elif type_of_uncertainty_set == 'elliptical':

            constraints += [sum([a * b for a, b in
                                 zip([c * d for c, d in
                                      zip(mean_vector, intercept)], monomials)]) + gamma * s_main ** 0.5 <= 1]
        ss = []

        for i in range(len(perturbation_matrix[0])):
            positive_pert, negative_pert = [], []
            positive_monomials, negative_monomials = [], []

            if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'elliptical':

                s = Variable("s^%s_%s" % (i, m))
                ss.append(s)
            else:
                s = s_main
            for j in range(len(perturbation_matrix)):
                if perturbation_matrix[j][i] > 0:
                    positive_pert.append(mean_vector[j] * perturbation_matrix[j][i])
                    positive_monomials.append(monomials[j])
                elif perturbation_matrix[j][i] < 0:
                    negative_pert.append(-mean_vector[j] * perturbation_matrix[j][i])
                    negative_monomials.append(monomials[j])
            if enable_sp:
                with SignomialsEnabled():
                    if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'one norm':
                        if negative_pert and not positive_pert:
                            constraints += [sum([a * b for a, b in
                                                 zip(negative_pert, negative_monomials)]) <= s]
                        elif positive_pert and not negative_pert:
                            constraints += [sum([a * b for a, b in
                                                 zip(positive_pert, positive_monomials)]) <= s]
                        else:
                            constraints += [sum([a * b for a, b in
                                                 zip(positive_pert, positive_monomials)]) -
                                            sum([a * b for a, b in
                                                 zip(negative_pert, negative_monomials)]) <= s]
                            constraints += [sum([a * b for a, b in
                                                 zip(negative_pert, negative_monomials)]) -
                                            sum([a * b for a, b in
                                                 zip(positive_pert, positive_monomials)]) <= s]
                    elif type_of_uncertainty_set == 'elliptical':
                        if negative_pert and not positive_pert:
                            constraints += [(sum([a * b for a, b in
                                                 zip(negative_pert, negative_monomials)]))**2 <= s]
                        elif positive_pert and not negative_pert:
                            constraints += [(sum([a * b for a, b in
                                                 zip(positive_pert, positive_monomials)]))**2 <= s]
                        else:
                            dummiest = Variable()
                            constraints += [dummiest**2 <= s]
                            constraints += [(sum([a * b for a, b in zip(positive_pert, positive_monomials)]) -
                                             (sum([a * b for a, b in zip(negative_pert, negative_monomials)]))) <= dummiest]
                            constraints += [(sum([a * b for a, b in zip(negative_pert, negative_monomials)]) -
                                             (sum([a * b for a, b in zip(positive_pert, positive_monomials)]))) <= dummiest]
            else:
                if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'one norm':
                    if positive_pert:
                        constraints += [sum([a * b for a, b in
                                             zip(positive_pert, positive_monomials)]) <= s]
                    if negative_pert:
                        constraints += [sum([a * b for a, b in
                                             zip(negative_pert, negative_monomials)]) <= s]
                elif type_of_uncertainty_set == 'elliptical':
                    constraints += [sum([a * b for a, b in
                                         zip(positive_pert, positive_monomials)]) ** 2
                                    + sum([a * b for a, b in
                                           zip(negative_pert, negative_monomials)]) ** 2 <= s]
        if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'elliptical':
            constraints.append(sum(ss) <= s_main)
        return constraints

    def robustify_large_posynomial(self, type_of_uncertainty_set, m,
                                   setting):
        """
        generate a safe approximation for large posynomials with uncertain coefficients
        :param type_of_uncertainty_set: 'box', elliptical, or 'one norm'
        :param m: Index
        :param setting: robustness setting
        :return: set of robust constraints
        """
        p_direct_uncertain_vars = [var for var in self.p.varkeys if RobustGPTools.is_directly_uncertain(var)]
        p_indirect_uncertain_vars = [var for var in self.p.varkeys if RobustGPTools.is_indirectly_uncertain(var)]

        new_direct_uncertain_vars = []
        for var in p_indirect_uncertain_vars:
            new_direct_uncertain_vars += list(RobustGPTools.\
                replace_indirect_uncertain_variable_by_equivalent(var.key.rel, 1).keys())
        new_direct_uncertain_vars = [var for var in new_direct_uncertain_vars
                                     if RobustGPTools.is_directly_uncertain(var)]
        p_uncertain_vars = list(set(p_direct_uncertain_vars) | set(new_direct_uncertain_vars))
        if (not p_uncertain_vars and not p_indirect_uncertain_vars) or setting.get('gamma') == 0:
            return [self.p <= 1]

        perturbation_matrix, intercept, mean_vector = \
            self.linearize_perturbations(p_uncertain_vars, setting.get('numberOfRegressionPoints'))

        monomials = self.no_coefficient_monomials()
        if self.setting.get('constraintwise'):
            gamma = self.p.pof
        else:
            gamma = setting.get('gamma')
        constraints = RobustifyLargePosynomial. \
            generate_robust_constraints(gamma, type_of_uncertainty_set, monomials,
                                        perturbation_matrix, intercept,
                                        mean_vector, setting.get('enableSP'), m)
        if self.p.pof:
            for constraint in constraints:
                constraint.pof = self.p.pof
        return constraints

if __name__ == '__main__':
    pass

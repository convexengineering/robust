import numpy as np
from gpkit import Variable, Monomial, SignomialsEnabled, Posynomial
from numbers import Number

from RobustGPTools import RobustGPTools


class RobustifyLargePosynomial:

    def __init__(self, p, type_of_uncertainty_set, number_of_stds, setting):
        self.p = p
        self.type_of_uncertainty_set = type_of_uncertainty_set
        self.number_of_stds = number_of_stds
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
            for i in xrange(len(array)):
                output = output + RobustifyLargePosynomial. \
                    merge_mesh_grid(array[i], n / (len(array) + 0.0))
            return output

    @staticmethod
    def perturbation_function(perturbation_vector, number_of_regression_points):
        """
        A method used to do the linear regression
        :param perturbation_vector: A list representing the perturbation associated
        with each uncertain parameter
        :param number_of_regression_points: The number of regression points
        per dimension
        :return: the regression coefficients and intercept
        """
        print perturbation_vector
        dim = len(perturbation_vector)
        # print(dim, number_of_regression_points)
        if dim != 1:
            x = np.meshgrid(*[np.linspace(-1, 1, number_of_regression_points)] * dim)
        else:
            x = [np.linspace(-1, 1, number_of_regression_points)]

        result, input_list = [], []
        for _ in xrange(number_of_regression_points ** dim):
            input_list.append([])

        for i in xrange(dim):
            temp = RobustifyLargePosynomial.merge_mesh_grid(x[i], number_of_regression_points ** dim)
            for j in xrange(number_of_regression_points ** dim):
                input_list[j].append(temp[j])

        for i in xrange(number_of_regression_points ** dim):
            output = 1
            for j in xrange(dim):
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

        y = [[(min_value - max_value) / (input_list[min_index][0] - input_list[max_index][0])]]

        for i in range(1, dim):
            y.append([(input_list[min_index][i] - input_list[max_index][i]) /
                      (input_list[min_index][0] - input_list[max_index][0])])

        b = []
        for i in xrange(number_of_regression_points ** dim):
            if i != max_index and i != min_index:
                b.append([result[i] - max_value - y[0][0] * (input_list[i][0] - input_list[max_index][0])])

        capital_a = []
        for i in range(0, number_of_regression_points ** dim):
            if i != max_index and i != min_index:
                capital_a.append([])
                for j in range(1, dim):
                    capital_a[-1].append(input_list[i][j] - input_list[max_index][j] -
                                         y[j][0] * (input_list[i][0] - input_list[max_index][0]))

        capital_a_trans = map(list, zip(*capital_a))

        capital_b = np.dot(capital_a_trans, capital_a)
        r_h_s = np.dot(capital_a_trans, b)
        try:
            solution = list(np.linalg.solve(capital_b, r_h_s))
            solution = [list(i) for i in solution]

            temp = y[0][0] - list(list(np.dot(map(list, zip(*solution)), y[1:]))[0])
            temp = [list(temp)]

            coeff = temp + solution
            intercept = max_value - np.dot(map(list, zip(*coeff)), input_list[max_index])
            intercept = list(intercept)[0]
            coeff = [i[0] for i in coeff]
        except:
            coeff = [y[0][0]]
            intercept = max_value - coeff[0]*input_list[max_index][0]

        return coeff, intercept

    def linearize_perturbations(self, p_uncertain_vars, number_of_regression_points):
        """
        A method used to linearize uncertain exponential functions
        :param p_uncertain_vars: the uncertain variables in the posynomial
        :param number_of_regression_points: The number of regression points per dimension
        :return: The linear regression of all the exponential functions, and the mean vector
        """
        center, scale = [], []
        mean_vector = []
        coeff, intercept = [], []

        for i in xrange(len(p_uncertain_vars)):
            eta_min, eta_max = RobustGPTools.generate_etas(p_uncertain_vars[i], self.type_of_uncertainty_set,
                                                           self.number_of_stds, self.setting)
            center.append((eta_min + eta_max) / 2.0)
            scale.append(eta_max - center[i])

        perturbation_matrix = []
        for i in xrange(len(self.p.exps)):
            # print self.p.exps[i]
            only_uncertain_vars_monomial_exps = RobustGPTools.\
                only_uncertain_vars_monomial(self.p.exps[i])
            # print only_uncertain_vars_monomial_exps
            perturbation_matrix.append([])
            mon_uncertain_vars = [var for var in only_uncertain_vars_monomial_exps
                                  if RobustGPTools.is_directly_uncertain(var)]
            # print mon_uncertain_vars
            mean = 1
            for j, var in enumerate(p_uncertain_vars):
                if var.key in mon_uncertain_vars:
                    mean = mean * np.exp(center[j]*only_uncertain_vars_monomial_exps.get(var.key))
                    perturbation_matrix[i].append(np.exp(only_uncertain_vars_monomial_exps.get(var.key) * scale[j]))
                else:
                    perturbation_matrix[i].append(0)
            coeff.append([])
            intercept.append([])
            coeff[i], intercept[i] = RobustifyLargePosynomial. \
                perturbation_function(perturbation_matrix[i], number_of_regression_points)
            mean_vector.append(mean)

        return coeff, intercept, mean_vector

    def no_coefficient_monomials(self):
        """
        separates the monomials in a posynomial into a list of monomials
        :return: The list of monomials
        """
        monomials = []
        for i in xrange(len(self.p.exps)):
            monomials.append(Monomial(self.p.exps[i], self.p.cs[i]))
        return monomials

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

        for i in xrange(len(perturbation_matrix[0])):
            positive_pert, negative_pert = [], []
            positive_monomials, negative_monomials = [], []

            if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'elliptical':

                s = Variable("s^%s_%s" % (i, m))
                ss.append(s)
            else:
                s = s_main
            for j in xrange(len(perturbation_matrix)):
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

                        constraints += [(sum([a * b for a, b in
                                              zip(positive_pert, positive_monomials)])
                                         - sum([a * b for a, b in
                                                zip(negative_pert, negative_monomials)])) ** 2 <= s]
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
            print 'eker', RobustGPTools.\
                replace_indirect_uncertain_variable_by_equivalent(var.key.rel, 1)
            print 'eker', RobustGPTools.\
                replace_indirect_uncertain_variable_by_equivalent(var.key.rel, 1).keys()
            print 'eker', [i.key for i in RobustGPTools.\
                replace_indirect_uncertain_variable_by_equivalent(var.key.rel, 1).keys()]
            new_direct_uncertain_vars += RobustGPTools.\
                replace_indirect_uncertain_variable_by_equivalent(var.key.rel, 1).keys()
        print new_direct_uncertain_vars
        new_direct_uncertain_vars = [var for var in new_direct_uncertain_vars
                                     if RobustGPTools.is_directly_uncertain(var)]
        print new_direct_uncertain_vars
        p_uncertain_vars = list(set(p_direct_uncertain_vars) | set(new_direct_uncertain_vars))
        print p_uncertain_vars
        if (not p_uncertain_vars and not p_indirect_uncertain_vars) or setting.get('gamma') == 0:
            return [self.p <= 1]

        perturbation_matrix, intercept, mean_vector = \
            self.linearize_perturbations(p_uncertain_vars, setting.get('numberOfRegressionPoints'))

        monomials = self.no_coefficient_monomials()
        constraints = RobustifyLargePosynomial. \
            generate_robust_constraints(setting.get('gamma'), type_of_uncertainty_set, monomials,
                                        perturbation_matrix, intercept,
                                        mean_vector, setting.get('enableSP'), m)
        return constraints

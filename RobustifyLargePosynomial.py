import numpy as np
from gpkit import Variable, Monomial, SignomialsEnabled, Posynomial
from sklearn import linear_model


class RobustifyLargePosynomial:

    p = Posynomial()

    def __init__(self, p):
        self.p = p

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
        dim = len(perturbation_vector)
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

        clf = linear_model.LinearRegression()
        clf.fit(input_list, result)
        return clf.coef_, clf.intercept_

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
            pr = p_uncertain_vars[i].key.pr
            center.append(np.sqrt(1 - pr ** 2 / 10000.0))
            scale.append(0.5 * np.log((1 + pr / 100.0) / (1 - pr / 100.0)))

        perturbation_matrix = []
        for i in xrange(len(self.p.exps)):
            perturbation_matrix.append([])
            mon_uncertain_vars = [var for var in p_uncertain_vars if var in self.p.exps[i]]
            mean = 1
            for j, var in enumerate(p_uncertain_vars):
                if var.key in mon_uncertain_vars:
                    mean = mean * center[j] ** (self.p.exps[i].get(var.key))
                    perturbation_matrix[i].append(np.exp(self.p.exps[i].get(var.key) * scale[j]))
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
    def generate_robust_constraints(type_of_uncertainty_set,
                                    monomials, perturbation_matrix,
                                    intercept, mean_vector, enable_sp, m):
        """

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
                                      zip(mean_vector, intercept)], monomials)]) + s_main <= 1]
        elif type_of_uncertainty_set == 'elliptical':

            constraints += [sum([a * b for a, b in
                                 zip([a * b for a, b in
                                      zip(mean_vector, intercept)], monomials)]) + s_main ** 0.5 <= 1]
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

    def robustify_large_posynomial(self, type_of_uncertainty_set, uncertain_vars, m,
                                   enable_sp, number_of_regression_points):
        """
        generate a safe approximation for large posynomials with uncertain coefficients
        :param type_of_uncertainty_set: 'box', elliptical, or 'one norm'
        :param uncertain_vars: Model's uncertain variables
        :param m: Index
        :param enable_sp: choose whether an sp compatible model is okay
        :param number_of_regression_points: number of regression points per dimension
        :return: set of robust constraints
        """
        p_uncertain_vars = [var for var in self.p.varkeys if var in uncertain_vars]

        perturbation_matrix, intercept, mean_vector = \
            self.linearize_perturbations(p_uncertain_vars, number_of_regression_points)

        if not p_uncertain_vars:
            return [self.p <= 1]

        monomials = self.no_coefficient_monomials()
        constraints = RobustifyLargePosynomial. \
            generate_robust_constraints(type_of_uncertainty_set, monomials,
                                        perturbation_matrix, intercept,
                                        mean_vector, enable_sp, m)
        return constraints

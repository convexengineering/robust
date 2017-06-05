from EquivalentModels import TractableModel, SameModel
from RobustifyLargePosynomial import RobustifyLargePosynomial
from gpkit import Model

import numpy as np


class RobustGPModel(Model):
    """
    Creates robust Models starting from a model with uncertain coefficients
    """

    initial_guess = None
    number_of_pwl_approximations = None

    def setup(self, model, gamma, type_of_uncertainty_set, r_min=5, tol=0.001,
              simple_model=False, number_of_regression_points=2,
              linearize_two_term=True, enable_sp=True, boyd=False, two_term=None,
              simple_two_term=True, maximum_number_of_permutations=30):
        """
        Constructs a robust model starting from a model with uncertain coefficients
        :param model: the original uncertain model
        :param gamma: Controls the size of the uncertainty set
        :param type_of_uncertainty_set: box, elliptical, or one norm set
        :param r_min: The minimum number of PWL functions
        :param tol: Determines the accuracy of PWL
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param number_of_regression_points: The number of points per dimension used to replace exponential uncertainty
        function by a linear function
        :param linearize_two_term: linearize two term functions rather than considering them large posynomials
        :param enable_sp: choose to solve an SP to get a better solution
        :param boyd: choose to apply boyd's two term approximation
        :param two_term: whether two term approximation is preferred
        :param simple_two_term: whether a simple two term approximation is preferred
        :param maximum_number_of_permutations: the number of maximum two term approximations
        :return: The robust Model, The initial guess if the robust model is an SP, and the number of PWL functions used
        to approximate two term monomials
        """
        r = r_min
        error = 1
        sol = 0

        if two_term is None:
            if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'one norm':
                two_term = True
            elif type_of_uncertainty_set == 'elliptical':
                two_term = False

        no_data_constraints_upper, no_data_constraints_lower, data_constraints = RobustGPModel. \
            robust_model_fixed_r(model, gamma, r, type_of_uncertainty_set,
                                 tol, simple_model, number_of_regression_points,
                                 two_term, simple_two_term, maximum_number_of_permutations,
                                 linearize_two_term, False, boyd)

        model_upper = Model(model.cost,
                            [no_data_constraints_upper, data_constraints])
        model_upper.substitutions.update(model.substitutions)

        model_lower = Model(model.cost,
                            [no_data_constraints_lower, data_constraints])
        model_lower.substitutions.update(model.substitutions)

        while r <= 20 and error > tol:
            flag = 0
            try:
                sol_upper = model_upper.solve(verbosity=0)
                sol = sol_upper
            except:
                flag = 1
            try:
                sol_lower = model_lower.solve(verbosity=0)
            except RuntimeError:
                r = 21
                sol = self.solve(verbosity=0)
                break

            if flag != 1:
                try:
                    error = (sol_upper.get('cost').m -
                             sol_lower.get('cost').m) / (0.0 + sol_lower.get('cost').m)
                except:
                    error = (sol_upper.get('cost') -
                             sol_lower.get('cost')) / (0.0 + sol_lower.get('cost'))
            r += 1
            no_data_constraints_upper, no_data_constraints_lower, data_constraints = \
                self.robust_model_fixed_r(model, gamma, r, type_of_uncertainty_set, tol,
                                          simple_model, number_of_regression_points,
                                          two_term, simple_two_term, maximum_number_of_permutations,
                                          linearize_two_term, False, boyd)

            model_upper = Model(model.cost,
                                [no_data_constraints_upper, data_constraints])
            model_upper.substitutions.update(model.substitutions)

            model_lower = Model(model.cost,
                                [no_data_constraints_lower, data_constraints])
            model_lower.substitutions.update(model.substitutions)

        initial_guess = sol.get("variables")

        if enable_sp:
            no_data_constraints_upper, no_data_constraints_lower, data_constraints = \
                self.robust_model_fixed_r(model, gamma, r - 1, type_of_uncertainty_set,
                                          tol, False, number_of_regression_points,
                                          False, simple_two_term, maximum_number_of_permutations,
                                          linearize_two_term, True, boyd)

            model_upper = Model(model.cost,
                                [no_data_constraints_upper, data_constraints])
            model_upper.substitutions.update(model.substitutions)

            model_lower = Model(model.cost,
                                [no_data_constraints_lower, data_constraints])
            model_lower.substitutions.update(model.substitutions)

            subs_vars = model_upper.substitutions.keys()

            for i in xrange(len(subs_vars)):
                del initial_guess[subs_vars[i].key]

        self.number_of_pwl_approximations = r
        self.initial_guess = initial_guess
        self.cost = model.cost

        return [no_data_constraints_upper, data_constraints]

    @staticmethod
    def uncertain_variables_exponents(data_monomials, uncertain_vars):
        """
        gets the exponents of uncertain variables
        :param data_monomials:  the uncertain posynomials
        :param uncertain_vars: the uncertain variables of the model
        :return: the 2 dimensional array of exponents(matrix)
        """
        exps_of_uncertain_vars = \
            np.array([[-p.exps[0].get(var.key, 0) for var in uncertain_vars]
                      for p in data_monomials])
        return exps_of_uncertain_vars

    @staticmethod
    def normalize_perturbation_vector(uncertain_vars):
        """
        normalizes the perturbation elements
        :param uncertain_vars: the uncertain variables of the model
        :return: the centering and scaling vector
        """
        prs = np.array([var.key.pr for var in uncertain_vars])
        eta_max = np.log(1 + prs / 100.0)
        eta_min = np.log(1 - prs / 100.0)
        centering_vector = (eta_min + eta_max) / 2.0
        scaling_vector = eta_max - centering_vector
        return centering_vector, scaling_vector

    @staticmethod
    def construct_robust_monomial_coefficients(exps_of_uncertain_vars, gamma,
                                               type_of_uncertainty_set,
                                               centering_vector, scaling_vector):
        """
        robustify monomials
        :param exps_of_uncertain_vars: the matrix of exponents
        :param gamma: controls the size of the uncertainty set
        :param type_of_uncertainty_set: box, elliptical, or one norm
        :param centering_vector: centers the perturbations around zero
        :param scaling_vector: scales the perturbation to 1
        :return: the coefficient that will multiply the left hand side of a constraint
        """
        b_pert = (exps_of_uncertain_vars * scaling_vector[np.newaxis])
        coefficient = []
        for i in xrange(exps_of_uncertain_vars.shape[0]):
            norm = 0
            centering = 0
            for j in range(exps_of_uncertain_vars.shape[1]):
                if type_of_uncertainty_set == 'box':
                    norm += np.abs(b_pert[i][j])
                elif type_of_uncertainty_set == 'elliptical':
                    norm += b_pert[i][j] ** 2
                elif type_of_uncertainty_set == 'one norm':
                    norm = max(norm, np.abs(b_pert[i][j]))
                else:
                    raise Exception('This type of set is not supported')
                centering = centering + exps_of_uncertain_vars[i][j] * centering_vector[j]
            if type_of_uncertainty_set == 'elliptical':
                norm = np.sqrt(norm)
            coefficient.append([np.exp(gamma * norm) / np.exp(centering)])
        return coefficient

    @staticmethod
    def robust_model_fixed_r(model, gamma, r, type_of_uncertainty_set,
                             tol, simple_model, number_of_regression_points, two_term,
                             simple_two_term, maximum_number_of_permutations,
                             linearize_two_term, enable_sp, boyd):
        """
        generates a robust model for a fixed number of piece-wise linear functions
        :param model: the original model
        :param gamma: Controls the size of the uncertainty set
        :param r: the number of piece-wise linear functions per two term constraints
        :param type_of_uncertainty_set: box, elliptical, or one norm set
        :param tol: Determines the accuracy of PWL
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param number_of_regression_points: The number of points per dimension used to
        replace exponential uncertainty
        function by a linear function
        :param two_term: Solve the problem using two term decoupling rather than linear
        approximation of exponential uncertainties. If the problem is small or the
        number of uncertain variables per constraint is small, switch two_term to True
        :param simple_two_term: if a simple two term approximation is preferred
        :param maximum_number_of_permutations: the maximum allowed number of permutations for two term approximation
        :param linearize_two_term: linearize two term functions rather than considering
        them large posynomials
        :param enable_sp: choose to solve an SP to get a better solution
        :param boyd: choose to apply boyd's two term approximation
        :return: the upper-bound of the robust model and the lower-bound of the robust model
        """

        if type_of_uncertainty_set == 'box':
            dependent_uncertainty_set = False
        else:
            dependent_uncertainty_set = True

        uncertain_vars = SameModel.uncertain_model_variables(model)

        tractable_model = TractableModel(model, r, tol, uncertain_vars, simple_model, dependent_uncertainty_set,
                                         two_term, maximum_number_of_permutations, simple_two_term,
                                         linearize_two_term, boyd)

        simplified_model_upper, simplified_model_lower = tractable_model.get_tractable_models()
        number_of_no_data_constraints = tractable_model.get_number_of_no_data_constraints()

        no_data_constraints_upper, no_data_constraints_lower = [], []
        data_constraints, data_monomials = [], []

        posynomials_upper = simplified_model_upper.as_posyslt1()
        posynomials_lower = simplified_model_lower.as_posyslt1()

        for i, p in enumerate(posynomials_upper):

            if i < number_of_no_data_constraints:
                no_data_constraints_upper += [p <= 1]
                no_data_constraints_lower += [posynomials_lower[i] <= 1]
            else:
                if len(p.exps) > 1:
                    robust_large_p = RobustifyLargePosynomial(p)
                    data_constraints.append(robust_large_p.
                                            robustify_large_posynomial
                                            (type_of_uncertainty_set, uncertain_vars, i,
                                             enable_sp, number_of_regression_points))
                else:
                    data_monomials.append(p)

        exps_of_uncertain_vars = RobustGPModel. \
            uncertain_variables_exponents(data_monomials, uncertain_vars)

        if exps_of_uncertain_vars.size > 0:
            centering_vector, scaling_vector = RobustGPModel. \
                normalize_perturbation_vector(uncertain_vars)
            # noinspection PyTypeChecker
            coefficient = RobustGPModel. \
                construct_robust_monomial_coefficients(exps_of_uncertain_vars, gamma,
                                                       type_of_uncertainty_set,
                                                       centering_vector, scaling_vector)
            for i in xrange(len(data_monomials)):
                data_constraints += [coefficient[i][0] * data_monomials[i] <= 1]

        return no_data_constraints_upper, no_data_constraints_lower, data_constraints

    def solve(self, verbosity=0, initial_guess=None):
        if initial_guess is None:
            initial_guess = {}
        try:
            sol = self.solve(verbosity=verbosity)
        except:
            sol = self.localsolve(verbosity=verbosity, x0=initial_guess)
        return sol

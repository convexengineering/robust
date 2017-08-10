from EquivalentModels import SameModel, EquivalentModel, TwoTermModel
from TwoTermApproximation import TwoTermApproximation
from RobustifyLargePosynomial import RobustifyLargePosynomial
from gpkit import Model, Monomial
from LinearizeTwoTermPosynomials import LinearizeTwoTermPosynomials

import numpy as np
import time


class RobustGPModel:
    ready_constraints = []
    tractable_posynomials = []
    to_linearize_posynomials = []
    large_posynomials = []

    model = None
    substitutions = None
    gamma = None
    type_of_uncertainty_set = None
    simple_model = None
    number_of_regression_points = None
    linearize_two_term = None
    enable_sp = None
    boyd = None
    two_term = None
    simple_two_term = None
    smart_two_term_choose = None
    maximum_number_of_permutations = None
    uncertain_vars = None
    slopes = None
    intercepts = None
    r_min = None
    tol = None
    r = None

    initial_guess = None
    robust_model = None

    def __init__(self, model, gamma, type_of_uncertainty_set, simple_model=False, number_of_regression_points=2,
                 linearize_two_term=True, enable_sp=True, boyd=False, two_term=False, simple_two_term=True,
                 smart_two_term_choose=True, maximum_number_of_permutations=30):
        if two_term:
            linearize_two_term = True
            enable_sp = False

        self.model = model
        self.substitutions = model.substitutions
        self.gamma = gamma
        self.type_of_uncertainty_set = type_of_uncertainty_set
        self.simple_model = simple_model
        self.number_of_regression_points = number_of_regression_points
        self.linearize_two_term = linearize_two_term
        self.enable_sp = enable_sp
        self.boyd = boyd
        self.two_term = two_term
        self.simple_two_term = simple_two_term
        self.smart_two_term_choose = smart_two_term_choose
        self.maximum_number_of_permutations = maximum_number_of_permutations

        self.ready_constraints = []
        self.tractable_posynomials = []
        self.to_linearize_posynomials = []
        self.large_posynomials = []
        self.slopes = None
        self.intercepts = None
        self.r_min = None
        self.tol = None
        self.r = None

        self.initial_guess = None
        self.robust_model = None

        if self.type_of_uncertainty_set == 'box':
            dependent_uncertainty_set = False
        else:
            dependent_uncertainty_set = True

        self.uncertain_vars = SameModel.uncertain_model_variables(model)

        if boyd:
            safe_model = TwoTermModel(model, self.uncertain_vars, 0, False, True, 1)
            safe_model_posynomials = safe_model.as_posyslt1()
            self.to_linearize_posynomials += safe_model_posynomials
            return

        equivalent_model = EquivalentModel(model, self.uncertain_vars, simple_model, dependent_uncertainty_set)
        equivalent_model_posynomials = equivalent_model.as_posyslt1()
        equivalent_model_no_data_constraints_number = equivalent_model.number_of_no_data_constraints

        for i, p in enumerate(equivalent_model_posynomials):
            if i < equivalent_model_no_data_constraints_number:
                self.ready_constraints += [p <= 1]
            else:
                if len(p.exps) == 1:
                    self.tractable_posynomials += [p]
                elif len(p.exps) == 2 and linearize_two_term:
                    self.to_linearize_posynomials += [p]
                else:
                    if two_term:
                        two_term_approximation = TwoTermApproximation(p, self.uncertain_vars, simple_two_term,
                                                                      False, smart_two_term_choose,
                                                                      maximum_number_of_permutations)
                        self.large_posynomials.append(two_term_approximation)
                    else:
                        robust_large_p = RobustifyLargePosynomial(p)
                        self.ready_constraints += robust_large_p. \
                            robustify_large_posynomial(gamma, type_of_uncertainty_set, self.uncertain_vars, i,
                                                       enable_sp, number_of_regression_points)

    def copy(self, robust_model):
        robust_model_copy = RobustGPModel(robust_model.model, robust_model.gamma, robust_model.type_of_uncertainty_set,
                                          robust_model.simple_model, robust_model.number_of_regression_points,
                                          robust_model.linearize_two_term, robust_model.enable_sp, robust_model.boyd,
                                          robust_model.two_term, robust_model.simple_two_term,
                                          robust_model.maximum_number_of_permutations)

        self.ready_constraints = robust_model.ready_constraints
        self.tractable_posynomials = robust_model.tractable_posynomials
        self.to_linearize_posynomials = robust_model.to_linearize_posynomials
        self.large_posynomials = robust_model.large_posynomials

        self.uncertain_vars = robust_model.uncertain_vars
        self.slopes = robust_model.slopes
        self.intercepts = robust_model.intercepts
        self.r_min = robust_model.r_min
        self.tol = robust_model.tol
        self.r = robust_model.r

        return robust_model_copy

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
        # mean_values = np.array([np.log(var.key.value) for var in uncertain_vars])
        eta_max = np.log(1 + prs / 100.0)
        eta_min = np.log(1 - prs / 100.0)
        # centering_vector = 0
        centering_vector = (eta_min + eta_max) / 2.0
        # scaling_vector = [a*b/100 for a, b in zip(mean_values, prs)]
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

    def calculate_value_of_two_term_approximated_posynomial(self, two_term_approximation, index_of_permutation,
                                                            solution):
        permutation = two_term_approximation.list_of_permutations[index_of_permutation]

        number_of_iterations = int(len(permutation) / 2)

        values = []

        for i in xrange(number_of_iterations):
            monomials = []

            first_monomial = Monomial(two_term_approximation.p.exps[permutation[2 * i]],
                                      two_term_approximation.p.cs[permutation[2 * i]])
            second_monomial = Monomial(two_term_approximation.p.exps[permutation[2 * i + 1]],
                                       two_term_approximation.p.cs[permutation[2 * i + 1]])

            monomials += [first_monomial]
            for j in xrange(self.r - 2):
                monomials += [first_monomial ** self.slopes[self.r - 3 - j] *
                              second_monomial ** self.slopes[j] * np.exp(self.intercepts[j])]
            monomials += [second_monomial]

            exps_of_uncertain_vars = RobustGPModel.uncertain_variables_exponents(monomials, self.uncertain_vars)
            centering_vector, scaling_vector = RobustGPModel.normalize_perturbation_vector(self.uncertain_vars)
            # noinspection PyTypeChecker
            coefficient = RobustGPModel. \
                construct_robust_monomial_coefficients(exps_of_uncertain_vars, self.gamma, self.type_of_uncertainty_set,
                                                       centering_vector, scaling_vector)

            subs_monomials = []
            for j in xrange(len(monomials)):
                monomials[j] *= coefficient[j][0]
                monomials[j] = monomials[j].sub(solution)
                monomials[j] = monomials[j].sub(self.model.substitutions)
                subs_monomials.append(monomials[j].cs[0])

            values.append(max(subs_monomials))

        if number_of_iterations % 2 != 0:
            the_monomial = Monomial(two_term_approximation.p.exps[permutation[len(permutation)-1]],
                                    two_term_approximation.p.cs[permutation[len(permutation)-1]])

            exps_of_uncertain_vars = RobustGPModel.uncertain_variables_exponents([the_monomial], self.uncertain_vars)
            centering_vector, scaling_vector = RobustGPModel.normalize_perturbation_vector(self.uncertain_vars)
            # noinspection PyTypeChecker
            coefficient = RobustGPModel. \
                construct_robust_monomial_coefficients(exps_of_uncertain_vars, self.gamma, self.type_of_uncertainty_set,
                                                       centering_vector, scaling_vector)
            the_monomial *= coefficient[0][0]
            the_monomial = the_monomial.sub(solution)
            the_monomial = the_monomial.sub(self.model.substitutions)
            values.append(the_monomial.cs[0])

        return sum(values)

    def find_permutation_with_minimum_value(self, two_term_approximation, solution):
        minimum_value = 1e5
        minimum_index = len(two_term_approximation.list_of_permutations)
        for i in xrange(len(two_term_approximation.list_of_permutations)):
            temp_value = self. \
                calculate_value_of_two_term_approximated_posynomial(two_term_approximation, i, solution)

            if temp_value < minimum_value:
                minimum_value = temp_value
                minimum_index = i

        return minimum_index

    def setup(self, r_min=10, tol=0.001):
        start_time = time.time()
        self.r_min = r_min
        self.tol = tol

        solution = None

        two_term_data_posynomials = []
        number_of_trials_until_feasibility_is_attained = 0

        while number_of_trials_until_feasibility_is_attained < self.maximum_number_of_permutations:
            for i, two_term_approximation in enumerate(self.large_posynomials):
                perm_index = np.random.choice(range(0, len(two_term_approximation.list_of_permutations)))
                permutation = two_term_approximation.list_of_permutations[perm_index]
                no_data, data = TwoTermApproximation.two_term_equivalent_posynomial(two_term_approximation.p, i,
                                                                                    permutation, False)
                self.ready_constraints += no_data
                two_term_data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

            two_term_data_posynomials += self.to_linearize_posynomials
            self.r, solution = self.find_number_of_piece_wise_linearization(two_term_data_posynomials)
            if self.r != 0:
                break

            number_of_trials_until_feasibility_is_attained += 1

        if number_of_trials_until_feasibility_is_attained >= self.maximum_number_of_permutations:
            print("not feasible !!")

        self.slopes, self.intercepts, _, _, _ = LinearizeTwoTermPosynomials.\
            two_term_posynomial_linearization_coeff(self.r, self.tol)

        old_solution = solution
        robust_model = None

        for _ in xrange(self.maximum_number_of_permutations):
            permutation_indices = self.new_permutation_indices(solution)

            two_term_data_posynomials = []

            for i, two_term_approximation in enumerate(self.large_posynomials):
                permutation = two_term_approximation.list_of_permutations[permutation_indices[i]]
                _, data = TwoTermApproximation.two_term_equivalent_posynomial(two_term_approximation.p, i,
                                                                              permutation, False)
                two_term_data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

            two_term_data_posynomials += self.to_linearize_posynomials
            robust_model, _ = self.linearize_and_return_upper_lower_models(two_term_data_posynomials, self.r)

            new_solution = RobustGPModel.internal_solve(robust_model, self.initial_guess)

            same_solution = RobustGPModel.same_solution(old_solution, new_solution)
            if same_solution:
                break
            else:
                old_solution = new_solution
        print("the model need %s seconds to setup" % (time.time() - start_time))
        self.robust_model = robust_model
        return robust_model

    def linearize_and_return_upper_lower_models(self, two_term_data_posynomials, r):
        no_data_upper_constraints = []
        no_data_lower_constraints = []
        data_constraints = []
        data_posynomials = []

        for i, two_term_p in enumerate(two_term_data_posynomials):
            linearize_p = LinearizeTwoTermPosynomials(two_term_p)
            no_data_upper, no_data_lower, data = linearize_p.linearize_two_term_posynomial(i, r, self.tol)

            no_data_upper_constraints += no_data_upper
            no_data_lower_constraints += no_data_lower
            data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

        all_tractable_posynomials = self.tractable_posynomials + data_posynomials

        exps_of_uncertain_vars = RobustGPModel.uncertain_variables_exponents(all_tractable_posynomials,
                                                                             self.uncertain_vars)

        if exps_of_uncertain_vars.size > 0:
            centering_vector, scaling_vector = RobustGPModel.normalize_perturbation_vector(self.uncertain_vars)
            # noinspection PyTypeChecker
            coefficient = RobustGPModel.construct_robust_monomial_coefficients(exps_of_uncertain_vars, self.gamma,
                                                                               self.type_of_uncertainty_set,
                                                                               centering_vector, scaling_vector)
            for i in xrange(len(all_tractable_posynomials)):
                data_constraints += [coefficient[i][0] * all_tractable_posynomials[i] <= 1]

        model_upper = Model(self.model.cost, [no_data_upper_constraints, self.ready_constraints, data_constraints])
        model_upper.substitutions.update(self.substitutions)

        model_lower = Model(self.model.cost, [no_data_lower_constraints, self.ready_constraints, data_constraints])
        model_lower.substitutions.update(self.substitutions)

        return model_upper, model_lower

    def find_number_of_piece_wise_linearization(self, two_term_data_posynomials):
        error = 2 * self.tol
        r = self.r_min

        sol_upper = None
        sol_lower = None

        lower_used = 0

        while r <= 20 and error > self.tol:
            model_upper, model_lower = self.linearize_and_return_upper_lower_models(two_term_data_posynomials, r)

            upper_model_infeasible = 0
            try:
                sol_upper = RobustGPModel.internal_solve(model_upper, self.initial_guess)
            except:
                upper_model_infeasible = 1
            try:
                sol_lower = RobustGPModel.internal_solve(model_lower, self.initial_guess)
            except RuntimeError:
                return 0, None

            if upper_model_infeasible != 1:
                try:
                    error = (sol_upper.get('cost').m -
                             sol_lower.get('cost').m) / (0.0 + sol_lower.get('cost').m)
                except:
                    error = (sol_upper.get('cost') -
                             sol_lower.get('cost')) / (0.0 + sol_lower.get('cost'))
            elif r == 20:
                lower_used = 1
                print("The safe approximation is infeasible, the lower piece-wise model is considered")
            r += 1
        if lower_used == 0:
            solution = sol_upper
        else:
            solution = sol_lower
        return r, solution

    def solve(self, verbosity=0):
        if self.robust_model is None:
            self.setup()
        if self.initial_guess is None:
            initial_guess = {}
        else:
            initial_guess = self.initial_guess
        try:
            sol = self.robust_model.solve(verbosity=verbosity)
        except:
            sol = self.robust_model.localsolve(verbosity=verbosity, x0=initial_guess)
        return sol

    @staticmethod
    def internal_solve(model, initial_guess):
        if initial_guess is None:
            initial_guess = {}
        try:
            sol = model.solve(verbosity=0)
        except:
            sol = model.localsolve(verbosity=0, x0=initial_guess)
        return sol

    def new_permutation_indices(self, solution):
        permutation_indices = []
        for two_term_approximation in self.large_posynomials:
            permutation_indices.append(self.find_permutation_with_minimum_value(two_term_approximation, solution))
        return permutation_indices

    @staticmethod
    def same_solution(solution_one, solution_two):
        keys_one = solution_one['variables'].keys()
        keys_two = solution_two['variables'].keys()

        if keys_one != keys_two:
            return False

        for key in keys_one:
            relative_difference = np.abs((solution_one['variables'][key] - solution_two['variables'][key])/solution_one['variables'][key])
            if relative_difference > 0.0001:
                return False
        return True

    def generate_initial_guess(self, robust_model, simple=True):
        two_term = None
        if self.type_of_uncertainty_set == 'box' or self.type_of_uncertainty_set == 'one norm':
            two_term = True
        elif self.type_of_uncertainty_set == 'elliptical':
            two_term = False

        if simple:
            robust_model_initial_guess = RobustGPModel(robust_model.model, robust_model.gamma,
                                                       robust_model.type_of_uncertainty_set,
                                                       robust_model.simple_model, robust_model.number_of_regression_points,
                                                       robust_model.linearize_two_term, False, False,
                                                       two_term, True, 1)
        else:
            robust_model_initial_guess = RobustGPModel(robust_model.model, robust_model.gamma, robust_model.type_of_uncertainty_set,
                                                       robust_model.simple_model, robust_model.number_of_regression_points,
                                                       robust_model.linearize_two_term, False, robust_model.boyd,
                                                       two_term, robust_model.simple_two_term,
                                                       robust_model.maximum_number_of_permutations)

        robust_model_initial_guess.setup()
        return robust_model_initial_guess

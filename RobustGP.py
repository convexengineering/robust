import numpy as np
import time
import warnings
from gpkit import Model, Monomial

from RobustGPTools import RobustGPTools
from EquivalentModels import EquivalentModel, TwoTermBoydModel
from TwoTermApproximation import TwoTermApproximation
from RobustifyLargePosynomial import RobustifyLargePosynomial
from LinearizeTwoTermPosynomials import LinearizeTwoTermPosynomials


class RobustGPSetting:
    def __init__(self, **options):
        self._options = {
            'gamma': 1,
            'simpleModel': False,
            'numberOfRegressionPoints': 2,
            'linearizeTwoTerm': True,
            'enableSP': True,
            'boyd': False,
            'twoTerm': True,
            'simpleTwoTerm': False,
            'smartTwoTermChoose': False,
            'allowedNumOfPerms': 30,
            'linearizationTolerance': 0.01,
            'minNumOfLinearSections': 12,
            'maxNumOfLinearSections': 20,
        }
        for key, value in options.iteritems():
            self._options[key] = value

    def get(self, option_name):
        return self._options[option_name]

    def set(self, option_name, value):
        self._options[option_name] = value


class RobustGPModel:
    def __init__(self):
        self.model = None
        self.substitutions = None
        self.type_of_uncertainty_set = None

        self.ready_constraints = []
        self.tractable_posynomials = []
        self.to_linearize_posynomials = []
        self.large_posynomials = []

        self.uncertain_vars, self.indirect_uncertain_vars = None, None

        self.slopes = None
        self.intercepts = None
        self.lower_approximation_used = False
        self.r = None
        self.initial_guess = None
        self.robust_model = None
        self.setup_time = None

        self.setting = None

    @classmethod
    def construct(cls, model, type_of_uncertainty_set, **options):
        robust_gp = cls()

        robust_gp.model = model
        robust_gp.substitutions = model.substitutions
        robust_gp.type_of_uncertainty_set = type_of_uncertainty_set
        robust_gp.uncertain_vars, robust_gp.indirect_uncertain_vars = RobustGPTools.uncertain_model_variables(model)
        robust_gp.setting = RobustGPSetting(**options)

        if robust_gp.setting.get('twoTerm'):
            robust_gp.setting.set('linearizeTwoTerm', True)
            robust_gp.setting.set('enableSP', False)

        if robust_gp.setting.get('simpleModel'):
            robust_gp.setting.set('allowedNumOfPerms', 1)

        if robust_gp.type_of_uncertainty_set == 'box':
            dependent_uncertainty_set = False
        else:
            dependent_uncertainty_set = True

        equality_constraints = False

        if robust_gp.setting.get('boyd'):
            robust_gp.setting.set('allowedNumOfPerms', 0)
            safe_model = TwoTermBoydModel(model)
            # print(safe_model)
            safe_model_posynomials = safe_model.as_posyslt1()
            for p in safe_model_posynomials:
                if len(p.exps) == 1:
                    if 1 / p in safe_model_posynomials:
                        equality_constraints = True
                        robust_gp.ready_constraints += [p <= 1]
                    else:
                        robust_gp.tractable_posynomials += [p]
                else:
                    robust_gp.to_linearize_posynomials += [p]
            if equality_constraints:
                warnings.warn('equality constraints will not be robustified')
            return robust_gp

        equivalent_model = EquivalentModel(model, robust_gp.uncertain_vars, robust_gp.indirect_uncertain_vars,
                                           dependent_uncertainty_set, robust_gp.setting)
        equivalent_model_posynomials = equivalent_model.as_posyslt1()
        equivalent_model_no_data_constraints_number = equivalent_model.number_of_no_data_constraints
        # print(equivalent_model)
        for i, p in enumerate(equivalent_model_posynomials):
            if i < equivalent_model_no_data_constraints_number:
                robust_gp.ready_constraints += [p <= 1]
            else:
                if len(p.exps) == 1:
                    if 1 / p in equivalent_model_posynomials:
                        equality_constraints = True
                        robust_gp.ready_constraints += [p <= 1]
                    else:
                        robust_gp.tractable_posynomials += [p]
                elif len(p.exps) == 2 and robust_gp.setting.get('linearizeTwoTerm'):
                    robust_gp.to_linearize_posynomials += [p]
                else:
                    if robust_gp.setting.get('twoTerm'):
                        two_term_approximation = TwoTermApproximation(p, robust_gp.uncertain_vars,
                                                                      robust_gp.indirect_uncertain_vars,
                                                                      robust_gp.setting)
                        robust_gp.large_posynomials.append(two_term_approximation)
                    else:
                        robust_large_p = RobustifyLargePosynomial(p)
                        robust_gp.ready_constraints += robust_large_p. \
                            robustify_large_posynomial(robust_gp.type_of_uncertainty_set, robust_gp.uncertain_vars,
                                                       robust_gp.indirect_uncertain_vars, i, robust_gp.setting)

        if not robust_gp.large_posynomials:
            robust_gp.setting.set('allowedNumOfPerms', 0)

        if equality_constraints:
            warnings.warn('equality constraints will not be robustified')
        return robust_gp

    @classmethod
    def from_robust_gp_model(cls, robust_gp):
        new_robust_gp = cls()

        new_robust_gp.model = robust_gp.model
        new_robust_gp.substitutions = robust_gp.substitutions
        new_robust_gp.type_of_uncertainty_set = robust_gp.type_of_uncertainty_set
        new_robust_gp.ready_constraints = robust_gp.ready_constraints
        new_robust_gp.tractable_posynomials = robust_gp.tractable_posynomials
        new_robust_gp.to_linearize_posynomials = robust_gp.to_linearize_posynomials
        new_robust_gp.large_posynomials = robust_gp.large_posynomials
        new_robust_gp.uncertain_vars, new_robust_gp.indirect_uncertain_vars = robust_gp.uncertain_vars, robust_gp.indirect_uncertain_vars
        new_robust_gp.slopes = robust_gp.slopes
        new_robust_gp.intercepts = robust_gp.intercepts
        new_robust_gp.lower_approximation_used = robust_gp.lower_approximation_used
        new_robust_gp.r = robust_gp.r
        new_robust_gp.initial_guess = robust_gp.initial_guess
        new_robust_gp.robust_model = robust_gp.robust_model
        new_robust_gp.setup_time = robust_gp.setup_time
        new_robust_gp.setting = robust_gp.setting

        return new_robust_gp

    #    @classmethod
    #    def from_constraints(cls, type_of_uncertainty_set, ready_constraints, tractable_posynomials,
    #                         to_linearize_posynomials, large_posynomials, unclassified_constraints, uncertain_vars,
    #                         indirect_uncertain_vars, slopes, intercepts, *substitutions, **options):
    #        robust_gp = cls()
    #        if substitutions:
    #            robust_gp.substitutions = robust_gp.substitutions

    #        robust_gp.type_of_uncertainty_set = type_of_uncertainty_set
    #        robust_gp.ready_constraints = ready_constraints
    #        robust_gp.tractable_posynomials = tractable_posynomials
    #        robust_gp.to_linearize_posynomials = to_linearize_posynomials
    #        robust_gp.large_posynomials = large_posynomials
    #        robust_gp.uncertain_vars, robust_gp.indirect_uncertain_vars = uncertain_vars, indirect_uncertain_vars
    #        robust_gp.slopes = slopes
    #        robust_gp.intercepts = intercepts
    #        robust_gp.setting = RobustGPSetting(**options)

    @staticmethod
    def uncertain_variables_exponents(data_monomials, uncertain_vars, indirect_uncertain_vars):
        """
        gets the exponents of uncertain variables
        :param data_monomials:  the uncertain posynomials
        :param uncertain_vars: the uncertain variables of the model
        :param indirect_uncertain_vars: the indirectly uncertain variables of the model
        :return: the 2 dimensional array of exponents(matrix)
        """
        direct_uncertain_vars_data_monomials = []
        for monomial in data_monomials:
            new_monomial = RobustGPTools. \
                only_uncertain_vars_monomial(monomial.exps[0], monomial.cs[0], indirect_uncertain_vars)
            direct_uncertain_vars_data_monomials.append(new_monomial)

        exps_of_uncertain_vars = \
            np.array([[-p.exps[0].get(var.key, 0) for var in uncertain_vars]
                      for p in direct_uncertain_vars_data_monomials])
        return exps_of_uncertain_vars

    @staticmethod
    def normalize_perturbation_vector(uncertain_vars, gamma):
        """
        normalizes the perturbation elements
        :param uncertain_vars: the uncertain variables of the model
        :param gamma: the size of the uncertainty set
        :return: the centering and scaling vector
        """
        prs = np.array([var.key.pr * gamma for var in uncertain_vars])
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

            exps_of_uncertain_vars = RobustGPModel. \
                uncertain_variables_exponents(monomials, self.uncertain_vars, self.indirect_uncertain_vars)
            centering_vector, scaling_vector = RobustGPModel. \
                normalize_perturbation_vector(self.uncertain_vars, self.setting.get('gamma'))
            # noinspection PyTypeChecker
            coefficient = RobustGPModel. \
                construct_robust_monomial_coefficients(exps_of_uncertain_vars, self.setting.get('gamma'),
                                                       self.type_of_uncertainty_set,
                                                       centering_vector, scaling_vector)

            subs_monomials = []
            for j in xrange(len(monomials)):
                monomials[j] *= coefficient[j][0]
                monomials[j] = monomials[j].sub(solution)
                monomials[j] = monomials[j].sub(self.model.substitutions)
                subs_monomials.append(monomials[j].cs[0])

            values.append(max(subs_monomials))

        if number_of_iterations % 2 != 0:
            the_monomial = Monomial(two_term_approximation.p.exps[permutation[len(permutation) - 1]],
                                    two_term_approximation.p.cs[permutation[len(permutation) - 1]])

            exps_of_uncertain_vars = RobustGPModel. \
                uncertain_variables_exponents([the_monomial], self.uncertain_vars, self.indirect_uncertain_vars)
            centering_vector, scaling_vector = RobustGPModel. \
                normalize_perturbation_vector(self.uncertain_vars, self.setting.get('gamma'))
            # noinspection PyTypeChecker
            coefficient = RobustGPModel. \
                construct_robust_monomial_coefficients(exps_of_uncertain_vars, self.setting.get('gamma'),
                                                       self.type_of_uncertainty_set,
                                                       centering_vector, scaling_vector)
            the_monomial *= coefficient[0][0]
            the_monomial = the_monomial.sub(solution)
            the_monomial = the_monomial.sub(self.model.substitutions)
            values.append(the_monomial.cs[0])

        return sum(values)

    def find_permutation_with_minimum_value(self, two_term_approximation, solution):
        minimum_value = np.inf
        minimum_index = len(two_term_approximation.list_of_permutations)
        for i in xrange(len(two_term_approximation.list_of_permutations)):
            temp_value = self. \
                calculate_value_of_two_term_approximated_posynomial(two_term_approximation, i, solution)
            # print temp_value
            if temp_value < minimum_value:
                minimum_value = temp_value
                minimum_index = i

        return minimum_index

    def setup(self, **options):
        for option, key in options.iteritems():
            self.setting.set(option, key)
        start_time = time.time()

        solution = None

        two_term_data_posynomials = []
        number_of_trials_until_feasibility_is_attained = 0
        while number_of_trials_until_feasibility_is_attained <= min(10, self.setting.get('allowedNumOfPerms')):
            for i, two_term_approximation in enumerate(self.large_posynomials):
                perm_index = np.random.choice(range(0, len(two_term_approximation.list_of_permutations)))
                permutation = two_term_approximation.list_of_permutations[perm_index]
                no_data, data = TwoTermApproximation. \
                    two_term_equivalent_posynomial(two_term_approximation.p, i, permutation, False)
                self.ready_constraints += no_data
                two_term_data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

            two_term_data_posynomials += self.to_linearize_posynomials

            self.r, solution, self.robust_model = self.find_number_of_piece_wise_linearization(
                two_term_data_posynomials)

            if self.r != 0 or self.lower_approximation_used:
                break
            number_of_trials_until_feasibility_is_attained += 1

        if number_of_trials_until_feasibility_is_attained > min(10, self.setting.get('allowedNumOfPerms')):
            if not self.lower_approximation_used:
                raise Exception('Not Feasible')

        self.slopes, self.intercepts, _, _, _ = LinearizeTwoTermPosynomials. \
            two_term_posynomial_linearization_coeff(self.r, self.setting.get('linearizationTolerance'))

        old_solution = solution
        for _ in xrange(self.setting.get('allowedNumOfPerms')):
            permutation_indices = self.new_permutation_indices(solution)
            two_term_data_posynomials = []

            for i, two_term_approximation in enumerate(self.large_posynomials):
                permutation = two_term_approximation.list_of_permutations[permutation_indices[i]]
                _, data = TwoTermApproximation.two_term_equivalent_posynomial(two_term_approximation.p, i,
                                                                              permutation, False)
                two_term_data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

            two_term_data_posynomials += self.to_linearize_posynomials
            self.robust_model, _ = self.linearize_and_return_upper_lower_models(two_term_data_posynomials, self.r)

            new_solution = RobustGPModel.internal_solve(self.robust_model, self.initial_guess)

            same_solution = RobustGPModel.same_solution(old_solution, new_solution)
            if same_solution:
                break
            else:
                old_solution = new_solution
        self.setup_time = time.time() - start_time

    def linearize_and_return_upper_lower_models(self, two_term_data_posynomials, r):
        no_data_upper_constraints = []
        no_data_lower_constraints = []
        data_constraints = []
        data_posynomials = []

        for i, two_term_p in enumerate(two_term_data_posynomials):
            linearize_p = LinearizeTwoTermPosynomials(two_term_p)
            no_data_upper, no_data_lower, data = linearize_p. \
                linearize_two_term_posynomial(i, r, self.setting.get('linearizationTolerance'))
            no_data_upper_constraints += no_data_upper
            no_data_lower_constraints += no_data_lower
            data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

        all_tractable_posynomials = self.tractable_posynomials + data_posynomials
        exps_of_uncertain_vars = RobustGPModel. \
            uncertain_variables_exponents(all_tractable_posynomials, self.uncertain_vars, self.indirect_uncertain_vars)
        if exps_of_uncertain_vars.size > 0:
            centering_vector, scaling_vector = RobustGPModel. \
                normalize_perturbation_vector(self.uncertain_vars, self.setting.get('gamma'))
            # noinspection PyTypeChecker
            coefficient = RobustGPModel. \
                construct_robust_monomial_coefficients(exps_of_uncertain_vars, self.setting.get('gamma'),
                                                       self.type_of_uncertainty_set, centering_vector, scaling_vector)
            for i in xrange(len(all_tractable_posynomials)):
                data_constraints += [coefficient[i][0] * all_tractable_posynomials[i] <= 1]

        model_upper = Model(self.model.cost, [no_data_upper_constraints, self.ready_constraints, data_constraints])
        model_upper.substitutions.update(self.substitutions)

        model_lower = Model(self.model.cost, [no_data_lower_constraints, self.ready_constraints, data_constraints])
        model_lower.substitutions.update(self.substitutions)

        return model_upper, model_lower

    def find_number_of_piece_wise_linearization(self, two_term_data_posynomials):
        error = 2 * self.setting.get('linearizationTolerance')
        r = self.setting.get('minNumOfLinearSections')

        sol_upper = None
        sol_lower = None

        model_upper = None
        model_lower = None

        while r <= self.setting.get('maxNumOfLinearSections') and error > self.setting.get('linearizationTolerance'):

            model_upper, model_lower = self.linearize_and_return_upper_lower_models(two_term_data_posynomials, r)

            upper_model_infeasible = 0
            try:
                sol_upper = RobustGPModel.internal_solve(model_upper, self.initial_guess)
            except:
                upper_model_infeasible = 1
            try:
                sol_lower = RobustGPModel.internal_solve(model_lower, self.initial_guess)
            except:
                return 0, None, None

            if upper_model_infeasible != 1:
                try:
                    error = (sol_upper.get('cost').m -
                             sol_lower.get('cost').m) / (0.0 + sol_lower.get('cost').m)
                except:
                    error = (sol_upper.get('cost') -
                             sol_lower.get('cost')) / (0.0 + sol_lower.get('cost'))
            elif r == self.setting.get('maxNumOfLinearSections'):
                self.lower_approximation_used = True

            r += 1

        if not self.lower_approximation_used:
            solution = sol_upper
            robust_model = model_upper
        else:
            solution = sol_lower
            robust_model = model_lower

        return r - 1, solution, robust_model

    def solve(self, verbosity=1, **options):
        if self.robust_model is None:
            self.setup(**options)
        if self.initial_guess is None:
            initial_guess = {}
        else:
            initial_guess = self.initial_guess
        if self.lower_approximation_used:
            warnings.warn("the model is infeasible, the lower piece-wise approximation is used")
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
            relative_difference = np.abs((solution_one['variables'][key] -
                                          solution_two['variables'][key]) / solution_one['variables'][key])
            try:
                if relative_difference > 0.0001:
                    return False
            except:
                if all(i > 0.0001 for i in relative_difference):
                    return False
        return True

    def generate_initial_guess(self, robust_model):
        robust_model_initial_guess = RobustGPModel. \
            construct(robust_model.model, robust_model.type_of_uncertainty_set, simpleModel=True)
        robust_model_initial_guess.setup()
        solution = robust_model_initial_guess.solve(verbosity=0)
        self.initial_guess = solution['variables']
        return

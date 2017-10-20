from gpkit import Model, Monomial
from gpkit.nomials import SignomialInequality
import numpy as np
from time import time
import warnings

from RobustGP import RobustGPModel, RobustGPSetting
from RobustGPTools import RobustGPTools
from EquivalentPosynomials import EquivalentPosynomials
from TwoTermApproximation import TwoTermApproximation
from RobustifyLargePosynomial import RobustifyLargePosynomial
from LinearizeTwoTermPosynomials import LinearizeTwoTermPosynomials


class RobustSPSetting(RobustGPSetting):
    def __init__(self, **options):
        if 'SPRelativeTolerance' in options:
            RobustGPSetting.__init__(self, **options)
        else:
            RobustGPSetting.__init__(self, SPRelativeTolerance=1e-4, **options)


class RobustSPModel:  # 4.53

    def __init__(self, model, type_of_uncertainty_set, **options):
        self.original_model = model
        self.substitutions = model.substitutions
        self.type_of_uncertainty_set = type_of_uncertainty_set
        self.setting = RobustSPSetting(**options)
        # self.options = options
        nominal_solve = model.localsolve(verbosity=0)
        self.nominal_solution = nominal_solve.get('variables')
        self.nominal_cost = nominal_solve['cost']
        self.sequence_of_rgps = []
        self.number_of_rgp_approximations = None
        self.r = None
        self.solve_time = None
        self.uncertain_vars, self.indirect_uncertain_vars = RobustGPTools.uncertain_model_variables(model)
        self.slopes = None
        self.intercepts = None
        self.lower_approximation_used = False

        if self.setting.get('twoTerm'):
            self.setting.set('linearizeTwoTerm', True)
            self.setting.set('enableSP', False)

        if self.setting.get('simpleModel'):
            self.setting.set('allowedNumOfPerms', 1)

        if self.type_of_uncertainty_set == 'box':
            self.dependent_uncertainty_set = False
        else:
            self.dependent_uncertainty_set = True

        all_constraints = model.flat(constraintsets=False)

        gp_posynomials = []
        self.sp_constraints = []

        for cs in all_constraints:
            if isinstance(cs, SignomialInequality):
                self.sp_constraints.append(cs)
            else:
                gp_posynomials.append(cs.as_posyslt1())

        self.offset = len(gp_posynomials)

        self.ready_gp_constraints, self.tractable_gp_posynomials, self.to_linearize_gp_posynomials, self.large_gp_posynomials = RobustSPModel. \
            classify_gp_constraints(gp_posynomials, self.type_of_uncertainty_set, self.uncertain_vars,
                                    self.indirect_uncertain_vars, self.setting, self.dependent_uncertainty_set)

    def localsolve(self, verbosity=1, **options):
        for option, key in options.iteritems():
            self.setting.set(option, key)
        start_time = time()

        solution = None
        robust_model = None

        two_term_data_posynomials = []
        number_of_trials_until_feasibility_is_attained = 0
        ready_sp_constraints, tractable_sp_posynomials, to_linearize_sp_posynomials, large_sp_posynomials = self.\
            approximate_and_classify_sp_constraints(self.sp_constraints, self.nominal_solution)

        ready_constraints = self.ready_gp_constraints + ready_sp_constraints
        tractable_posynomials = self.tractable_gp_posynomials + tractable_sp_posynomials
        to_linearize_posynomials = self.to_linearize_gp_posynomials + to_linearize_sp_posynomials
        large_posynomials = self.large_gp_posynomials + large_sp_posynomials

        while number_of_trials_until_feasibility_is_attained <= min(10, self.setting.get('allowedNumOfPerms')):
            for i, two_term_approximation in enumerate(large_posynomials):
                perm_index = np.random.choice(range(0, len(two_term_approximation.list_of_permutations)))
                permutation = two_term_approximation.list_of_permutations[perm_index]
                # print (permutation)
                no_data, data = TwoTermApproximation. \
                    two_term_equivalent_posynomial(two_term_approximation.p, i, permutation, False)
                ready_constraints += no_data
                two_term_data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

            two_term_data_posynomials += to_linearize_posynomials

            self.r, solution, robust_model = self.find_number_of_piece_wise_linearization(
                two_term_data_posynomials, ready_constraints, tractable_posynomials)

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
            ready_sp_constraints, tractable_sp_posynomials, to_linearize_sp_posynomials, large_sp_posynomials = self.\
                approximate_and_classify_sp_constraints(self.sp_constraints, old_solution['variables'])

            ready_constraints = self.ready_gp_constraints + ready_sp_constraints
            tractable_posynomials = self.tractable_gp_posynomials + tractable_sp_posynomials
            to_linearize_posynomials = self.to_linearize_gp_posynomials + to_linearize_sp_posynomials
            large_posynomials = self.large_gp_posynomials + large_sp_posynomials

            permutation_indices = self.new_permutation_indices(old_solution, large_posynomials)
            two_term_data_posynomials = []

            for i, two_term_approximation in enumerate(large_posynomials):
                permutation = two_term_approximation.list_of_permutations[permutation_indices[i]]
                # print ("used perm", permutation)
                # print ("exps", two_term_approximation.p.exps)
                no_data, data = TwoTermApproximation.two_term_equivalent_posynomial(two_term_approximation.p, i, permutation, False)
                ready_constraints += no_data
                two_term_data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

            two_term_data_posynomials += to_linearize_posynomials
            robust_model, _ = self.linearize_and_return_upper_lower_models(two_term_data_posynomials, self.r, ready_constraints, tractable_posynomials)

            new_solution = RobustGPModel.internal_solve(robust_model, None)

            same_solution = RobustGPModel.same_solution(old_solution, new_solution)
            if same_solution:
                break
            else:
                old_solution = new_solution
        self.solve_time = time() - start_time
        return robust_model.solve(verbosity=verbosity)

    def approximate_and_classify_sp_constraints(self, sp_constraints, solution):
        sp_gp_approximation = [cs.as_approxsgt(solution) for cs in sp_constraints]
        return RobustSPModel. \
            classify_gp_constraints(sp_gp_approximation, self.type_of_uncertainty_set, self.uncertain_vars,
                                    self.indirect_uncertain_vars, self.setting, self.dependent_uncertainty_set)

    @staticmethod
    def classify_gp_constraints(gp_posynomials, type_of_uncertainty_set, uncertain_vars, indirect_uncertain_vars,
                                setting, dependent_uncertainty_set):
        data_gp_posynomials = []
        ready_gp_constraints = []
        for i, p in enumerate(gp_posynomials):
            equivalent_p = EquivalentPosynomials(p, uncertain_vars, indirect_uncertain_vars, i,
                                                 setting.get('simpleModel'), dependent_uncertainty_set)
            no_data, data = equivalent_p.no_data_constraints, equivalent_p.data_constraints

            data_gp_posynomials += data
            ready_gp_constraints += [posy <= 1 for posy in no_data]

        equality_constraints = False
        tractable_gp_posynomials = []
        to_linearize_gp_posynomials = []
        large_gp_posynomials = []
        for i, p in enumerate(data_gp_posynomials):
            if len(p.exps) == 1:
                if 1 / p in data_gp_posynomials:
                    equality_constraints = True
                    ready_gp_constraints += [p <= 1]
                else:
                    tractable_gp_posynomials += [p]
            elif len(p.exps) == 2 and setting.get('linearizeTwoTerm'):
                to_linearize_gp_posynomials += [p]
            else:
                if setting.get('twoTerm'):
                    two_term_approximation = TwoTermApproximation(p, uncertain_vars,
                                                                  indirect_uncertain_vars,
                                                                  setting)
                    large_gp_posynomials.append(two_term_approximation)
                else:
                    robust_large_p = RobustifyLargePosynomial(p)
                    ready_gp_constraints += robust_large_p. \
                        robustify_large_posynomial(type_of_uncertainty_set, uncertain_vars,
                                                   indirect_uncertain_vars, i, setting)

        if equality_constraints:
            warnings.warn('equality constraints will not be robustified')

        return ready_gp_constraints, tractable_gp_posynomials, to_linearize_gp_posynomials, large_gp_posynomials

    def robustify_monomial(self, monomial):
        new_monomial = RobustGPTools. \
            only_uncertain_vars_monomial(monomial.exps[0], monomial.cs[0], self.indirect_uncertain_vars)
        m_direct_uncertain_vars = [var for var in new_monomial.varkeys if var in self.uncertain_vars]
        # m_indirect_uncertain_vars = [var for var in new_monomial.varkeys if var in self.indirect_uncertain_vars]
        total_center = 0
        norm = 0
        for var in m_direct_uncertain_vars:
            pr = var.key.pr * self.setting.get('gamma')
            eta_max = np.log(1 + pr / 100.0)
            eta_min = np.log(1 - pr / 100.0)
            center = (eta_min + eta_max) / 2.0
            scale = eta_max - center
            exponent = -new_monomial.exps[0].get(var.key)
            pert = exponent * scale

            if self.type_of_uncertainty_set == 'box':
                norm += np.abs(pert)
            elif self.type_of_uncertainty_set == 'elliptical':
                norm += pert ** 2
            elif self.type_of_uncertainty_set == 'one norm':
                norm = max(norm, np.abs(pert))
            else:
                raise Exception('This type of set is not supported')
            total_center = total_center + exponent * center
        if self.type_of_uncertainty_set == 'elliptical':
            norm = np.sqrt(norm)
        return monomial * np.exp(self.setting.get('gamma') * norm) / np.exp(total_center)

    def robustify_set_of_monomials(self, set_of_monomials):
        robust_set_of_monomial_constraints = []
        for monomial in set_of_monomials:
            robust_set_of_monomial_constraints += [self.robustify_monomial(monomial) <= 1]
        return robust_set_of_monomial_constraints

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

            subs_monomials = []
            for j in xrange(len(monomials)):
                monomials[j] = self.robustify_monomial(monomials[j])
                monomials[j] = monomials[j].sub(self.original_model.substitutions)
                monomials[j] = monomials[j].sub(solution['variables'])
                subs_monomials.append(monomials[j].cs[0])

            values.append(max(subs_monomials))

        if number_of_iterations % 2 != 0:
            the_monomial = Monomial(two_term_approximation.p.exps[permutation[len(permutation) - 1]],
                                    two_term_approximation.p.cs[permutation[len(permutation) - 1]])

            the_monomial = self.robustify_monomial(the_monomial)
            the_monomial = the_monomial.sub(self.original_model.substitutions)
            the_monomial = the_monomial.sub(solution['variables'])
            values.append(the_monomial.cs[0])

        return sum(values)

    def find_permutation_with_minimum_value(self, two_term_approximation, solution):
        minimum_value = np.inf
        minimum_index = len(two_term_approximation.list_of_permutations)
        for i in xrange(len(two_term_approximation.list_of_permutations)):
            temp_value = self. \
                calculate_value_of_two_term_approximated_posynomial(two_term_approximation, i, solution)
            # print "value in find", temp_value
            # print "the corresponding perm", two_term_approximation.list_of_permutations[i]
            if temp_value < minimum_value:
                minimum_value = temp_value
                minimum_index = i

        return minimum_index

    def linearize_and_return_upper_lower_models(self, two_term_data_posynomials, r, ready_constraints, tractable_posynomials):
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

        all_tractable_posynomials = tractable_posynomials + data_posynomials

        data_constraints += self.robustify_set_of_monomials(all_tractable_posynomials)

        model_upper = Model(self.original_model.cost, [no_data_upper_constraints, ready_constraints, data_constraints])
        model_upper.substitutions.update(self.substitutions)

        model_lower = Model(self.original_model.cost, [no_data_lower_constraints, ready_constraints, data_constraints])
        model_lower.substitutions.update(self.substitutions)

        return model_upper, model_lower

    def find_number_of_piece_wise_linearization(self, two_term_data_posynomials, ready_constraints, tractable_posynomials):
        error = 2 * self.setting.get('linearizationTolerance')
        r = self.setting.get('minNumOfLinearSections')

        sol_upper = None
        sol_lower = None

        model_upper = None
        model_lower = None

        while r <= self.setting.get('maxNumOfLinearSections') and error > self.setting.get('linearizationTolerance'):

            model_upper, model_lower = self.linearize_and_return_upper_lower_models(two_term_data_posynomials, r, ready_constraints, tractable_posynomials)

            upper_model_infeasible = 0
            try:
                sol_upper = RobustGPModel.internal_solve(model_upper, None)
            except:
                upper_model_infeasible = 1
            try:
                sol_lower = RobustGPModel.internal_solve(model_lower, None)
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

    @staticmethod
    def internal_solve(model, initial_guess):
        if initial_guess is None:
            initial_guess = {}
        try:
            sol = model.solve(verbosity=0)
        except:
            sol = model.localsolve(verbosity=0, x0=initial_guess)
        return sol

    def new_permutation_indices(self, solution, large_posynomials):
        permutation_indices = []
        for two_term_approximation in large_posynomials:
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

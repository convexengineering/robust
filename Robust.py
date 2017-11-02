from gpkit import Model, Monomial, Variable
from gpkit.nomials import SignomialInequality
import numpy as np
from time import time
import warnings

from RobustGPTools import RobustGPTools
from EquivalentPosynomials import EquivalentPosynomials
from EquivalentModels import TwoTermBoydModel
from TwoTermApproximation import TwoTermApproximation
from RobustifyLargePosynomial import RobustifyLargePosynomial
from LinearizeTwoTermPosynomials import LinearizeTwoTermPosynomials


class RobustnessSetting:
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
            'iterationsRelativeTolerance': 1e-4,
            'iterationLimit': 10
        }
        for key, value in options.iteritems():
            self._options[key] = value

        if self._options['twoTerm']:
            self._options['linearizeTwoTerm'] = True
            self._options['enableSP'] = False

        if self._options['simpleModel']:
            self._options['allowedNumOfPerms'] = 1

    def get(self, option_name):
        return self._options[option_name]

    def set(self, option_name, value):
        self._options[option_name] = value


class RobustModel:
    def __init__(self, model, type_of_uncertainty_set, **options):
        self.nominal_model = model
        self.substitutions = model.substitutions
        self.type_of_uncertainty_set = type_of_uncertainty_set

        self.setting = RobustnessSetting(**options)
        slopes_intercepts = LinearizeTwoTermPosynomials. \
            two_term_posynomial_linearization_coeff(self.setting.get('minNumOfLinearSections'))
        self.robust_solve_properties = {'setuptime': 0,
                                        'numoflinearsections': self.setting.get('minNumOfLinearSections'),
                                        'slopes': slopes_intercepts[0],
                                        'intercepts': slopes_intercepts[1]
                                        }
        self.nominal_solve = RobustModel.nominalsolve(model, verbosity=0)
        self.nominal_solution = self.nominal_solve.get('variables')
        self.nominal_cost = self.nominal_solve['cost']

        self._sequence_of_rgps = []
        self._robust_model = None

        self.uncertain_vars, self.indirect_uncertain_vars = RobustGPTools.uncertain_model_variables(model)
        self.lower_approximation_is_feasible = False

        if self.type_of_uncertainty_set == 'box':
            self.dependent_uncertainty_set = False
        else:
            self.dependent_uncertainty_set = True

        self.ready_gp_constraints = []
        self.tractable_gp_posynomials = []
        self.to_linearize_gp_posynomials = []
        self.large_gp_posynomials = []
        self.sp_constraints = []

        if self.setting.get('boyd'):
            self.setting.set('iterationLimit', 1)
            try:
                safe_model = TwoTermBoydModel(model)
            except:
                raise Exception("boyd's formulation is not supported for sp models")
            safe_model_posynomials = safe_model.as_posyslt1()
            equality_constraints = False
            for p in safe_model_posynomials:
                if len(p.exps) == 1:
                    if 1 / p in safe_model_posynomials:
                        equality_constraints = True
                        self.ready_gp_constraints += [p <= 1]
                    else:
                        self.tractable_gp_posynomials += [p]
                else:
                    self.to_linearize_gp_posynomials += [p]
            if equality_constraints:
                warnings.warn('equality constraints will not be robustified')
            self.number_of_gp_posynomials = 0
            return

        all_constraints = model.flat(constraintsets=False)

        gp_posynomials = []

        for cs in all_constraints:
            if isinstance(cs, SignomialInequality):
                self.sp_constraints.append(cs)
            else:
                gp_posynomials += cs.as_posyslt1()

        self.number_of_gp_posynomials = len(gp_posynomials)

        constraints_posynomials_tuple = self.classify_gp_constraints(gp_posynomials)

        self.ready_gp_constraints = constraints_posynomials_tuple[0]
        self.tractable_gp_posynomials = constraints_posynomials_tuple[1]
        self.to_linearize_gp_posynomials = constraints_posynomials_tuple[2]
        self.large_gp_posynomials = constraints_posynomials_tuple[3]

    def setup(self, **options):
        for option, key in options.iteritems():
            self.setting.set(option, key)

        start_time = time()

        old_solution = self.nominal_solve
        reached_feasibility = 0

        for _ in xrange(self.setting.get('iterationLimit')):
            ready_sp_constraints, tractable_sp_posynomials, to_linearize_sp_posynomials, large_sp_posynomials = self. \
                approximate_and_classify_sp_constraints(old_solution, self.number_of_gp_posynomials)

            ready_constraints = self.ready_gp_constraints + ready_sp_constraints
            tractable_posynomials = self.tractable_gp_posynomials + tractable_sp_posynomials
            to_linearize_posynomials = self.to_linearize_gp_posynomials + to_linearize_sp_posynomials
            large_posynomials = self.large_gp_posynomials + large_sp_posynomials

            permutation_indices = self.new_permutation_indices(old_solution, large_posynomials)

            two_term_data_posynomials = []

            for i, two_term_approximation in enumerate(large_posynomials):
                permutation = two_term_approximation.list_of_permutations[permutation_indices[i]]
                no_data, data = TwoTermApproximation. \
                    two_term_equivalent_posynomial(two_term_approximation.p, i, permutation, False)
                ready_constraints += no_data
                two_term_data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

            two_term_data_posynomials += to_linearize_posynomials

            try:
                if not reached_feasibility:
                    self.robust_solve_properties['numoflinearsections'], new_solution, self._robust_model = self. \
                        find_number_of_piece_wise_linearization(two_term_data_posynomials, ready_constraints,
                                                                tractable_posynomials)
                else:
                    self._robust_model, _ = self. \
                        linearize_and_return_upper_lower_models(two_term_data_posynomials,
                                                                self.robust_solve_properties['numoflinearsections'],
                                                                ready_constraints, tractable_posynomials)
                    new_solution = RobustModel.nominalsolve(self._robust_model, verbosity=0)
                reached_feasibility += 1
                rel_tol = np.abs((new_solution['cost'] - old_solution['cost']) / old_solution['cost'])
            except:
                if not reached_feasibility:
                    self.robust_solve_properties['numoflinearsections'], new_solution, self._robust_model = self. \
                        find_number_of_piece_wise_linearization(two_term_data_posynomials, ready_constraints,
                                                                tractable_posynomials, feasible=True)
                else:
                    self._robust_model, _ = self. \
                        linearize_and_return_upper_lower_models(two_term_data_posynomials,
                                                                self.robust_solve_properties['numoflinearsections'],
                                                                ready_constraints, tractable_posynomials, feasible=True)
                    new_solution = RobustModel.nominalsolve(self._robust_model, verbosity=0)
                rel_tol = 2 * self.setting.get('iterationsRelativeTolerance')

            if reached_feasibility <= 1:
                self.robust_solve_properties['slopes'], self.robust_solve_properties['intercepts'], \
                _, _, _ = LinearizeTwoTermPosynomials.two_term_posynomial_linearization_coeff(
                    self.robust_solve_properties['numoflinearsections'])

            self._sequence_of_rgps.append(self._robust_model)

            if rel_tol <= self.setting.get('iterationsRelativeTolerance'):
                break
            else:
                old_solution = new_solution

        if reached_feasibility < 1:
            raise Exception("feasibility is not reached after %s iterations. If the solution seems to converge, try "
                            "increasing the iterationLimit. Increasing the allowed number of permutations might also "
                            "help" % self.setting.get('iterationLimit'))
        self.robust_solve_properties['setuptime'] = time() - start_time

    def robustsolve(self, verbosity=1, **options):
        if self._robust_model is None:
            self.setup(**options)
        try:
            sol = self._robust_model.solve(verbosity=verbosity)
        except:
            sol = self._robust_model.localsolve(verbosity=verbosity)
        if verbosity > 0:
            print ("solving needed %s iterations." % len(self._sequence_of_rgps))
            print ("setting up took %s seconds." % self.robust_solve_properties['setuptime'])
        sol.update(self.robust_solve_properties)
        return sol

    def approximate_and_classify_sp_constraints(self, solution, number_of_gp_posynomials):
        sp_gp_approximation = [cs.as_gpconstr(x0=solution["freevariables"], substitutions=solution["constants"]).
                                   as_posyslt1()[0] for cs in self.sp_constraints]
        return self.classify_gp_constraints(sp_gp_approximation, number_of_gp_posynomials)

    def classify_gp_constraints(self, gp_posynomials, offset=0):
        data_gp_posynomials = []
        ready_gp_constraints = []
        for i, p in enumerate(gp_posynomials):
            equivalent_p = EquivalentPosynomials(p, self.uncertain_vars, self.indirect_uncertain_vars,
                                                 i + offset, self.setting.get('simpleModel'),
                                                 self.dependent_uncertainty_set)
            no_data, data = equivalent_p.no_data_constraints, equivalent_p.data_constraints
            data_gp_posynomials += [posy.as_posyslt1()[0] for posy in data]
            ready_gp_constraints += no_data

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
            elif len(p.exps) == 2 and self.setting.get('linearizeTwoTerm'):
                to_linearize_gp_posynomials += [p]
            else:
                if self.setting.get('twoTerm'):
                    two_term_approximation = TwoTermApproximation(p, self.uncertain_vars, self.indirect_uncertain_vars,
                                                                  self.setting)
                    large_gp_posynomials.append(two_term_approximation)
                else:
                    robust_large_p = RobustifyLargePosynomial(p)
                    ready_gp_constraints += robust_large_p. \
                        robustify_large_posynomial(self.type_of_uncertainty_set, self.uncertain_vars,
                                                   self.indirect_uncertain_vars, i + offset, self.setting)

        if equality_constraints:
            warnings.warn('equality constraints will not be robustified')

        return ready_gp_constraints, tractable_gp_posynomials, to_linearize_gp_posynomials, large_gp_posynomials

    def robustify_monomial(self, monomial):

        new_monomial_exps = RobustGPTools. \
            only_uncertain_vars_monomial(monomial.exps[0], monomial.cs[0], self.indirect_uncertain_vars)

        m_direct_uncertain_vars = [var for var in new_monomial_exps.keys() if var in self.uncertain_vars]
        total_center = 0
        norm = 0
        for var in m_direct_uncertain_vars:
            pr = var.key.pr * self.setting.get('gamma')
            eta_max = np.log(1 + pr / 100.0)
            eta_min = np.log(1 - pr / 100.0)
            center = (eta_min + eta_max) / 2.0
            scale = eta_max - center
            exponent = -new_monomial_exps.get(var.key)
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

    def robustify_set_of_monomials(self, set_of_monomials, feasible=False):
        robust_set_of_monomial_constraints = []
        slackvar = Variable()
        for monomial in set_of_monomials:
            robust_set_of_monomial_constraints += [self.robustify_monomial(monomial) <= slackvar ** feasible]
        robust_set_of_monomial_constraints += [slackvar >= 1, slackvar <= 1000]
        return robust_set_of_monomial_constraints, slackvar

    def calculate_value_of_two_term_approximated_posynomial(self, two_term_approximation, index_of_permutation,
                                                            solution):
        permutation = two_term_approximation.list_of_permutations[index_of_permutation]

        number_of_iterations = int(len(permutation) / 2)
        num_of_linear_sections = self.robust_solve_properties['numoflinearsections']
        slopes = self.robust_solve_properties['slopes']
        intercepts = self.robust_solve_properties['intercepts']
        values = []

        for i in xrange(number_of_iterations):
            monomials = []

            first_monomial = Monomial(two_term_approximation.p.exps[permutation[2 * i]],
                                      two_term_approximation.p.cs[permutation[2 * i]])
            second_monomial = Monomial(two_term_approximation.p.exps[permutation[2 * i + 1]],
                                       two_term_approximation.p.cs[permutation[2 * i + 1]])

            monomials += [first_monomial]
            for j in xrange(num_of_linear_sections - 2):
                monomials += [first_monomial ** slopes[num_of_linear_sections - 3 - j] *
                              second_monomial ** slopes[j] * np.exp(intercepts[j])]
            monomials += [second_monomial]

            subs_monomials = []
            for j in xrange(len(monomials)):
                monomials[j] = self.robustify_monomial(monomials[j])
                monomials[j] = monomials[j].sub(self.substitutions)
                monomials[j] = monomials[j].sub(solution['variables'])
                subs_monomials.append(monomials[j].cs[0])

            values.append(max(subs_monomials))

        if number_of_iterations % 2 != 0:
            the_monomial = Monomial(two_term_approximation.p.exps[permutation[len(permutation) - 1]],
                                    two_term_approximation.p.cs[permutation[len(permutation) - 1]])

            the_monomial = self.robustify_monomial(the_monomial)
            the_monomial = the_monomial.sub(self.substitutions)
            the_monomial = the_monomial.sub(solution['variables'])
            values.append(the_monomial.cs[0])

        return sum(values)

    def find_permutation_with_minimum_value(self, two_term_approximation, solution):
        minimum_value = np.inf
        minimum_index = len(two_term_approximation.list_of_permutations)
        for i in xrange(len(two_term_approximation.list_of_permutations)):
            temp_value = self. \
                calculate_value_of_two_term_approximated_posynomial(two_term_approximation, i, solution)
            if temp_value < minimum_value:
                minimum_value = temp_value
                minimum_index = i

        return minimum_index

    def linearize_and_return_upper_lower_models(self, two_term_data_posynomials, r, ready_constraints,
                                                tractable_posynomials, feasible=False):
        no_data_upper_constraints = []
        no_data_lower_constraints = []
        data_posynomials = []

        for i, two_term_p in enumerate(two_term_data_posynomials):
            linearize_p = LinearizeTwoTermPosynomials(two_term_p)
            no_data_upper, no_data_lower, data = linearize_p. \
                linearize_two_term_posynomial(i, r)
            no_data_upper_constraints += no_data_upper
            no_data_lower_constraints += no_data_lower
            data_posynomials += [constraint.as_posyslt1()[0] for constraint in data]

        all_tractable_posynomials = tractable_posynomials + data_posynomials

        data_constraints, slackvar = self.robustify_set_of_monomials(all_tractable_posynomials, feasible)
        upper_cons, lower_cons = [no_data_upper_constraints, ready_constraints, data_constraints], \
                                 [no_data_upper_constraints, ready_constraints, data_constraints]
        # st = time()
        model_upper = Model(self.nominal_model.cost * slackvar ** (100 * feasible), upper_cons)
        # [no_data_upper_constraints, ready_constraints, data_constraints])
        # print "mod up", time() - st
        # st = time()
        model_lower = Model(self.nominal_model.cost * slackvar ** (100 * feasible), lower_cons)
        # [no_data_lower_constraints, ready_constraints, data_constraints])
        # print "mod low", time() - st
        model_upper.substitutions.update(self.substitutions)
        model_lower.substitutions.update(self.substitutions)
        model_upper.unique_varkeys, model_lower.unique_varkeys = [self.nominal_model.varkeys] * 2
        model_upper.reset_varkeys()
        model_lower.reset_varkeys()

        return model_upper, model_lower

    def find_number_of_piece_wise_linearization(self, two_term_data_posynomials, ready_constraints,
                                                tractable_posynomials, feasible=False):
        error = 2 * self.setting.get('linearizationTolerance')
        r = self.setting.get('minNumOfLinearSections')

        sol_upper = None

        model_upper = None

        while r <= self.setting.get('maxNumOfLinearSections') and error > self.setting.get('linearizationTolerance'):
            model_upper, model_lower = self. \
                linearize_and_return_upper_lower_models(two_term_data_posynomials, r,
                                                        ready_constraints, tractable_posynomials, feasible)
            upper_model_infeasible = 0
            try:
                sol_upper = RobustModel.nominalsolve(model_upper, verbosity=0)
            except:
                upper_model_infeasible = 1
            try:
                sol_lower = RobustModel.nominalsolve(model_lower, verbosity=0)
            except:
                raise Exception("The model is infeasible")

            if upper_model_infeasible != 1:
                error = (sol_upper.get('cost') - sol_lower.get('cost')) / sol_lower.get('cost')
            elif r == self.setting.get('maxNumOfLinearSections'):
                self.lower_approximation_is_feasible = True
                raise Exception("The model is infeasible. The lower approximation of the model is feasible, try "
                                "increasing the maximum number of linear sections")
            r += 1
        return r - 1, sol_upper, model_upper

    def new_permutation_indices(self, solution, large_posynomials):
        permutation_indices = []
        for two_term_approximation in large_posynomials:
            permutation_indices.append(self.find_permutation_with_minimum_value(two_term_approximation, solution))
        return permutation_indices

    @staticmethod
    def nominalsolve(model, verbosity=0):
        try:
            return model.solve(verbosity=verbosity)
        except:
            return model.localsolve(verbosity=verbosity)

    def get_robust_model(self):
        if self.sp_constraints:
            return self._sequence_of_rgps
        else:
            return self._robust_model

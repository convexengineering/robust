from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import range
from builtins import object
from gpkit import Model, Variable, SignomialsEnabled
from gpkit.nomials import SignomialInequality, MonomialEquality
from gpkit.exceptions import InvalidGPConstraint
from gpkit.nomials import SingleSignomialEquality
import numpy as np
from time import time
import warnings
from scipy.stats import norm

from .robust_gp_tools import RobustGPTools
from .equivalent_posynomials import EquivalentPosynomials
from .equivalent_models import TwoTermBoydModel
from .twoterm_approximation import TwoTermApproximation
from .robustify_large_posynomial import RobustifyLargePosynomial
from .linearize_twoterm_posynomials import LinearizeTwoTermPosynomials


class RobustnessSetting(object):
    def __init__(self, **options):
        self._options = {
            'gamma': 1,
            'simpleModel': False,
            'numberOfRegressionPoints': 2,
            'numberOfRegressionPointsElliptical': 25,
            'linearizeTwoTerm': True,
            'enableSP': True,
            'boyd': False,
            'twoTerm': True,
            'simpleTwoTerm': False,
            'smartTwoTermChoose': False,
            'allowedNumOfPerms': 30,
            'linearizationTolerance': 0.001,
            'minNumOfLinearSections': 12,
            'maxNumOfLinearSections': 20,
            'iterationsRelativeTolerance': 1e-4,
            'iterationLimit': 10,
            'lognormal': True
        }
        for key, value in options.items():
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


class RobustModel(object):
    """
    RobustModel extends gpkit.Model through the robust counterpart.
    """
    def __init__(self, model, type_of_uncertainty_set, **options):
        self.nominal_model = model
        self.substitutions = model.substitutions
        self.type_of_uncertainty_set = type_of_uncertainty_set

        self.setting = RobustnessSetting(**options)
        slopes, intercepts, _, _, _ = LinearizeTwoTermPosynomials. \
            linearization_coeff(self.setting.get('minNumOfLinearSections'))
        self.robust_solve_properties = {'setuptime': 0,
                                        'numoflinearsections': self.setting.get('minNumOfLinearSections'),
                                        'slopes': slopes,
                                        'intercepts': intercepts
                                        }

        if 'nominalsolve' in options:
            self.nominal_solve = options['nominalsolve']
        else:
            self.nominal_solve = RobustModel.internalsolve(model, verbosity=0)
        self.nominal_solution = self.nominal_solve.get('variables')
        self.nominal_cost = self.nominal_solve['cost']

        self._sequence_of_rgps = []
        self._robust_model = None

        self.lower_approximation_is_feasible = False

        if self.type_of_uncertainty_set == 'box':
            self.dependent_uncertainty_set = False
        else:
            self.dependent_uncertainty_set = True
            if self.type_of_uncertainty_set == 'elliptical':
                self.setting.set('numberOfRegressionPoints', self.setting.get('numberOfRegressionPointsElliptical'))

        self.ready_gp_constraints = []
        self.to_linearize_gp_posynomials = [] # two-term posynomials
        self.large_gp_posynomials = [] # posynomials that need to be broken up
        self.sp_constraints = []
        self.sp_equality_constraints = []

        equality_constraints = False

        if self.setting.get('boyd'):
            self.setting.set('iterationLimit', 1)
            try:
                safe_model = TwoTermBoydModel(model)
            except InvalidGPConstraint:
                raise Exception("Boyd's formulation is not supported for SP models.")
            safe_model_constraints = safe_model.flat()
            del safe_model
            for cs in safe_model_constraints:
                if isinstance(cs, MonomialEquality):
                    self.ready_gp_constraints += [cs]
                    equality_constraints = True
                else:
                    p, = cs.unsubbed
                    if len(p.exps) == 1:
                        robust_monomial, _ = self.robustify_monomial(p)
                        self.ready_gp_constraints += [robust_monomial <= 1]
                    else:
                        self.to_linearize_gp_posynomials += [p]
            del safe_model_constraints

            if equality_constraints:
                warnings.warn('equality constraints will not be robustified')
            self.number_of_gp_posynomials = 0
            return

        all_constraints = model.flat()

        gp_posynomials = []

        # Classifying all constraints
        for cs in all_constraints:
            if isinstance(cs, SingleSignomialEquality):
                self.sp_equality_constraints.append(cs)
            elif isinstance(cs, SignomialInequality):
                self.sp_constraints.append(cs)
            elif isinstance(cs, MonomialEquality):
                self.ready_gp_constraints += [cs]
                equality_constraints = True

            else:
                gp_posynomials += cs.unsubbed

        self.number_of_gp_posynomials = len(gp_posynomials)

        # Classifying GP-compatible constraints.
        ready_gp_constraints, self.to_linearize_gp_posynomials, \
        self.large_gp_posynomials = self.classify_gp_constraints(gp_posynomials)
        self.ready_gp_constraints += ready_gp_constraints

        if equality_constraints:
            warnings.warn('Equality constraints will not be robustified.')

    def setup(self, verbosity=0, **options):
        for option, key in options.items():
            self.setting.set(option, key)

        start_time = time()

        old_solution = self.nominal_solve
        reached_feasibility = 0

        for count in range(self.setting.get('iterationLimit')):
            if verbosity > 0:
                print("iteration %s" % (count + 1))
            ready_sp_constraints, to_linearize_sp_posynomials, large_sp_posynomials = self. \
                approximate_and_classify_sp_constraints(old_solution, self.number_of_gp_posynomials)

            ready_constraints = self.ready_gp_constraints + ready_sp_constraints
            to_linearize_posynomials = self.to_linearize_gp_posynomials + to_linearize_sp_posynomials
            large_posynomials = self.large_gp_posynomials + large_sp_posynomials

            permutation_indices = self.new_permutation_indices(old_solution, large_posynomials)

            two_term_data_posynomials = []

            for i, two_term_approximation in enumerate(large_posynomials):
                permutation = two_term_approximation.list_of_permutations[permutation_indices[i]]
                no_data, data = TwoTermApproximation. \
                    equivalent_posynomial(two_term_approximation.p, i, permutation, False)
                ready_constraints += no_data
                two_term_data_posynomials += [constraint.unsubbed[0] for constraint in data]
            two_term_data_posynomials += to_linearize_posynomials
            if reached_feasibility:
                self._robust_model, _ = self. \
                    linearize_and_return_upper_lower_models(two_term_data_posynomials,
                                                            self.robust_solve_properties['numoflinearsections'],
                                                            ready_constraints)
                new_solution = RobustModel.internalsolve(self._robust_model, verbosity=0)
            else:
                try:
                    self.robust_solve_properties['numoflinearsections'], new_solution, self._robust_model = self. \
                        find_number_of_piece_wise_linearization(two_term_data_posynomials, ready_constraints)
                    reached_feasibility += 1
                except Exception:
                    self.robust_solve_properties['numoflinearsections'], new_solution, self._robust_model = self. \
                        find_number_of_piece_wise_linearization(two_term_data_posynomials, ready_constraints,
                                                                feasible=True)
            try:
                rel_tol = np.abs((new_solution['cost'] - old_solution['cost']) / old_solution['cost'])
            except:
                # Triggers when nominal solution to goal problem is the nominal problem.
                if verbosity > 0:
                    print("Goal program proceeding...")
                rel_tol = 1
            if verbosity > 0:
                if not reached_feasibility:
                    print("feasibility is not reached yet")
                elif reached_feasibility == 1:
                    print("feasibility is reached")
                print("relative tolerance = %s" % rel_tol)
            if reached_feasibility <= 1 and two_term_data_posynomials:
                self.robust_solve_properties['slopes'], self.robust_solve_properties['intercepts'], _, _, _ = \
                    LinearizeTwoTermPosynomials.linearization_coeff(
                        self.robust_solve_properties['numoflinearsections'])

            self._sequence_of_rgps.append(self._robust_model)

            if rel_tol <= self.setting.get('iterationsRelativeTolerance'):
                break
            else:
                old_solution = new_solution

        if reached_feasibility < 1:
            raise Exception("feasibility is not reached. If the solution seems to converge, try "
                            "increasing iterationLimit = %s. Increasing the allowed number of permutations might also "
                            "help" % self.setting.get('iterationLimit'))
        self.robust_solve_properties['setuptime'] = time() - start_time

    def robustsolve(self, verbosity=1, **options):
        if self._robust_model is None:
            self.setup(verbosity, **options)
        try:
            sol = self._robust_model.solve(verbosity=verbosity)
        except InvalidGPConstraint:
            sol = self._robust_model.localsolve(verbosity=verbosity)
        if verbosity > 0:
            print ("solving needed %s iterations." % len(self._sequence_of_rgps))
            print ("setting up took %s seconds." % self.robust_solve_properties['setuptime'])
        sol.update(self.robust_solve_properties)
        return sol

    def approximate_and_classify_sp_constraints(self, solution, number_of_gp_posynomials):
        sp_gp_approximation = []
        with SignomialsEnabled():
            for cs in self.sp_constraints:
                css = SignomialInequality(cs.left.sub(solution["constants"]), cs.oper, cs.right.sub(solution["constants"]))
                sp_gp_approximation.append(css.as_gpconstr(x0=solution["freevariables"]).unsubbed[0])
            ready_sp_constraints, to_linearize_sp_posynomials, large_sp_posynomial = self.\
                classify_gp_constraints(sp_gp_approximation, number_of_gp_posynomials)
            for cs in self.sp_equality_constraints:
                css = SingleSignomialEquality(cs.left.sub(solution["constants"]), cs.right.sub(solution["constants"]))
                ready_sp_constraints.append(css.as_gpconstr(x0=solution["freevariables"]))
        return ready_sp_constraints, to_linearize_sp_posynomials, large_sp_posynomial

    def classify_gp_constraints(self, gp_posynomials, offset=0):
        data_gp_posynomials = []
        ready_gp_constraints = []
        for i, p in enumerate(gp_posynomials):
            equivalent_p = EquivalentPosynomials(p, i + offset, self.setting.get('simpleModel'),
                                                 self.dependent_uncertainty_set)
            no_data, data = equivalent_p.no_data_constraints, equivalent_p.data_constraints
            data_gp_posynomials += [posy.unsubbed[0] for posy in data]
            ready_gp_constraints += no_data

        to_linearize_gp_posynomials = []
        large_gp_posynomials = []
        for i, p in enumerate(data_gp_posynomials):
            if len(p.exps) == 1:
                robust_monomial, _ = self.robustify_monomial(p)
                ready_gp_constraints += [robust_monomial <= 1]
            elif len(p.exps) == 2 and self.setting.get('linearizeTwoTerm'):
                to_linearize_gp_posynomials += [p]
            else:
                if self.setting.get('twoTerm'):
                    two_term_approximation = TwoTermApproximation(p, self.setting)
                    large_gp_posynomials.append(two_term_approximation)
                else:
                    robust_large_p = RobustifyLargePosynomial(p, self.type_of_uncertainty_set, self.setting)
                    ready_gp_constraints += robust_large_p. \
                        robustify_large_posynomial(self.type_of_uncertainty_set, i + offset, self.setting)

        return ready_gp_constraints, to_linearize_gp_posynomials, large_gp_posynomials

    def robustify_monomial(self, monomial):
        new_monomial_exps = RobustGPTools. \
            only_uncertain_vars_monomial(monomial.exps[0])
        m_direct_uncertain_vars = [var for var in list(new_monomial_exps.keys()) if RobustGPTools.is_uncertain(var)]

        l_norm = 0
        for var in m_direct_uncertain_vars:
            eta = RobustGPTools.generate_etas(var)
            exponent = -new_monomial_exps.get(var.key)
            pert = exponent * eta

            if self.type_of_uncertainty_set == 'box':
                l_norm += np.abs(pert)
            elif self.type_of_uncertainty_set == 'elliptical':
                l_norm += pert ** 2
            elif self.type_of_uncertainty_set == 'one norm':
                l_norm = max(l_norm, np.abs(pert))
            else:
                raise Exception('This type of set is not supported')
        if self.type_of_uncertainty_set == 'elliptical':
            l_norm = np.sqrt(l_norm)
        g = self.setting.get('gamma')
        # Fifth order Taylor approx of the e**gamma, so that gamma can be a variable
        if l_norm != 0:
            robust_monomial = monomial**(1/l_norm) * (1.+g+1./2.*g**2+1./6.*g**3+1./24.*g**4+1./120.*g**5)
        else:
            l_norm = 1
            robust_monomial = monomial
        return robust_monomial, l_norm

    def robustify_set_of_monomials(self, set_of_monomials, feasible=False):
        robust_set_of_monomial_constraints = []
        slackvar = Variable()
        for monomial in set_of_monomials:
            robust_monomial, l_norm = self.robustify_monomial(monomial)
            robust_set_of_monomial_constraints += [robust_monomial <= slackvar ** (feasible/l_norm)]
        robust_set_of_monomial_constraints += [slackvar >= 1, slackvar <= 1000]
        return robust_set_of_monomial_constraints, slackvar

    def calculate_value_of_two_term_approximated_posynomial(self, two_term_approximation, index_of_permutation,
                                                            solution):
        """... for a given index among all permutations, and a solution to the model. """
        permutation = two_term_approximation.list_of_permutations[index_of_permutation]

        number_of_two_terms = int(len(permutation) / 2)
        num_of_linear_sections = self.robust_solve_properties['numoflinearsections']
        slopes = self.robust_solve_properties['slopes']
        intercepts = self.robust_solve_properties['intercepts']
        values = []

        mons = two_term_approximation.p.chop()

        for i in range(number_of_two_terms):
            monomials = []
            first_monomial = mons[2*i]
            second_monomial = mons[2*i+1]

            monomials += [first_monomial]
            for j in range(num_of_linear_sections - 2):
                monomials += [first_monomial ** slopes[num_of_linear_sections - 3 - j] *
                              second_monomial ** slopes[j] * np.exp(intercepts[j])]
            monomials += [second_monomial]
            subs_monomials = []
            for j in range(len(monomials)):
                # st3 = time()
                robust_monomial, _ = self.robustify_monomial(monomials[j])
                monomials[j] = robust_monomial.sub(solution['variables'])
                # print "subs for a monomial is taking too much time", time()-st3
                subs_monomials.append(monomials[j].cs[0])
            values.append(max(subs_monomials))
        if number_of_two_terms % 2 != 0:
            monomial = mons[len(permutation) - 1]
            robust_monomial, _ = self.robustify_monomial(monomial)
            monomial = robust_monomial.sub(solution['variables'])
            values.append(monomial.cs[0])
        return sum(values)

    def find_permutation_with_minimum_value(self, two_term_approximation, solution):
        """ ... to be able to find the least conservative two term approximation. """
        minimum_value = np.inf
        minimum_index = len(two_term_approximation.list_of_permutations)
        for i in range(len(two_term_approximation.list_of_permutations)):
            temp_value = self. \
                calculate_value_of_two_term_approximated_posynomial(two_term_approximation, i, solution)
            if temp_value < minimum_value:
                minimum_value = temp_value
                minimum_index = i
        return minimum_index

    def linearize_and_return_upper_lower_models(self, two_term_data_posynomials, r, ready_constraints, feasible=False):
        no_data_upper_constraints = []
        no_data_lower_constraints = []
        data_posynomials = []

        for i, two_term_p in enumerate(two_term_data_posynomials):
            linearize_p = LinearizeTwoTermPosynomials(two_term_p)
            no_data_upper, no_data_lower, data = linearize_p. \
                linearize(i, r)
            no_data_upper_constraints += no_data_upper
            no_data_lower_constraints += no_data_lower
            data_posynomials += [constraint.unsubbed[0] for constraint in data]
            del linearize_p, no_data_lower, no_data_upper
        data_constraints, slackvar = self.robustify_set_of_monomials(data_posynomials, feasible)

        upper_cons, lower_cons = [no_data_upper_constraints, ready_constraints, data_constraints], \
                                 [no_data_lower_constraints, ready_constraints, data_constraints]

        model_upper = Model(self.nominal_model.cost * slackvar ** (100 * feasible), upper_cons)
        model_lower = Model(self.nominal_model.cost * slackvar ** (100 * feasible), lower_cons)
        model_upper.substitutions.update(self.substitutions)
        model_lower.substitutions.update(self.substitutions)
        model_upper.unique_varkeys, model_lower.unique_varkeys = [self.nominal_model.varkeys] * 2
        model_upper.varkeys.update(model_upper.unique_varkeys)
        model_lower.varkeys.update(model_lower.unique_varkeys)
        del upper_cons, lower_cons, no_data_lower_constraints, no_data_upper_constraints, data_posynomials
        return model_upper, model_lower

    def find_number_of_piece_wise_linearization(self, two_term_data_posynomials, ready_constraints, feasible=False):

        min_r = self.setting.get('minNumOfLinearSections')
        max_r = self.setting.get('maxNumOfLinearSections')
        r = None
        error = None
        sol_upper = None
        model_upper = None

        while min_r <= max_r:
            r = int((min_r + max_r) / 2.0)
            model_upper, model_lower = self. \
                linearize_and_return_upper_lower_models(two_term_data_posynomials, r, ready_constraints, feasible)
            upper_model_infeasible = 0
            try:
                sol_upper = RobustModel.internalsolve(model_upper, verbosity=0)
            except (RuntimeWarning, ValueError):
                upper_model_infeasible = 1
            try:
                sol_lower = RobustModel.internalsolve(model_lower, verbosity=0)
            except (RuntimeWarning, ValueError):
                raise Exception("The model is infeasible")
            if not two_term_data_posynomials:
                self.robust_solve_properties['upperLowerRelError'] = 0
                return 0, sol_upper, model_upper
            if upper_model_infeasible != 1:
                error = (sol_upper.get('cost') - sol_lower.get('cost')) / sol_lower.get('cost')
                if error <= self.setting.get('linearizationTolerance'):
                    max_r = r
                else:
                    min_r = r + 1
            elif r == self.setting.get('maxNumOfLinearSections'):
                self.lower_approximation_is_feasible = True
                raise Exception("The model is infeasible. The lower approximation of the model is feasible, try "
                                "increasing the maximum number of linear sections")
            else:
                min_r = r + 1
            del model_lower, sol_lower
            if max_r == min_r and r == max_r:
                break
        self.robust_solve_properties['upperLowerRelError'] = error
        return r, sol_upper, model_upper

    def new_permutation_indices(self, solution, large_posynomials):
        permutation_indices = []
        for two_term_approximation in large_posynomials:
            permutation_indices.append(self.find_permutation_with_minimum_value(two_term_approximation, solution))
        return permutation_indices

    @staticmethod
    def internalsolve(model, verbosity=0):
        try:
            return model.solve(verbosity=verbosity)
        except InvalidGPConstraint:
            return model.localsolve(verbosity=verbosity)

    def get_robust_model(self):
        if self.sp_constraints:
            return self._sequence_of_rgps
        else:
            return self._robust_model

    def nominalsolve(self):
        return self.nominal_solve

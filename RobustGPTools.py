from numbers import Number
from gpkit import Monomial, Model
import numpy as np
from copy import copy


class RobustGPTools:

    def __init__(self):
        pass

    @staticmethod
    def uncertain_model_variables(model):
        """
        Finds the uncertain variables of a model
        :return: the uncertain variables
        """
        subs_vars = list(model.substitutions.keys())
        uncertain_vars = [var for var in subs_vars if isinstance(var.key.pr, Number)]
        indirect_uncertain_vars = [var for var in subs_vars if isinstance(var.key.pr, Monomial)]
        return uncertain_vars, indirect_uncertain_vars

    @staticmethod
    def only_uncertain_vars_monomial(original_monomial_exps, original_monomial_cs, indirect_uncertain_vars):
        indirect_monomial_uncertain_vars = [var for var in original_monomial_exps.keys() if var in indirect_uncertain_vars]
        new_monomial_exps = copy(original_monomial_exps)
        for var in indirect_monomial_uncertain_vars:
            new_vars_exps = RobustGPTools.\
                replace_indirect_uncertain_variable_by_equivalent(var.key.pr, original_monomial_exps[var])
            del new_monomial_exps[var]
            new_monomial_exps.update(new_vars_exps)
        return new_monomial_exps

    @staticmethod
    def replace_indirect_uncertain_variable_by_equivalent(monomial, exps):
        equivalent = {}

        for var in monomial.exps[0]:
            if var.key.pr is None or isinstance(var.key.pr, Number):
                equivalent.update({var: exps*monomial.exps[0][var]})
            elif isinstance(var.key.pr, Monomial):
                equivalent.update(RobustGPTools.
                                  replace_indirect_uncertain_variable_by_equivalent(var.key.pr, monomial.exps[0][var]))
        return equivalent

    @staticmethod
    def check_if_no_data(p_uncertain_vars, monomial):
        """
        Checks if there is no uncertain data in a monomial
        :param p_uncertain_vars: the posynomial's uncertain variables
        :param monomial: the monomial to be checked for
        :return: True or False
        """
        intersection = [var for var in p_uncertain_vars if var.key in monomial]
        if not intersection:
            return True
        else:
            return False

    @staticmethod
    def probability_of_failure(rob_model, number_of_iterations, design_variables):

        failure = 0
        success = 0

        solution = rob_model.solve()
        variable_solution = solution['variables']
        sum_cost = 0
        for i in xrange(number_of_iterations):
            print('iteration: %s' % i)
            new_model = RobustGPTools.evaluate_random_model(rob_model.model, variable_solution, design_variables)
            fail_success, cost = RobustGPTools.fail_or_success(new_model)
            print cost
            sum_cost = sum_cost + cost
            if fail_success:
                success = success + 1
            else:
                failure = failure + 1
        if success > 0:
            cost_average = sum_cost / (success + 0.0)
        else:
            cost_average = None
        prob = failure / (failure + success + 0.0)
        return prob, cost_average

    @staticmethod
    def evaluate_random_model(old_model, solution, design_variables):
        model = SameModel(old_model)
        model.substitutions.update(old_model.substitutions)
        free_vars = [var for var in model.varkeys.keys()
                     if var not in model.substitutions.keys()]
        uncertain_vars = [var for var in model.substitutions.keys()
                          if "pr" in var.key.descr]
        for key in free_vars:
            if key.descr['name'] in design_variables:
                try:
                    model.substitutions[key] = solution.get(key).m
                except:
                    model.substitutions[key] = solution.get(key)
        for key in uncertain_vars:
            val = model[key].key.descr["value"]
            pr = model[key].key.descr["pr"]
            if pr != 0:
                # sigma = pr * val / 300.0
                # new_val = np.random.normal(val,sigma)
                new_val = np.random.uniform(val - pr * val / 100.0, val + pr * val / 100.0)
                model.substitutions[key] = new_val
        return model

    @staticmethod
    def fail_or_success(model):
        try:
            try:
                sol = model.solve(verbosity=0)
            except:
                sol = model.localsolve(verbosity=0)
            return True, sol['cost']
        except:
            return False, 0


class SameModel(Model):
    """
    copies a model without the substitutions
    """

    def setup(self, model):
        """
        replicates a gp model into a new model
        :param model: the original model
        :return: the new model
        """
        constraints = []
        # unc = SameModel.uncertain_model_variables(model)
        for i, p in enumerate(model.as_posyslt1()):
            # p_unc = [var.key.name for var in p.varkeys if var in unc]
            # if 'P_{acc}' in p_unc:
                # print p
                # print p_unc
                # print len(p.exps)
                # eq_p = EquivalentPosynomials(p, unc, 0, False, False)
                # print("------------------------------------")
                # print(eq_p.no_data_constraints)
                # print(eq_p.data_constraints)
                # print("__________________________________")
            # print p_unc
            constraints.append(p <= 1)
        self.cost = model.cost
        return constraints

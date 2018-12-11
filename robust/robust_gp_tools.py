from gpkit import Model, nomials
from gpkit.nomials import MonomialEquality, PosynomialInequality
from gpkit.exceptions import InvalidGPConstraint
from gpkit.small_scripts import mag
import numpy as np
from copy import copy

import multiprocessing as mp

class RobustGPTools:
    def __init__(self):
        pass

    @staticmethod
    def variables_bynameandmodels(model, name, **descr):
        all_vars = model.variables_byname(name)
        if 'models' in descr:
            temp_vars = []
            for var in all_vars:
                if set(descr['models']) <= set(var.key.models):
                    temp_vars.append(var)
            all_vars = temp_vars
            if 'modelnums' in descr:
                temp_vars = []
                for var in all_vars:
                    if all(var.key.modelnums[var.key.models.index(model)] == descr['modelnums'][i]
                           for i, model in enumerate(descr['models'])):
                        temp_vars.append(var)
                all_vars = temp_vars
        return all_vars

    @staticmethod
    def generate_etas(var):

        r = var.key.r
        eta_max = np.log(r)
        eta_min = - np.log(r)

        return eta_min, eta_max

    @staticmethod
    def is_directly_uncertain(variable):
        return variable.key.r is not None and variable.key.r > 1 and variable.key.rel is None

    @staticmethod
    def is_indirectly_uncertain(variable):
        return variable.key.rel is not None

    @staticmethod
    def is_uncertain(variable):
        return RobustGPTools.is_indirectly_uncertain(variable) or RobustGPTools.is_directly_uncertain(variable)

    @staticmethod
    def from_nomial_array_to_variables(model, the_vars, nomial_array):
        if isinstance(nomial_array, nomials.variables.Variable):
            the_vars.append(nomial_array.key)
            return

        for i in model[nomial_array.key.name]:
            RobustGPTools.from_nomial_array_to_variables(model, the_vars, i)
        return the_vars

    @staticmethod
    def only_uncertain_vars_monomial(original_monomial_exps):
        indirect_monomial_uncertain_vars = [var for var in original_monomial_exps.keys() if
                                            RobustGPTools.is_indirectly_uncertain(var)]
        new_monomial_exps = copy(original_monomial_exps)
        for var in indirect_monomial_uncertain_vars:
            new_vars_exps = RobustGPTools. \
                replace_indirect_uncertain_variable_by_equivalent(var.key.rel, original_monomial_exps[var])
            del new_monomial_exps[var]
            new_monomial_exps.update(new_vars_exps)
        return new_monomial_exps

    @staticmethod
    def replace_indirect_uncertain_variable_by_equivalent(monomial, exps):
        equivalent = {}

        for var in monomial.exps[0]:
            if RobustGPTools.is_indirectly_uncertain(var):
                equivalent.update(RobustGPTools.
                                  replace_indirect_uncertain_variable_by_equivalent(var.key.rel, monomial.exps[0][var]))
            else:
                equivalent.update({var: exps * monomial.exps[0][var]})
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
    def probability_of_failure(model, solution, directly_uncertain_vars_subs, number_of_iterations, verbosity=0, parallel=False):
        if parallel:
            pool = mp.Pool(mp.cpu_count()-1)
            processes = []
            for i in range(number_of_time_average_solves):
                p = pool.apply_async(confirmSuccess, (model, solution, directly_uncertain_vars_subs[i]))
                processes.append(p)
            pool.close()
            pool.join()
            results = [p.get() for p in processes]
        else:
            results = [confirmSuccess(model, solution, directly_uncertain_vars_subs[i]) for i in range(number_of_iterations)]

        costs = [0 if results[i] is None else mag(results[i]) for i in range(number_of_iterations)]
        print costs
        if np.sum(costs) > 0:
            inds = list(np.nonzero(costs)[0])
            nonzero_costs = [costs[i] for i in inds]
            cost_average = np.mean(nonzero_costs)
            cost_variance = np.var(nonzero_costs)
        else:
            cost_average = None
            cost_variance = None
        prob = 1. - (len(np.nonzero(costs)[0])/(number_of_iterations + 0.0))
        return prob, cost_average, cost_variance

    class DesignedModel(Model):
        def setup(self, model, solution, directly_uncertain_vars_subs):
            subs = {k: v for k, v in solution["freevariables"].items()
                    if k in model.varkeys and k.key.fix is True}
            subs.update(model.substitutions)
            subs.update(directly_uncertain_vars_subs)
            self.cost = model.cost
            return model, subs

    @staticmethod
    def fail_or_success(model):
        try:
            try:
                sol = model.solve(verbosity=0)
            except InvalidGPConstraint:
                sol = model.localsolve(verbosity=0)
            return True, sol['cost']
        except:  # ValueError:
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
        all_constraints = model.flat(constraintsets=False)
        constraints = []
        for cs in all_constraints:
            if isinstance(cs, MonomialEquality):
                constraints += [cs]
            elif isinstance(cs, PosynomialInequality):
                constraints += [cs.as_posyslt1()[0] <= 1]
        self.cost = model.cost
        return constraints


class EqualModel(Model):
    def setup(self, model):
        subs = model.substitutions
        self.cost = model.cost
        return model, subs

def confirmSuccess(model, solution, uncertainsub):
    new_model = RobustGPTools.DesignedModel(model, solution, uncertainsub)
    fail_success, cost = RobustGPTools.fail_or_success(new_model)
    return cost

if __name__ == '__main__':
    pass

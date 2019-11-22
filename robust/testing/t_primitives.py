import numpy as np
import os
import pickle

import unittest
from gpkit.tests.helpers import run_tests
from gpkit import Variable, Model

from robust.robust_gp_tools import RobustGPTools
from robust.twoterm_approximation import TwoTermApproximation
from robust.robust import RobustModel
from robust.margin import MarginModel
from robust.testing.models import simple_wing
from robust.testing.models import sp_test_model, gp_test_model

class TestPrimitives(unittest.TestCase):
    def test_MarginModel(self):
        """ Tests creation and solution of MarginModel"""
        m = sp_test_model()
        mm = MarginModel(m, gamma=0.5)
        margin_solution = mm.localsolve(verbosity=0)
        uncertain_varkeys = [k for k in m.varkeys if RobustGPTools.is_directly_uncertain(k)]
        # Checking margin allocation
        for key in list(uncertain_varkeys):
            assert(mm.substitutions.get(key) == m.substitutions.get(key) *
                                                (1.+mm.setting.get("gamma")*key.pr/100.))
        self.assertGreater(margin_solution['cost'], mm.nominal_cost)

    def test_GoalProgram(self):
        """ Tests creation and solution of RobustModels with variable Gamma,
            and tightness of the two solution methods."""
        m = sp_test_model()
        n = 6
        gammas = np.linspace(0.,1.,n)
        Gamma = Variable('\\Gamma', '-', 'Uncertainty bound')
        solBound = Variable('1+\\delta', '-', 'Acceptable optimal solution bound', fix = True)
        nominal_cost = m.localsolve(verbosity=0)['cost']
        box_cost = [RobustModel(m, 'box', gamma = gammas[i]).robustsolve(verbosity=0)['cost']
                    for i in range(n)]/(np.ones(n)*nominal_cost)
        ell_cost = [RobustModel(m, 'elliptical', gamma = gammas[i]).robustsolve(verbosity=0)['cost']
                    for i in range(n)]/(np.ones(n)*nominal_cost)
        self.assertTrue(all(box_cost >= nominal_cost))
        self.assertTrue(all(ell_cost >= nominal_cost))
        # Creating goal model
        gm = Model(1 / Gamma, [m, m.cost <= nominal_cost * solBound, Gamma <= 1e30, solBound <= 1e30],
                  m.substitutions)
        goal_box_gamma = []
        goal_ell_gamma = []
        for i in range(1,n):
            gm.substitutions.update({'1+\\delta': box_cost[i]})
            robust_goal_bm = RobustModel(gm, 'box', gamma=Gamma)
            goal_box_gamma.append(robust_goal_bm.robustsolve(verbosity=0)['cost']**-1)
            gm.substitutions.update({'1+\\delta': ell_cost[i]})
            robust_goal_em = RobustModel(gm, 'elliptical', gamma=Gamma)
            goal_ell_gamma.append(robust_goal_em.robustsolve(verbosity=0)['cost']**-1)
            self.assertAlmostEqual(goal_box_gamma[i-1], gammas[i], places=5)
            self.assertAlmostEqual(goal_ell_gamma[i-1], gammas[i], places=5)

    def test_conservativeness(self):
        """ Testing conservativeness of solution methods"""
        m = gp_test_model()
        sm = m.solve(verbosity=0)
        sem = RobustModel(m, 'elliptical').robustsolve(verbosity=0)
        smm = MarginModel(m).solve(verbosity=0)
        sbm = RobustModel(m, 'box').robustsolve(verbosity=0)
        self.assertTrue(sm['cost'] <= sem['cost'] <= smm['cost'] <= sbm['cost'])

    # def test_robustify_monomial(self):
    #     """ Testing whether monomials are robustified correctly"""
    #     m = gp_test_model()
    #     monys = []
    #     for c in m.flat(constraintsets=False):
    #         for monomial in c.as_posyslt1()[0].chop():
    #             monys.append(monomial)
    #     uncertain_vars = [i for i in m.varkeys if RobustGPTools.is_directly_uncertain(i)]
    #     rm = RobustModel(m, 'box')
    #     robust_monys = [rm.robustify_monomial(mony) for mony in monys]
    #
    # def test_two_term_tolerance(self):
    #     m = gp_test_model()
    #     rm = RobustModel(m, 'box')
    #     posy = [c for c in m.flat(constraintsets=False)][1].left
    #     tta = TwoTermApproximation(posy, rm.setting)
    #     data_constr = []
    #     no_data_constr = []
    #     for i,v in enumerate(tta.list_of_permutations):
    #         ndc, dc = tta.equivalent_posynomial(posy, i, v, False)
    #         no_data_constr.append(ndc)
    #         data_constr.append(dc)
    #
    #     rm.calculate_value_of_two_term_approximated_posynomial(two_term_approximation, index_of_permutation,
    #                                                         solution)

    def test_methods(self):
        m = gp_test_model()
        nominal_solution = m.solve(verbosity=0)
        methods = [{'name': 'BestPairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
                   {'name': 'LinearizedPerturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
                   {'name': 'SimpleConservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True},
                   {'name': 'TwoTerm', 'twoTerm': False, 'boyd': True, 'simpleModel': False}
                   ]
        uncertainty_sets = ['box', 'elliptical']
        gamma = 0.5
        for method in methods:
            for uncertainty_set in uncertainty_sets:
                rm = RobustModel(m, uncertainty_set, gamma=gamma, twoTerm=method['twoTerm'],
                                           boyd=method['boyd'], simpleModel=method['simpleModel'],
                                           nominalsolve=nominal_solution)
                sol = rm.robustsolve(verbosity=0)
                # sol.save(os.path.dirname(__file__) +
                #                            'diffs/test_methods/' +
                #                            method['name'] + '_' + uncertainty_set)
                self.assertTrue(sol.almost_equal(pickle.load(open(os.path.dirname(__file__) +
                                           '/diffs/test_methods/' +
                                           method['name'] + '_' + uncertainty_set)), reltol=1e-4))

TESTS = [TestPrimitives]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

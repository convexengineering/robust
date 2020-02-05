from builtins import range
from gpkit import Variable, SignomialsEnabled
from gpkit.nomials import Posynomial
import numpy as np

import unittest
from gpkit.tests.helpers import run_tests

from robust.robust_gp_tools import RobustGPTools
from robust.testing.models import sp_test_model, simple_wing

class TestRobustGPTools(unittest.TestCase):
    def test_check_if_no_data(self):
        for _ in range(20):
            number_of_monomials = int(30*np.random.random())+1
            number_of_gp_variables = int(np.random.rand()*15) + 1
            number_of_uncertain_variables = int(np.random.rand()*4) + 1
            vector_to_choose_from = [0, 0, 0, 0, 0, 0, 0, 0, 1, -1]

            m = number_of_monomials*[1]
            p_uncertain_vars = []
            data_monomials = []

            for j in range(number_of_monomials):
                for i in range(number_of_gp_variables):
                    x = Variable('x_%s' % i)
                    m[j] *= x**(np.random.rand()*10 - 5)

            for i in range(number_of_uncertain_variables):
                u = Variable('u_%s' % i, np.random.random(), pr=100*np.random.random())
                p_uncertain_vars.append(u)
                neg_pos_neutral_powers = [vector_to_choose_from[int(10*np.random.random())] for _ in range(number_of_monomials)]

                for j in range(number_of_monomials):
                    m[j] *= u**(np.random.rand()*5*neg_pos_neutral_powers[j])
                    if neg_pos_neutral_powers[j] != 0:
                        data_monomials.append(j)

            for i in range(number_of_monomials):
                if i in data_monomials:
                    # noinspection PyUnresolvedReferences
                    self.assertFalse(RobustGPTools.check_if_no_data(p_uncertain_vars, m[i].exps[0]))
                else:
                    # noinspection PyUnresolvedReferences
                    self.assertTrue(RobustGPTools.check_if_no_data(p_uncertain_vars, m[i].exps[0]))

    def test_monomials_from_data(self):
        mtest = simple_wing()
        constraints = [i for i in mtest.flat(constraintsets=False)]
        for constraint in constraints:
            # Just testing LHS of constraints
            exps = [i.exps[0] for i in constraint.left.chop()]
            cs = [i.cs[0] for i in constraint.left.chop()]
            new_monos = RobustGPTools.monomials_from_data(exps, cs)
            self.assertIsInstance(sum(new_monos), Posynomial)
            self.assertEqual([i.exps for i in new_monos],
                             [j.exps for j in constraint.left.chop()])

TESTS = [TestRobustGPTools]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

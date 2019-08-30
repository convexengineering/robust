from robust.robust_gp_tools import RobustGPTools
from robust.margin import MarginModel
from models import sp_test_model
import unittest
from gpkit.tests.helpers import run_tests

class TestPrimitives(unittest.TestCase):
    def test_MarginModel(self):
        """ Tests creation and solution of MarginModel"""
        m = sp_test_model()
        mm = MarginModel(m, gamma = 0.5)
        margin_solution = mm.localsolve(verbosity=0)
        uncertain_varkeys = [k for k in m.varkeys if RobustGPTools.is_directly_uncertain(k)]
        # Checking margin allocation
        for key in list(uncertain_varkeys):
            assert(mm.substitutions.get(key) == m.substitutions.get(key) *
                                                (1.+mm.setting.get("gamma")*key.pr/100.))
        self.assertGreater(margin_solution['cost'], mm.nominal_cost)

TESTS = [TestPrimitives]

if __name__ == "__main__":
    run_tests(TESTS)

from robust.robust_gp_tools import RobustGPTools
from robust.robust import RobustModel
from robust.margin import MarginModel
from robust.testing.models import gp_test_model, sp_test_model

import numpy as np

def test_MarginModel():
    """ Tests creation and solution of MarginModel"""
    m = sp_test_model()
    mm = MarginModel(m, gamma=0.5)
    margin_solution = mm.localsolve(verbosity=0)
    uncertain_varkeys = [k for k in m.varkeys if RobustGPTools.is_directly_uncertain(k)]
    # Checking margin allocation
    for key in list(uncertain_varkeys):
        assert(mm.substitutions.get(key) == m.substitutions.get(key) *
                                            (1.+mm.setting.get("gamma")*key.pr/100.))
    assert(margin_solution['cost'] >= mm.nominal_cost)

def test_conservativeness():
    """ Testing conservativeness of solution methods"""
    m = sp_test_model()
    sm = m.localsolve(verbosity=0)
    sem = RobustModel(m, 'elliptical').robustsolve(verbosity=0)
    smm = MarginModel(m).localsolve(verbosity=0)
    sbm = RobustModel(m, 'box').robustsolve(verbosity=0)
    assert(sm['cost'] <= sem['cost'] <= smm['cost'] <= sbm['cost'])

# def test_robustify_monomial():
#     """ Testing whether monomials are robustified correctly"""
#     m = gp_test_model()
#     chopped_posys = []
#     for c in m.flat(constraintsets=False):
#         chopped_posys.append(c.as_posyslt1()[0].chop())
#     rm = RobustModel(m, 'elliptical')

def test_methods():
    m = gp_test_model()
    nominal_solution = m.solve(verbosity=0)
    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True}
               ]
    uncertainty_sets = ['box', 'elliptical']
    for method in methods:
        for uncertainty_set in uncertainty_sets:
            gamma = 1.2*np.random.uniform()
            rm = RobustModel(m, uncertainty_set, gamma=gamma, twoTerm=method['twoTerm'],
                                       boyd=method['boyd'], simpleModel=method['simpleModel'],
                                       nominalsolve=nominal_solution)
            _ = rm.robustsolve(verbosity=0)

def test():
    test_MarginModel()
    test_conservativeness()
    test_methods()

if __name__ == "__main__":
    test()

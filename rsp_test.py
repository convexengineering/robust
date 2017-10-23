from gpkit.constraints.sgp import SequentialGeometricProgram
import GPModels as models
from gpkit.nomials import SignomialInequality

simp = models.simpleWingSP()

allcs = simp.flat(constraintsets=False)

gpcs = []
spcs = []
appspcs = []

x0 = simp.localsolve()['variables']

for cs in allcs:
    if isinstance(cs, SignomialInequality):
        spcs.append(cs.as_posyslt1())
        appspcs.extend(cs.as_approxsgt(x0))
    else:
        gpcs.append(cs)

"""
from RobustSP import RobustSPModel
import GPModels

model = GPModels.simpleWingSP()
printing_list = []
for gamma in [1]:  # gammas:
    printing_list.append("________________________________________________________________________")

    print "box boyd:"
    model_box_boyd = RobustSPModel(model, 'box', boyd=True)
    try:
        sol_model_box_boyd = model_box_boyd.localsolve(verbosity=0)
        st = ''
        if model_box_boyd.lower_approximation_used:
            st = 'box boyd infeasible, lower approximation used:'
        printing_list.append(st + "box boyd %s, gamma = %s" %
                             (sol_model_box_boyd['cost'], gamma))
    except:
        printing_list.append('box boyd infeasible')

    print "ell boyd:"
    model_ell_boyd = RobustSPModel(model, 'elliptical', boyd=True)
    try:
        sol_model_ell_boyd = model_ell_boyd.localsolve(verbosity=0)
        st = ''
        if model_ell_boyd.lower_approximation_used:
            st = 'ell boyd infeasible, lower approximation used:'
        printing_list.append(st+"elliptical boyd %s, gamma = %s"
                             % (sol_model_ell_boyd['cost'], gamma))
    except:
        printing_list.append('elliptical boyd infeasible')

    print "one norm boyd:"
    model_one_boyd = RobustSPModel(model, 'one norm', boyd=True)
    try:
        sol_model_one_boyd = model_one_boyd.localsolve(verbosity=0)
        if model_one_boyd.lower_approximation_used:
            printing_list.append('one norm boyd infeasible, lower approximation used:')
        printing_list.append("one norm boyd %s, gamma = %s"
                             % (sol_model_one_boyd['cost'], gamma))
    except:
        printing_list.append('one norm boyd infeasible')
# -------------------------------------------------------------------------------------------------------------------------------------------------
    print "box cons:"
    model_box_cons = RobustSPModel(model, 'box', simpleModel=True)
    try:
        sol_model_box_cons = model_box_cons.localsolve(verbosity=0)
        if model_box_cons.lower_approximation_used:
            printing_list.append('box cons infeasible, lower approximation used:')
        printing_list.append("box cons %s, gamma = %s"
                             % (sol_model_box_cons['cost'], gamma))
    except:
        printing_list.append('box cons infeasible')

    print "ell cons:"
    model_ell_cons = RobustSPModel(model, 'elliptical', simpleModel=True)
    try:
        sol_model_ell_cons = model_ell_cons.localsolve(verbosity=0)
        if model_ell_cons.lower_approximation_used:
            printing_list.append('ell cons infeasible, lower approximation used:')
        printing_list.append("elliptical cons %s, gamma = %s"
                             % (sol_model_ell_cons['cost'], gamma))
    except:
        printing_list.append('elliptical cons infeasible')

    print "one norm cons:"
    model_one_cons = RobustSPModel(model, 'one norm', simpleModel=True)
    try:
        sol_model_one_cons = model_one_cons.localsolve(verbosity=0)
        if model_one_cons.lower_approximation_used:
            printing_list.append('one norm cons infeasible, lower approximation used:')
        printing_list.append("one norm cons %s, gamma = %s"
                             % (sol_model_one_cons['cost'], gamma))
    except:
        printing_list.append('one norm cons infeasible')
# -------------------------------------------------------------------------------------------------------------------------------------------------
    print "box :"
    model_box = RobustSPModel(model, 'box', twoTerm=False)
    try:
        sol_model_box = model_box.localsolve(verbosity=0)
        if model_box.lower_approximation_used:
            printing_list.append('box infeasible, lower approximation used:')
        printing_list.append("box %s, gamma = %s"
                             % (sol_model_box['cost'], gamma))
    except:
        printing_list.append('box infeasible')

    print "ell:"
    model_ell = RobustSPModel(model, 'elliptical', twoTerm=False)
    try:
        sol_model_ell = model_ell.localsolve(verbosity=0)
        if model_ell.lower_approximation_used:
            printing_list.append('ell infeasible, lower approximation used:')
        printing_list.append("elliptical %s, gamma = %s"
                             % (sol_model_ell['cost'], gamma))
    except:
        printing_list.append('elliptical infeasible')

    print "one norm:"
    model_one = RobustSPModel(model, 'one norm', twoTerm=False)
    try:
        sol_model_one = model_one.localsolve(verbosity=0)
        if model_one.lower_approximation_used:
            printing_list.append('one norm infeasible, lower approximation used:')
        printing_list.append("one norm %s, gamma = %s"
                             % (sol_model_one['cost'], gamma))
    except:
        printing_list.append('one norm infeasible')
# -------------------------------------------------------------------------------------------------------------------------------------------------
    print "box two term:"
    model_box_two_term = RobustSPModel(model, 'box')
    try:
        sol_model_box_two_term = model_box_two_term.localsolve(verbosity=0)
        if model_box_two_term.lower_approximation_used:
            printing_list.append('box two term infeasible, lower approximation used:')
        printing_list.append("box two term %s, gamma = %s"
                             % (sol_model_box_two_term['cost'], gamma))
    except:
        printing_list.append('box two term infeasible')

    print "ell two term:"
    model_ell_two_term = RobustSPModel(model, 'elliptical')
    try:
        sol_model_ell_two_term = model_ell_two_term.localsolve(verbosity=0)
        if model_ell_two_term.lower_approximation_used:
            printing_list.append('ell two term infeasible, lower approximation used:')
        printing_list.append("elliptical two term %s, gamma = %s"
                             % (sol_model_ell_two_term['cost'], 1))
    except:
        printing_list.append('elliptical two term infeasible')

    print "one norm two term:"
    model_one_two_term = RobustSPModel(model, 'one norm')
    try:
        sol_model_one_two_term = model_one_two_term.localsolve(verbosity=0)
        if model_one_two_term.lower_approximation_used:
            printing_list.append('one norm two term infeasible, lower approximation used:')
        printing_list.append("one norm two term %s, gamma = %s"
                             % (sol_model_one_two_term['cost'], gamma))
    except:
        printing_list.append('one norm two term infeasible')
# -------------------------------------------------------------------------------------------------------------------------------------------------
    print "deterministic:"
    try:
        sol_model = model.localsolve(verbosity=0)
        printing_list.append("deterministic %s, gamma = %s"
                             % (sol_model['cost'], gamma))
    except:
        printing_list.append('deterministic infeasible')

for statement in printing_list:
    print(statement)
"""

from Robust import RobustModel
import GPModels as Models
import numpy as np

from RobustGPTools import RobustGPTools
from gpkit.small_scripts import mag
import matplotlib.pyplot as plt

the_model = Models.mike_solar_model(20)
the_nominal_solve = the_model.solve(verbosity=0)
the_gamma = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.29, 0.33, 0.39, 0.45, 0.6, 0.87, 1.05]
the_number_of_iterations = 300
the_min_num_of_linear_sections = 3
the_max_num_of_linear_sections = 50
tol = 0.01
the_verbosity = 0
factor = 0.79

nominal_solve_time = 0
for i in xrange(100):
    nominal_solve_time = nominal_solve_time + the_model.solve(verbosity=0)['soltime']
nominal_solve_time = nominal_solve_time/100

the_directly_uncertain_vars_subs = [{k: np.random.uniform(v - k.key.pr * v / 100.0, v + k.key.pr * v / 100.0)
                                     for k, v in the_model.substitutions.items()
                                     if k in the_model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                    for _ in xrange(the_number_of_iterations)]


def different_uncertainty_sets(gamma, directly_uncertain_vars_subs, number_of_iterations,
                               min_num_of_linear_sections, max_num_of_linear_sections, verbosity):

    print ("box, uncertain exponents, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    iter_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    print ("box, uncertain coeffients, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, twoTerm=False, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    if gamma == factor*the_gamma[-1]:
        print "eer"
        simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                          directly_uncertain_vars_subs, number_of_iterations,
                                                          verbosity=1)
    else:
        simulation = (-1, -1)
    coef_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    print ("box, simple conservative, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, simpleModel=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    if gamma == factor*the_gamma[-1]:
        simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                          directly_uncertain_vars_subs, number_of_iterations,
                                                          verbosity=1)
    else:
        simulation = (-1, -1)
    simple_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    print ("box, state of art, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, boyd=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    if gamma == factor*the_gamma[-1]:
        simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                          directly_uncertain_vars_subs, number_of_iterations,
                                                          verbosity=1)
    else:
        simulation = (-1, -1)
    boyd_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    print ("elliptical, uncertain exponents, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    iter_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    print ("elliptical, uncertain coeffients, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, twoTerm=False, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    if gamma == factor*the_gamma[-1]:
        simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                          directly_uncertain_vars_subs, number_of_iterations,
                                                          verbosity=1)
    else:
        simulation = (-1, -1)
    coef_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    print ("elliptical, simple conservative, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, simpleModel=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    if gamma == factor*the_gamma[-1]:
        simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                          directly_uncertain_vars_subs, number_of_iterations,
                                                          verbosity=1)
    else:
        simulation = (-1, -1)
    simple_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    print ("elliptical, state of art, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, boyd=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=tol)
    robust_model_solve_time = 0
    for _ in xrange(100):
        robust_model_solve_time = robust_model_solve_time + robust_model.robustsolve(verbosity=0)['soltime']
    robust_model_solve_time = robust_model_solve_time/100

    if gamma == factor*the_gamma[-1]:
        simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                          directly_uncertain_vars_subs, number_of_iterations,
                                                          verbosity=1)
    else:
        simulation = (-1, -1)
    boyd_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solve_time,
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'], mag(robust_model_solution['upperLowerRelError']))
    del robust_model, robust_model_solution, simulation

    return iter_box, coef_box, simple_box, boyd_box, iter_ell, coef_ell, simple_ell, boyd_ell

iter_box_prob_of_failure = []
iter_box_obj = []
iter_box_worst_obj = []

coef_box_prob_of_failure = []
coef_box_obj = []
coef_box_worst_obj = []

simple_box_prob_of_failure = []
simple_box_obj = []
simple_box_worst_obj = []

boyd_box_prob_of_failure = []
boyd_box_obj = []
boyd_box_worst_obj = []

iter_ell_prob_of_failure = []
iter_ell_obj = []
iter_ell_worst_obj = []

coef_ell_prob_of_failure = []
coef_ell_obj = []
coef_ell_worst_obj = []

simple_ell_prob_of_failure = []
simple_ell_obj = []
simple_ell_worst_obj = []

boyd_ell_prob_of_failure = []
boyd_ell_obj = []
boyd_ell_worst_obj = []

aeys = []
bs = []
cees = []
ds = []
es = []
fs = []
gs = []
hs = []

the_model = Models.mike_solar_model(20)
for a_gamma in the_gamma:
    a, b, g, c, d, e, h, f = different_uncertainty_sets(factor*a_gamma, the_directly_uncertain_vars_subs,
                                                        the_number_of_iterations, the_min_num_of_linear_sections,
                                                        the_max_num_of_linear_sections, the_verbosity)

    iter_box_prob_of_failure.append(a[0])
    iter_box_obj.append(mag(a[1]))
    iter_box_worst_obj.append(mag(a[2]))
    coef_box_prob_of_failure.append(b[0])
    coef_box_obj.append(mag(b[1]))
    coef_box_worst_obj.append(mag(b[2]))
    simple_box_prob_of_failure.append(g[0])
    simple_box_obj.append(mag(g[1]))
    simple_box_worst_obj.append(mag(g[2]))
    boyd_box_prob_of_failure.append(c[0])
    boyd_box_obj.append(mag(c[1]))
    boyd_box_worst_obj.append(mag(c[2]))
    iter_ell_prob_of_failure.append(d[0])
    iter_ell_obj.append(mag(d[1]))
    iter_ell_worst_obj.append(mag(d[2]))
    coef_ell_prob_of_failure.append(e[0])
    coef_ell_obj.append(mag(e[1]))
    coef_ell_worst_obj.append(mag(e[2]))
    simple_ell_prob_of_failure.append(h[0])
    simple_ell_obj.append(mag(h[1]))
    simple_ell_worst_obj.append(mag(h[2]))
    boyd_ell_prob_of_failure.append(f[0])
    boyd_ell_obj.append(mag(f[1]))
    boyd_ell_worst_obj.append(mag(f[2]))

    aeys.append(a)
    bs.append(b)
    cees.append(c)
    ds.append(d)
    es.append(e)
    fs.append(f)
    gs.append(g)
    hs.append(h)

for i in xrange(len(aeys)):
    print "gamma =", the_gamma[i]
    print aeys[i]
    print bs[i]
    print gs[i]
    print cees[i]
    print ds[i]
    print es[i]
    print hs[i]
    print fs[i]


average_setup_time = [sum([aeys[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([bs[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([gs[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([cees[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([ds[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([es[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([hs[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([fs[i][3] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1)]
rel_average_setup_time = [i / the_nominal_solve['soltime'] for i in average_setup_time]

average_solve_time = [sum([aeys[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([bs[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([gs[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([cees[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([ds[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([es[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([hs[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                      sum([fs[i][4] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1)]
rel_average_solve_time = [i / the_nominal_solve['soltime'] for i in average_solve_time]

average_number_of_constraints = [sum([aeys[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                                 sum([bs[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                                 sum([gs[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                                 sum([cees[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                                 sum([ds[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                                 sum([es[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                                 sum([hs[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                                 sum([fs[i][5] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1)]
rel_average_number_of_constraints = [float(i) / len(the_model.as_posyslt1()) for i in average_number_of_constraints]

average_rel_error = [sum([aeys[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                     sum([bs[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                     sum([gs[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                     sum([cees[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                     sum([ds[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                     sum([es[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                     sum([hs[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1),
                     sum([fs[i][7] for i in range(1, len(the_gamma))]) / (len(the_gamma) - 1)]

average_objective_value = [aeys[-1][1], bs[-1][1], gs[-1][1], cees[-1][1], ds[-1][1], es[-1][1], hs[-1][1], fs[-1][1]]
rel_average_objective_value = [mag(i / the_nominal_solve['cost']) for i in average_objective_value]

worst_objective_value = [aeys[-1][2], bs[-1][2], gs[-1][2], cees[-1][2], ds[-1][2], es[-1][2], hs[-1][2], fs[-1][2]]
rel_worst_objective_value = [mag(i / the_nominal_solve['cost']) for i in worst_objective_value]

print ('Average Setup Time = %s' % rel_average_setup_time)
print ('Average Solve Time = %s' % rel_average_solve_time)
print ('Average Number Of Constraints = %s' % rel_average_number_of_constraints)
print ('Average Relative Error = %s' % average_rel_error)
print ('Average Objective Value = %s' % rel_average_objective_value)
print ('Average Worst Value = %s' % rel_worst_objective_value)

"""
plt.figure()
plt.plot(the_gamma, iter_box_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_obj, 'ro', label='Two Term')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Average Performance: Box Uncertainty Set")
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_box_worst_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_worst_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_worst_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_worst_obj, 'ro', label='Two Term')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Worst-Case Performance: Box Uncertainty Set")
plt.legend(loc=0)
plt.show()
"""
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lines1 = ax1.plot(the_gamma, iter_box_obj, 'r--', label='Weight')
lines2 = ax2.plot(the_gamma, iter_box_prob_of_failure, 'b-', label='Probability of Failure')
ax1.set_xlabel(r'Uncertainty Set Scaling Factor $\Gamma$', fontsize=20)
ax1.set_ylabel("Weight", fontsize=18)
ax2.set_ylabel("Probability of Failure", fontsize=18)
plt.title("Box Uncertainty Set", fontsize=22)
lines = lines1 + lines2
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc="upper center")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lines1 = ax1.plot(the_gamma, iter_ell_obj, 'r--', label='Weight')
lines2 = ax2.plot(the_gamma, iter_ell_prob_of_failure, 'b-', label='Probability of Failure')
ax1.set_xlabel(r'Uncertainty Set Scaling Factor $\Gamma$', fontsize=20)
ax1.set_ylabel("Weight", fontsize=18)
ax2.set_ylabel("Probability of Failure", fontsize=18)
plt.title("Elliptical Uncertainty Set", fontsize=22)
lines = lines1 + lines2
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc="upper center")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.arange(4)
lines1 = ax1.bar(x + [0.2, 0.2, 0.2, 0.2],
                rel_average_objective_value[0:4],
                [0.25, 0.25, 0.25, 0.25], color='r', label='Weight')
lines2 = ax2.bar(x + [0.5, 0.5, 0.5, 0.5],
                 rel_average_number_of_constraints[0:4],
                 [0.25, 0.25, 0.25, 0.25], color='b', label='No. of Constraints')
ax1.set_ylabel("Scaled Average Weight", fontsize=18)
ax2.set_ylabel("Scaled Number of Constraints", fontsize=18)
ax1.set_ylim((1.8, 2.4))
plt.xticks(x + .45, ['Best Pairs', 'Linear Perts', 'Simple', 'Two Term'], fontsize=20)
plt.title('Box Uncertainty Set', fontsize=22)
lines = [lines1, lines2]
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc="upper left")
plt.show()
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.arange(4)
lines1 = ax1.bar(x + [0.2, 0.2, 0.2, 0.2],
                 rel_average_setup_time[0:4],
                 [0.25, 0.25, 0.25, 0.25], color='r', label='Setup Time')
lines2 = ax2.bar(x + [0.5, 0.5, 0.5, 0.5],
                 rel_average_solve_time[0:4],
                 [0.25, 0.25, 0.25, 0.25], color='b', label='Solve Time')
ax1.set_ylabel("Scaled Setup Time", fontsize=18)
ax2.set_ylabel("Scaled solve Time", fontsize=18)
plt.xticks(x + .45, ['Best Pairs', 'Linear Perts', 'Simple', 'Two Term'], fontsize=20)
plt.title('Box Uncertainty Set', fontsize=22)
lines = [lines1, lines2]
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc="upper left")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.arange(4)
lines1 = ax1.bar(x + [0.2, 0.2, 0.2, 0.2],
                 rel_average_objective_value[4:8],
                 [0.25, 0.25, 0.25, 0.25], color='r', label='Weight')
lines2 = ax2.bar(x + [0.5, 0.5, 0.5, 0.5],
                 rel_average_number_of_constraints[4:8],
                 [0.25, 0.25, 0.25, 0.25], color='b', label='No. of Constraints')
ax1.set_ylabel("Scaled Average Weight", fontsize=18)
ax2.set_ylabel("Scaled Number of Constraints", fontsize=18)
ax1.set_ylim((1.2, 1.6))
plt.xticks(x + .45, ['Best Pairs', 'Linear Perts', 'Simple', 'Two Term'], fontsize=20)
plt.title('Elliptical Uncertainty Set', fontsize=22)
lines = [lines1, lines2]
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc="upper left")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.arange(4)
lines1 = ax1.bar(x + [0.2, 0.2, 0.2, 0.2],
                 rel_average_setup_time[4:8],
                 [0.25, 0.25, 0.25, 0.25], color='r', label='Setup Time')
lines2 = ax2.bar(x + [0.5, 0.5, 0.5, 0.5],
                 rel_average_solve_time[4:8],
                 [0.25, 0.25, 0.25, 0.25], color='b', label='Solve Time')
ax1.set_ylabel("Scaled Setup Time", fontsize=18)
ax2.set_ylabel("Scaled solve Time", fontsize=18)
plt.xticks(x + .45, ["Best Pair", 'Linear Perts', 'Simple', 'Two Term'], fontsize=20)
plt.title('Elliptical Uncertainty Set', fontsize=22)
lines = [lines1, lines2]
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc="upper left")
plt.show()
"""
plt.figure()
plt.plot(the_gamma, iter_box_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Average Performance: Box Uncertainty Set")
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_box_worst_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_worst_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_worst_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_worst_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Worst-Case Performance: Box Uncertainty Set")
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_box_prob_of_failure, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_prob_of_failure, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_prob_of_failure, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_prob_of_failure, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Probability of Failure")
plt.title("The Probability of Failure: Box Uncertainty Set")
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_ell_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_ell_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_ell_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Average Performance: Elliptical Uncertainty Set")
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_worst_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_ell_worst_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_ell_worst_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_ell_worst_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Worst-Case Performance: Elliptical Uncertainty Set")
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_prob_of_failure, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_ell_prob_of_failure, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_ell_prob_of_failure, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_ell_prob_of_failure, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Probability of Failure")
plt.title("The Probability of Failure: Elliptical Uncertainty Set")
plt.legend(loc=0)
plt.show()
"""

from Robust import RobustModel
import GPModels as Models
import numpy as np

from RobustGPTools import RobustGPTools
from gpkit.small_scripts import mag
import matplotlib.pyplot as plt
import gc

the_model = Models.mike_solar_model(20)
# the_model = Models.simpleWing()
the_gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.05]  # [0, 0.5, 0.9]
the_number_of_iterations = 200
the_min_num_of_linear_sections = 29
the_max_num_of_linear_sections = 30
the_verbosity = 0
the_directly_uncertain_vars_subs = [{k: np.random.uniform(v - k.key.pr * v / 100.0, v + k.key.pr * v / 100.0)
                                     for k, v in the_model.substitutions.items()
                                     if k in the_model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                    for _ in xrange(the_number_of_iterations)]


def different_uncertainty_sets(gamma, directly_uncertain_vars_subs, number_of_iterations,
                               min_num_of_linear_sections, max_num_of_linear_sections, verbosity):

    print ("iter box gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'box', gamma=gamma)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    iter_box = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    print ("coef box gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'box', gamma=gamma, twoTerm=False)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    coef_box = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    print ("simple box gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'box', gamma=gamma, simpleModel=True)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    simple_box = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    print ("boyd box gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'box', gamma=gamma, boyd=True)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    boyd_box = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    print ("iter ell gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'elliptical', gamma=gamma)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    iter_ell = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    print ("coef ell gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'elliptical', gamma=gamma, twoTerm=True)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    coef_ell = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    print ("simple ell gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'elliptical', gamma=gamma, simpleModel=True)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    simple_ell = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    print ("boyd ell gamma = %s" % gamma)
    model = Models.mike_solar_model(20)
    # model = Models.simpleWing()
    robust_model = RobustModel(model, 'elliptical', gamma=gamma, boyd=True)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations)
    boyd_ell = (
        simulation[0], simulation[1], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del model, robust_model, robust_model_solution, simulation
    gc.collect(2)
    return iter_box, coef_box, simple_box, boyd_box, iter_ell, coef_ell, simple_ell, boyd_ell

iter_box_prob_of_failure = []
iter_box_obj = []
coef_box_prob_of_failure = []
coef_box_obj = []
simple_box_prob_of_failure = []
simple_box_obj = []
boyd_box_prob_of_failure = []
boyd_box_obj = []
iter_ell_prob_of_failure = []
iter_ell_obj = []
coef_ell_prob_of_failure = []
coef_ell_obj = []
simple_ell_prob_of_failure = []
simple_ell_obj = []
boyd_ell_prob_of_failure = []
boyd_ell_obj = []

aeys = []
bs = []
cs = []
ds = []
es = []
fs = []
gs = []
hs = []

for a_gamma in the_gamma:
    a, b, g, c, d, e, h, f = different_uncertainty_sets(0.85*a_gamma, the_directly_uncertain_vars_subs,
                                                        the_number_of_iterations, the_min_num_of_linear_sections,
                                                        the_max_num_of_linear_sections, the_verbosity)
    iter_box_prob_of_failure.append(a[0])
    iter_box_obj.append(mag(a[1]))
    coef_box_prob_of_failure.append(b[0])
    coef_box_obj.append(mag(b[1]))
    simple_box_prob_of_failure.append(g[0])
    simple_box_obj.append(mag(g[1]))
    boyd_box_prob_of_failure.append(c[0])
    boyd_box_obj.append(mag(c[1]))
    iter_ell_prob_of_failure.append(d[0])
    iter_ell_obj.append(mag(d[1]))
    coef_ell_prob_of_failure.append(e[0])
    coef_ell_obj.append(mag(e[1]))
    simple_ell_prob_of_failure.append(h[0])
    simple_ell_obj.append(mag(h[1]))
    boyd_ell_prob_of_failure.append(f[0])
    boyd_ell_obj.append(mag(f[1]))

    aeys.append(a)
    bs.append(b)
    cs.append(c)
    ds.append(d)
    es.append(e)
    fs.append(f)
    gs.append(g)
    hs.append(h)

    gc.collect(2)

for i in xrange(len(aeys)):
    print aeys[i]
    print bs[i]
    print gs[i]
    print cs[i]
    print ds[i]
    print es[i]
    print hs[i]
    print fs[i]

plt.figure()
plt.plot(the_gamma, iter_box_obj, label='uncertain exponents')
plt.plot(the_gamma, coef_box_obj, label='uncertain coefficients')
plt.plot(the_gamma, simple_box_obj, label='simple')
plt.plot(the_gamma, boyd_box_obj, label='boyd')
plt.xlabel("gamma")
plt.ylabel("objective function")
plt.title("box uncertainty set: %d simulations" % the_number_of_iterations)
plt.legend()
plt.show()

plt.figure()
plt.plot(the_gamma, iter_box_prob_of_failure, label='uncertain exponents')
plt.plot(the_gamma, coef_box_prob_of_failure, label='uncertain coefficients')
plt.plot(the_gamma, simple_box_prob_of_failure, label='simple')
plt.plot(the_gamma, boyd_box_prob_of_failure, label='boyd')
plt.xlabel("gamma")
plt.ylabel("probability of failure")
plt.title("box uncertainty set: %d simulations" % the_number_of_iterations)
plt.legend()
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_obj, label='uncertain exponents')
plt.plot(the_gamma, coef_ell_obj, label='uncertain coefficients')
plt.plot(the_gamma, simple_ell_obj, label='simple')
plt.plot(the_gamma, boyd_ell_obj, label='boyd')
plt.xlabel("gamma")
plt.ylabel("objective function")
plt.title("elliptical uncertainty set: %d simulations" % the_number_of_iterations)
plt.legend()
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_prob_of_failure, label='uncertain exponents')
plt.plot(the_gamma, coef_ell_prob_of_failure, label='uncertain coefficients')
plt.plot(the_gamma, simple_ell_prob_of_failure, label='simple')
plt.plot(the_gamma, boyd_ell_prob_of_failure, label='boyd')
plt.xlabel("gamma")
plt.ylabel("probability of failure")
plt.title("elliptical uncertainty set: %d simulations" % the_number_of_iterations)
plt.legend()
plt.show()

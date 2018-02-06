from Robust import RobustModel
import GPModels as Models
import numpy as np

from RobustGPTools import RobustGPTools, EqualModel
from gpkit.small_scripts import mag
import matplotlib.pyplot as plt

the_model = Models.mike_solar_model(20)
the_nominal_solve = the_model.solve()
the_gamma = [0.5, 1]
number_of_iterations = 5
min_num_of_linear_sections = 16
max_num_of_linear_sections = 16
verbosity = 0
factor = 0.79

directly_uncertain_vars_subs = [{k: np.random.uniform(v - k.key.pr * v / 100.0, v + k.key.pr * v / 100.0)
                                 for k, v in the_model.substitutions.items()
                                 if k in the_model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                for _ in xrange(number_of_iterations)]

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
    gamma = factor*a_gamma
    print ("box, uncertain exponents, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    iter_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation
    print iter_box

    print ("box, uncertain coeffients, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, twoTerm=False, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    coef_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation
    print coef_box

    print ("box, simple conservative, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, simpleModel=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    simple_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation

    print ("box, state of art, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    robust_model = RobustModel(the_model, 'box', gamma=gamma, boyd=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    boyd_box = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation

    print ("elliptical, uncertain coeffients, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    # model = EqualModel(the_model)
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001, )
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    iter_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation

    print ("elliptical, uncertain coeffients, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    # model = EqualModel(the_model)
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, twoTerm=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    coef_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation

    print ("elliptical, simple conservative, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    # model = EqualModel(the_model)
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, simpleModel=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    simple_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation

    print ("elliptical, state of art, gamma = %s, max PWL = %s, "
           "min PWL = %s" % (gamma, min_num_of_linear_sections, max_num_of_linear_sections))
    # model = EqualModel(the_model)
    robust_model = RobustModel(the_model, 'elliptical', gamma=gamma, boyd=True, nominalsolve=the_nominal_solve)
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                     minNumOfLinearSections=min_num_of_linear_sections,
                                                     maxNumOfLinearSections=max_num_of_linear_sections,
                                                     linearizationTolerance=0.001)
    simulation = RobustGPTools.probability_of_failure(the_model, robust_model_solution,
                                                      directly_uncertain_vars_subs, number_of_iterations, verbosity=1)
    boyd_ell = (
        simulation[0], simulation[1], robust_model_solution['cost'], robust_model_solution['setuptime'],
        robust_model_solution['soltime'],
        len([cs for cs in robust_model.get_robust_model().flat(constraintsets=False)]),
        robust_model_solution['numoflinearsections'])
    del robust_model, robust_model_solution, simulation

    iter_box_prob_of_failure.append(iter_box[0])
    iter_box_obj.append(mag(iter_box[1]))
    iter_box_worst_obj.append(mag(iter_box[2]))
    coef_box_prob_of_failure.append(coef_box[0])
    coef_box_obj.append(mag(coef_box[1]))
    coef_box_worst_obj.append(mag(coef_box[2]))
    simple_box_prob_of_failure.append(simple_box[0])
    simple_box_obj.append(mag(simple_box[1]))
    simple_box_worst_obj.append(mag(simple_box[2]))
    boyd_box_prob_of_failure.append(boyd_box[0])
    boyd_box_obj.append(mag(boyd_box[1]))
    boyd_box_worst_obj.append(mag(boyd_box[2]))
    iter_ell_prob_of_failure.append(iter_ell[0])
    iter_ell_obj.append(mag(iter_ell[1]))
    iter_ell_worst_obj.append(mag(iter_ell[2]))
    coef_ell_prob_of_failure.append(coef_ell[0])
    coef_ell_obj.append(mag(coef_ell[1]))
    coef_ell_worst_obj.append(mag(coef_ell[2]))
    simple_ell_prob_of_failure.append(simple_ell[0])
    simple_ell_obj.append(mag(simple_ell[1]))
    simple_ell_worst_obj.append(mag(simple_ell[2]))
    boyd_ell_prob_of_failure.append(boyd_ell[0])
    boyd_ell_obj.append(mag(boyd_ell[1]))
    boyd_ell_worst_obj.append(mag(boyd_ell[2]))

    aeys.append(iter_box)
    bs.append(coef_box)
    cees.append(boyd_box)
    ds.append(iter_ell)
    es.append(coef_ell)
    fs.append(boyd_ell)
    gs.append(simple_box)
    hs.append(simple_ell)

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

plt.figure()
plt.plot(the_gamma, iter_box_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Average Performance as a Function of the Size "
          "of the Box Uncertainty Set: %d Simulations" % number_of_iterations)
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_box_worst_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_worst_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_worst_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_worst_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Worst-Case Performance as a Function of the Size "
          "of the Box Uncertainty Set: %d Simulations" % number_of_iterations)
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_box_prob_of_failure, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_box_prob_of_failure, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_box_prob_of_failure, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_box_prob_of_failure, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Probability of Failure")
plt.title("The Probability of Failure as a Function of the Size "
          "of the Box Uncertainty Set: %d Simulations" % number_of_iterations)
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_ell_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_ell_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_ell_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Average Performance as a Function of the Size "
          "of the Elliptical Uncertainty Set: %d Simulations" % number_of_iterations)
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_worst_obj, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_ell_worst_obj, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_ell_worst_obj, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_ell_worst_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Worst-Case Performance as a Function of the Size "
          "of the Elliptical Uncertainty Set: %d Simulations" % number_of_iterations)
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(the_gamma, iter_ell_prob_of_failure, 'r--', label='Uncertain Exponents')
plt.plot(the_gamma, coef_ell_prob_of_failure, 'bs', label='Uncertain Coefficients')
plt.plot(the_gamma, simple_ell_prob_of_failure, 'g^', label='Simple Conservative')
plt.plot(the_gamma, boyd_ell_prob_of_failure, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Probability of Failure")
plt.title("The Probability of Failure as a Function of the Size "
          "of the Elliptical Uncertainty Set: %d Simulations" % number_of_iterations)
plt.legend(loc=0)
plt.show()

from __future__ import print_function
from __future__ import division
from builtins import range
import numpy as np
from gpkit.small_scripts import mag
from gpkit.exceptions import InvalidGPConstraint
from gpkit.small_scripts import mag
from gpkit import Model, Variable, Monomial

from robust.robust import RobustModel
from robust.robust_gp_tools import RobustGPTools
from robust.simulations.read_simulation_data import objective_proboffailure_vs_gamma

import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.stats as stats

# For the following simulation functions, we define common inputs as the following:
#     :model: GP or SP model of interest
#     :model_name: string for printing
#     :gamma: array of floats to specify set size
#     :number_of_iterations: # of MC samples
#     :numbers_of_linear_sections: array of integer sections
#     :linearization_tolerance: max error of pwl approx
#     :verbosity: 0-4 for printout
#     :file_name: directory for printing
#     :number_of_time_average_solves: # of solves for solution time analysis
#     :methods: type of conservative approximation used, dict
#     :uncertainty_sets: string defining type of set
#     :nominal_solution: solution of model with zero uncertainty
#     :nominal_solve_time: solve time of model with zero uncertainty
#     :nominal_number_of_constraints:
#     :directly_uncertain_vars_subs: dict of uncertain parameter MC samples
#     :return:

def pickleable_robust_solve_time(robust_model, verbosity, min_num_of_linear_sections,
                            max_num_of_linear_sections, linearization_tolerance):
    """
    Wrapper for robustsolve for parallelism.
    """
    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                             minNumOfLinearSections=min_num_of_linear_sections,
                                                             maxNumOfLinearSections=max_num_of_linear_sections,
                                                             linearizationTolerance=linearization_tolerance)
    return robust_model_solution['soltime']

def get_avg_robust_solve_time(number_of_time_average_solves, robust_model, verbosity, min_num_of_linear_sections,
                            max_num_of_linear_sections, linearization_tolerance, parallel=False):
    """
    Given a number of solves, gives the average solution time of a robust model. Parallel option.
    """
    if parallel:
        pool = mp.Pool(mp.cpu_count()-1)
        processes = []
        timesolutions = []
        for i in range(number_of_time_average_solves):
            p = pool.apply_async(pickleable_robust_solve_time, args=(robust_model, verbosity, min_num_of_linear_sections,
                            max_num_of_linear_sections,linearization_tolerance), callback=timesolutions.append)
            processes.append(p)
        pool.close()
        pool.join()
    else:
        solutions = [robust_model.robustsolve(verbosity=verbosity)
                                                               for i in range(number_of_time_average_solves)]
        timesolutions = [s['soltime'] for s in solutions]
    return np.mean(timesolutions)

def simulate_robust_model(model, method, uncertainty_set, gamma, directly_uncertain_vars_subs,
                          number_of_iterations, linearization_tolerance, min_num_of_linear_sections,
                          max_num_of_linear_sections, verbosity, nominal_solution,
                          number_of_time_average_solves, parallel=False):
    """
    Simulates a robust model given uncertain outcomes.
    """
    if verbosity > 0:
        print(
            method[
                'name'] + ' under ' + uncertainty_set + ' uncertainty set: \n' + '\t' + 'gamma = %s\n' % gamma
            + '\t' + 'minimum number of piecewise-linear sections = %s\n' % min_num_of_linear_sections
            + '\t' + 'maximum number of piecewise-linear sections = %s\n' % max_num_of_linear_sections)

    robust_model = RobustModel(model, uncertainty_set, gamma=gamma, twoTerm=method['twoTerm'],
                                   boyd=method['boyd'], simpleModel=method['simpleModel'],
                                   nominalsolve=nominal_solution)

    robust_model_solution = robust_model.robustsolve(verbosity=verbosity,
                                                             minNumOfLinearSections=min_num_of_linear_sections,
                                                             maxNumOfLinearSections=max_num_of_linear_sections,
                                                             linearizationTolerance=linearization_tolerance)

    robust_model_solve_time = get_avg_robust_solve_time(number_of_time_average_solves,
                                                        robust_model, verbosity, min_num_of_linear_sections,
                                                        max_num_of_linear_sections,
                                                        linearization_tolerance, parallel)

    simulation_results = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                                  directly_uncertain_vars_subs,
                                                                  number_of_iterations,
                                                                  verbosity=verbosity, parallel=parallel)

    return robust_model, robust_model_solution, robust_model_solve_time, simulation_results


def print_simulation_results(robust_model, robust_model_solution, robust_model_solve_time,
                             nominal_model_solve_time, nominal_no_of_constraints, nominal_cost,
                             simulation_results, file_id):
    file_id.write('\t\t\t' + 'Probability of failure: %s\n' % simulation_results[0])
    file_id.write('\t\t\t' + 'Average performance: %s\n' % mag(simulation_results[1]))
    file_id.write('\t\t\t' + 'Relative average performance: %s\n' %
                      (mag(simulation_results[1]) / float(mag(nominal_cost))))
    file_id.write('\t\t\t' + 'Worst-case performance: %s\n' % mag(robust_model_solution['cost']))
    file_id.write('\t\t\t' + 'Relative worst-case performance: %s\n' %
                      (mag(robust_model_solution['cost']) / float(mag(nominal_cost))))
    try:
        number_of_constraints = \
            len([cnstrnt for cnstrnt in robust_model.get_robust_model().flat(constraintsets=False)])
    except AttributeError:
        number_of_constraints = \
            len([cnstrnt for cnstrnt in robust_model.get_robust_model()[-1].flat(constraintsets=False)])
    file_id.write('\t\t\t' + 'Number of constraints: %s\n' % number_of_constraints)
    file_id.write('\t\t\t' + 'Relative number of constraints: %s\n' %
                      (number_of_constraints / float(nominal_no_of_constraints)))
    file_id.write('\t\t\t' + 'Setup time: %s\n' % robust_model_solution['setuptime'])
    file_id.write('\t\t\t' + 'Relative setup time: %s\n' %
                      (robust_model_solution['setuptime'] / float(nominal_model_solve_time)))
    file_id.write('\t\t\t' + 'Solve time: %s\n' % robust_model_solve_time)
    file_id.write('\t\t\t' + 'Relative solve time: %s\n' %
                      (robust_model_solve_time / float(nominal_model_solve_time)))
    file_id.write('\t\t\t' + 'Number of linear sections: %s\n' % robust_model_solution['numoflinearsections'])
    file_id.write(
        '\t\t\t' + 'Upper lower relative error: %s\n' % mag(robust_model_solution['upperLowerRelError']))


def print_variable_gamma_results(model, model_name, gammas, number_of_iterations,
                                    min_num_of_linear_sections, max_num_of_linear_sections, verbosity,
                                    linearization_tolerance, file_name, number_of_time_average_solves,
                                    methods, uncertainty_sets, nominal_solution, nominal_solve_time,
                                    nominal_number_of_constraints, directly_uncertain_vars_subs):

    f = open(file_name, 'w')
    f.write(model_name + ' Results: variable gamma\n')
    f.write('----------------------------------------------------------\n')
    cost_label = model.cost.str_without()
    split_label = cost_label.split(' ')
    capitalized_cost_label = ''
    for word in split_label:
        capitalized_cost_label += word.capitalize() + ' '
    f.write('Objective: %s\n' % capitalized_cost_label)
    f.write('Units: %s\n' % model.cost.units)
    f.write('----------------------------------------------------------\n')
    f.write('Number of iterations: %s\n' % number_of_iterations)
    f.write('Minimum number of piecewise-linear sections: %s\n' % min_num_of_linear_sections)
    f.write('Maximum number of piecewise-linear sections: %s\n' % max_num_of_linear_sections)
    f.write('Linearization tolerance: %s\n' % linearization_tolerance)
    f.write('----------------------------------------------------------\n')
    f.write('Nominal cost: %s\n' % nominal_solution['cost'])
    f.write('Average nominal solve time: %s\n' % nominal_solve_time)
    f.write('Nominal number of constraints: %s\n' % nominal_number_of_constraints)
    f.write('----------------------------------------------------------\n')

    for gamma in gammas:
        f.write('Gamma = %s:\n' % gamma)
        for method in methods:
            f.write('\t' + method['name'] + ':\n')
            for uncertainty_set in uncertainty_sets:
                f.write('\t\t' + uncertainty_set + ':\n')
                robust_model, robust_model_solution, robust_model_solve_time, simulation_results = \
                    simulate_robust_model(model, method, uncertainty_set, gamma, directly_uncertain_vars_subs,
                                          number_of_iterations, linearization_tolerance,
                                          min_num_of_linear_sections,
                                          max_num_of_linear_sections, verbosity, nominal_solution,
                                          number_of_time_average_solves)
                print_simulation_results(robust_model, robust_model_solution, robust_model_solve_time,
                                         nominal_solve_time, nominal_number_of_constraints, nominal_solution['cost'],
                                         simulation_results, f)
    f.close()

def variable_gamma_results(model, methods, gammas, number_of_iterations,
                                    min_num_of_linear_sections, max_num_of_linear_sections, verbosity,
                                    linearization_tolerance, number_of_time_average_solves,
                                    uncertainty_sets, nominal_solution, directly_uncertain_vars_subs, parallel=False):
    """
    Simulates a GP or SP model for a range of gammas, i.e. uncertainty set size.
    Outputs are dicts that have the key format: [deltaValue (float), methodName (string), uncertaintySet (string)]
    """
    solutions = {}
    solve_times = {}
    simulations = {}
    number_of_constraints = {}
    for gamma in gammas:
        for method in methods:
            for uncertainty_set in uncertainty_sets:
                ind = (gamma, method['name'], uncertainty_set)
                robust_model, robust_model_solution, robust_model_solve_time, simulation_results = \
                    simulate_robust_model(model, method, uncertainty_set, gamma, directly_uncertain_vars_subs,
                                          number_of_iterations, linearization_tolerance,
                                          min_num_of_linear_sections,
                                          max_num_of_linear_sections, verbosity, nominal_solution,
                                          number_of_time_average_solves, parallel)
                try:
                    nconstraints = \
                        len([cnstrnt for cnstrnt in robust_model.get_robust_model().flat(constraintsets=False)])
                except AttributeError:
                    nconstraints = \
                        len([cnstrnt for cnstrnt in robust_model.get_robust_model()[-1].flat(constraintsets=False)])
                solutions[ind] = robust_model_solution
                solve_times[ind] = robust_model_solve_time
                simulations[ind] = simulation_results
                number_of_constraints[ind] = nconstraints
    return solutions, solve_times, simulations, number_of_constraints

def variable_goal_results(model, methods, deltas, number_of_iterations,
                                    min_num_of_linear_sections, max_num_of_linear_sections, verbosity,
                                    linearization_tolerance, number_of_time_average_solves,
                                    uncertainty_sets, nominal_solution, directly_uncertain_vars_subs, parallel=False):
    """
    Simulates a GP or SP model for a range of deltas in the goal programming form.
    i.e. maximizes uncertainty set size given an acceptable penalty delta on the objective.
    Outputs are dicts that have the key format: [deltaValue (float), methodName (string), uncertaintySet (string)]
    """
    solutions = {}
    solve_times = {}
    simulations = {}
    number_of_constraints = {}
    Gamma = Variable('\\Gamma', '-', 'Uncertainty bound')
    solBound = Variable('1+\\delta', '-', 'Acceptable optimal solution bound', fix = True)
    origcost = model.cost
    mGoal = Model(1 / Gamma, [model, origcost <= Monomial(nominal_solution(origcost)) * solBound, Gamma <= 1e30, solBound <= 1e30],
                  model.substitutions)
    for delta in deltas:
        mGoal.substitutions.update({'1+\\delta': 1 + delta})
        for method in methods:
            for uncertainty_set in uncertainty_sets:
                robust_goal_model = RobustModel(mGoal, uncertainty_set, gamma=Gamma, twoTerm=method['twoTerm'],
                                   boyd=method['boyd'], simpleModel=method['simpleModel'])

                robust_model_solution = robust_goal_model.robustsolve(verbosity=verbosity,
                                                             minNumOfLinearSections=min_num_of_linear_sections,
                                                             maxNumOfLinearSections=max_num_of_linear_sections,
                                                             linearizationTolerance=linearization_tolerance)

                robust_model_solve_time = get_avg_robust_solve_time(number_of_time_average_solves,
                                                        robust_goal_model, verbosity, min_num_of_linear_sections,
                                                        max_num_of_linear_sections,
                                                        linearization_tolerance, parallel)

                simulation_results = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                                  directly_uncertain_vars_subs,
                                                                  number_of_iterations,
                                                                  verbosity=verbosity, parallel=parallel)
                try:
                    nconstraints = \
                        len([cnstrnt for cnstrnt in robust_goal_model.get_robust_model().flat(constraintsets=False)])
                except AttributeError:
                    nconstraints = \
                        len([cnstrnt for cnstrnt in robust_goal_model.get_robust_model()[-1].flat(constraintsets=False)])
                ind = (delta, method['name'], uncertainty_set)
                solutions[ind] = robust_model_solution
                solve_times[ind] = robust_model_solve_time
                simulations[ind] = simulation_results
                number_of_constraints[ind] = nconstraints
    return solutions, solve_times, simulations, number_of_constraints

def filter_gamma_result_dict(dict, tupInd1, tupVal1, tupInd2, tupVal2):
    """
    Filters the items in outputs of variable_gamma_results or variable_goal_results
    with 2 out of 3 keys.
    """
    filteredResult = {}
    for i in sorted(dict.keys()):
        if i[tupInd1] == tupVal1 and i[tupInd2] == tupVal2:
            filteredResult[i] = dict[i]
    return filteredResult

def plot_gamma_result_PoFandCost(title, objective_name, objective_units, filteredResult, filteredSimulations, stddev = True):
    gammas = []
    objective_costs = []
    pofs = []
    objective_stddev = []
    for i in sorted(filteredResult.keys()):
        gammas.append(i[0])
        objective_stddev.append(filteredSimulations[i][2])
        objective_costs.append(filteredSimulations[i][1])
        pofs.append(filteredSimulations[i][0])
    if not stddev:
        objective_stddev = None
    objective_proboffailure_vs_gamma(gammas, objective_costs, objective_name, objective_units,
                                     np.min(objective_costs), np.max(objective_costs), pofs, title, objective_stddev)

def plot_goal_result_PoFandCost(title, objective_name, objective_varkey, objective_units, filteredResult, filteredSimulations):
    gammas = []
    objective_costs = []
    pofs = []
    objective_stddev = []
    for i in sorted(filteredResult.keys()):
        gammas.append(filteredResult[i]("\\Gamma").magnitude)
        objective_stddev.append(filteredSimulations[i][2])
        objective_costs.append(mag(filteredResult[i](objective_varkey)))
        pofs.append(filteredSimulations[i][0])
    objective_proboffailure_vs_gamma(gammas, objective_costs, objective_name, objective_units,
                                     np.min(objective_costs), np.max(objective_costs), pofs, title, None)

def print_variable_pwlsections_results(model, model_name, gamma, number_of_iterations,
                                                      numbers_of_linear_sections, linearization_tolerance,
                                                      verbosity, file_name, number_of_time_average_solves,
                                                      methods, uncertainty_sets, nominal_solution, nominal_solve_time,
                                                      nominal_number_of_constraints, directly_uncertain_vars_subs):
    """
    Simulates a model for different numbers of PWL sections for each posy.
    """

    f = open(file_name, 'w')
    f.write(model_name + ' Results: variable piecewise-linear sections\n')
    f.write('----------------------------------------------------------\n')
    cost_label = model.cost.str_without()
    split_label = cost_label.split(' ')
    capitalized_cost_label = ''
    for word in split_label:
        capitalized_cost_label += word.capitalize() + ' '
    f.write('Objective: %s\n' % capitalized_cost_label)
    f.write('Units: %s\n' % model.cost.units)
    f.write('----------------------------------------------------------\n')
    f.write('Number of iterations: %s\n' % number_of_iterations)
    f.write('gamma: %s\n' % gamma)
    f.write('Linearization tolerance: %s\n' % linearization_tolerance)
    f.write('----------------------------------------------------------\n')
    f.write('Nominal cost: %s\n' % nominal_solution['cost'])
    f.write('Average nominal solve time: %s\n' % nominal_solve_time)
    f.write('Nominal number of constraints: %s\n' % nominal_number_of_constraints)
    f.write('----------------------------------------------------------\n')

    for number_of_linear_sections in numbers_of_linear_sections:
        f.write('number of piecewise-linear sections = %s:\n' % number_of_linear_sections)
        for method in methods:
            f.write('\t' + method['name'] + ':\n')
            for uncertainty_set in uncertainty_sets:
                f.write('\t\t' + uncertainty_set + ':\n')
                robust_model, robust_model_solution, robust_model_solve_time, simulation_results = \
                    simulate_robust_model(model, method, uncertainty_set, gamma, directly_uncertain_vars_subs,
                                          number_of_iterations, linearization_tolerance,
                                          number_of_linear_sections,
                                          number_of_linear_sections, verbosity, nominal_solution,
                                          number_of_time_average_solves)
                print_simulation_results(robust_model, robust_model_solution, robust_model_solve_time,
                                         nominal_solve_time, nominal_number_of_constraints, nominal_solution['cost'],
                                         simulation_results, f)
    f.close()


def generate_model_properties(model, number_of_time_average_solves, number_of_iterations, distribution = None):
    """
    Solves the nominal model, and generates MC samples
    :param model: GP or SP model of interest, with uncertainties
    :param number_of_time_average_solves: # of solves for solution time analysis
    :param number_of_iterations: # of MC samples for simulation
    :param distribution: distribution for MC samples, 'normal' or 'uniform otherwise
    :return: nominal solution, nominal solve time, nominal number of constraints, and MC samples of uncertain inputs
    """
    try:
        nominal_solution = model.solve(verbosity=0)
        nominal_solve_time = nominal_solution['soltime']
        for i in range(number_of_time_average_solves-1):
            nominal_solve_time += model.solve(verbosity=0)['soltime']
    except InvalidGPConstraint:
        nominal_solution = model.localsolve(verbosity=0, iteration_limit=100)
        nominal_solve_time = nominal_solution['soltime']
        for i in range(number_of_time_average_solves-1):
            nominal_solve_time += model.localsolve(verbosity=0, iteration_limit=100)['soltime']
    nominal_solve_time = nominal_solve_time / number_of_time_average_solves

    if distribution == 'normal' or 'Gaussian':
        directly_uncertain_vars_subs = [{k: stats.truncnorm.rvs(-3. , 3. , loc=v, scale=(v*k.key.pr/300.))
                                         for k, v in list(model.substitutions.items())
                                     if k in model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                    for _ in range(number_of_iterations)]
    else:
        directly_uncertain_vars_subs = [{k: np.random.uniform(v - k.key.pr * v / 100.0, v + k.key.pr * v / 100.0)
                                     for k, v in list(model.substitutions.items())
                                     if k in model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                    for _ in range(number_of_iterations)]
    nominal_number_of_constraints = len([cs for cs in model.flat(constraintsets=False)])

    return nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs

if __name__ == '__main__':
    pass

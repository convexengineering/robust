import numpy as np
from gpkit.small_scripts import mag
from gpkit.exceptions import InvalidGPConstraint
from gpkit.small_scripts import mag

from robust.robust import RobustModel
from robust.robust_gp_tools import RobustGPTools
from robust.simulations.read_simulation_data import objective_proboffailure_vs_gamma

import matplotlib.pyplot as plt
import multiprocessing as mp

def simulate_robust_model(model, method, uncertainty_set, gamma, directly_uncertain_vars_subs,
                          number_of_iterations, linearization_tolerance, min_num_of_linear_sections,
                          max_num_of_linear_sections, verbosity, nominal_solution,
                          number_of_time_average_solves):
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

    processes = [mp.Process(target=robustmodel.robustsolve, args=()) for _ in xrange(number_of_time_average_solves-1)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    robust_model_solve_time = [p.wait(100) for p in processes]
    robust_model_solve_time = [p['soltime'] for p in processes]
    robust_model_solve_time = sum(robust_model_solve_time) / number_of_time_average_solves
    simulation_results = RobustGPTools.probability_of_failure(model, robust_model_solution,
                                                                  directly_uncertain_vars_subs,
                                                                  number_of_iterations,
                                                                  verbosity=0)
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


def generate_variable_gamma_results(model, model_name, gammas, number_of_iterations,
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
                                    uncertainty_sets, nominal_solution, directly_uncertain_vars_subs):
    solutions = {}
    solve_times = {}
    prob_of_failure = {}
    avg_cost = {}
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
                                          number_of_time_average_solves)
                try:
                    nconstraints = \
                        len([cnstrnt for cnstrnt in robust_model.get_robust_model().flat(constraintsets=False)])
                except AttributeError:
                    nconstraints = \
                        len([cnstrnt for cnstrnt in robust_model.get_robust_model()[-1].flat(constraintsets=False)])
                solutions[ind] = robust_model_solution
                solve_times[ind] = robust_model_solve_time
                prob_of_failure[ind] = simulation_results[0]
                avg_cost[ind] = simulation_results[1]
                number_of_constraints[ind] = nconstraints
    return solutions, solve_times, prob_of_failure, avg_cost, number_of_constraints

def filter_gamma_result_dict(dict, tupInd1, tupVal1, tupInd2, tupVal2):
    filteredResult = {}
    for i in sorted(dict.iterkeys()):
        if i[tupInd1] == tupVal1 and i[tupInd2] == tupVal2:
            filteredResult[i] = dict[i]
    return filteredResult

def plot_gamma_result_PoFandCost(title, objective_name, objective_units, filteredResult, filteredPoF, filteredCost):
    gammas = []
    avg_costs = []
    pofs = []
    for i in sorted(filteredResult.iterkeys()):
        gammas.append(i[0])
        avg_costs.append(mag(filteredCost[i]))
        pofs.append(filteredPoF[i])
    min_obj = min(avg_costs)
    max_obj = max(avg_costs)
    objective_proboffailure_vs_gamma(gammas, avg_costs, objective_name, objective_units, min_obj, max_obj, pofs, title)

def generate_variable_piecewiselinearsections_results(model, model_name, gamma, number_of_iterations,
                                                      numbers_of_linear_sections, linearization_tolerance,
                                                      verbosity, file_name, number_of_time_average_solves,
                                                      methods, uncertainty_sets, nominal_solution, nominal_solve_time,
                                                      nominal_number_of_constraints, directly_uncertain_vars_subs):

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
    try:
        nominal_solution = model.solve(verbosity=0)
        nominal_solve_time = nominal_solution['soltime']
        for i in xrange(number_of_time_average_solves-1):
            nominal_solve_time += model.solve(verbosity=0)['soltime']
    except InvalidGPConstraint:
        nominal_solution = model.localsolve(verbosity=0)
        nominal_solve_time = nominal_solution['soltime']
        for i in xrange(number_of_time_average_solves-1):
            nominal_solve_time += model.localsolve(verbosity=0)['soltime']
    nominal_solve_time = nominal_solve_time / number_of_time_average_solves

    if distribution == 'normal' or 'Gaussian':
        directly_uncertain_vars_subs = [{k: v + 1./300.*v*k.key.pr*np.random.standard_normal()
                                         for k, v in model.substitutions.items()
                                     if k in model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                    for _ in xrange(number_of_iterations)]
    else:
        directly_uncertain_vars_subs = [{k: np.random.uniform(v - k.key.pr * v / 100.0, v + k.key.pr * v / 100.0)
                                     for k, v in model.substitutions.items()
                                     if k in model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                    for _ in xrange(number_of_iterations)]
    nominal_number_of_constraints = len([cs for cs in model.flat(constraintsets=False)])

    return nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs

if __name__ == '__main__':
    pass

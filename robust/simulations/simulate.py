import numpy as np
from gpkit.small_scripts import mag
from gpkit.exceptions import InvalidGPConstraint

from robust.robust import RobustModel
from robust.robust_gp_tools import RobustGPTools


def simulate_robust_model(the_model, the_method, the_uncertainty_set, the_gamma, the_directly_uncertain_vars_subs,
                          the_number_of_iterations, the_linearization_tolerance, the_min_num_of_linear_sections,
                          the_max_num_of_linear_sections, the_verbosity, the_nominal_solution,
                          the_number_of_time_average_solves):
    print(
        the_method[
            'name'] + ' under ' + the_uncertainty_set + ' uncertainty set: \n' + '\t' + 'gamma = %s\n' % the_gamma
        + '\t' + 'minimum number of piecewise-linear sections = %s\n' % the_min_num_of_linear_sections
        + '\t' + 'maximum number of piecewise-linear sections = %s\n' % the_max_num_of_linear_sections)

    the_robust_model = RobustModel(the_model, the_uncertainty_set, gamma=the_gamma, twoTerm=the_method['twoTerm'],
                                   boyd=the_method['boyd'], simpleModel=the_method['simpleModel'],
                                   nominalsolve=the_nominal_solution)
    the_robust_model_solution = the_robust_model.robustsolve(verbosity=the_verbosity,
                                                             minNumOfLinearSections=the_min_num_of_linear_sections,
                                                             maxNumOfLinearSections=the_max_num_of_linear_sections,
                                                             linearizationTolerance=the_linearization_tolerance)

    the_robust_model_solve_time = the_robust_model_solution['soltime']
    for _ in xrange(the_number_of_time_average_solves-1):
        the_robust_model_solve_time += the_robust_model.robustsolve(verbosity=0)['soltime']
    the_robust_model_solve_time = the_robust_model_solve_time / the_number_of_time_average_solves
    the_simulation_results = RobustGPTools.probability_of_failure(the_model, the_robust_model_solution,
                                                                  the_directly_uncertain_vars_subs,
                                                                  the_number_of_iterations,
                                                                  verbosity=0)
    return the_robust_model, the_robust_model_solution, the_robust_model_solve_time, the_simulation_results


def print_simulation_results(the_robust_model, the_robust_model_solution, the_robust_model_solve_time,
                             the_nominal_model_solve_time, the_nominal_no_of_constraints, the_nominal_cost,
                             the_simulation_results, the_file_id):
    the_file_id.write('\t\t\t' + 'Probability of failure: %s\n' % the_simulation_results[0])
    the_file_id.write('\t\t\t' + 'Average performance: %s\n' % mag(the_simulation_results[1]))
    the_file_id.write('\t\t\t' + 'Relative average performance: %s\n' %
                      (mag(the_simulation_results[1]) / float(mag(the_nominal_cost))))
    the_file_id.write('\t\t\t' + 'Worst-case performance: %s\n' % mag(the_robust_model_solution['cost']))
    the_file_id.write('\t\t\t' + 'Relative worst-case performance: %s\n' %
                      (mag(the_robust_model_solution['cost']) / float(mag(the_nominal_cost))))
    try:
        number_of_constraints = \
            len([cnstrnt for cnstrnt in the_robust_model.get_robust_model().flat(constraintsets=False)])
    except AttributeError:
        number_of_constraints = \
            len([cnstrnt for cnstrnt in the_robust_model.get_robust_model()[-1].flat(constraintsets=False)])
    the_file_id.write('\t\t\t' + 'Number of constraints: %s\n' % number_of_constraints)
    the_file_id.write('\t\t\t' + 'Relative number of constraints: %s\n' %
                      (number_of_constraints / float(the_nominal_no_of_constraints)))
    the_file_id.write('\t\t\t' + 'Setup time: %s\n' % the_robust_model_solution['setuptime'])
    the_file_id.write('\t\t\t' + 'Relative setup time: %s\n' %
                      (the_robust_model_solution['setuptime'] / float(the_nominal_model_solve_time)))
    the_file_id.write('\t\t\t' + 'Solve time: %s\n' % the_robust_model_solve_time)
    the_file_id.write('\t\t\t' + 'Relative solve time: %s\n' %
                      (the_robust_model_solve_time / float(the_nominal_model_solve_time)))
    the_file_id.write('\t\t\t' + 'Number of linear sections: %s\n' % the_robust_model_solution['numoflinearsections'])
    the_file_id.write(
        '\t\t\t' + 'Upper lower relative error: %s\n' % mag(the_robust_model_solution['upperLowerRelError']))


def generate_variable_gamma_results(the_model, the_model_name, the_gammas, the_number_of_iterations,
                                    the_min_num_of_linear_sections, the_max_num_of_linear_sections, the_verbosity,
                                    the_linearization_tolerance, the_file_name, the_number_of_time_average_solves,
                                    methods, uncertainty_sets, nominal_solution, nominal_solve_time,
                                    nominal_number_of_constraints, directly_uncertain_vars_subs):

    f = open(the_file_name, 'w')
    f.write(the_model_name + ' Results: variable gamma\n')
    f.write('----------------------------------------------------------\n')
    cost_label = the_model.cost.str_without()
    split_label = cost_label.split(' ')
    capitalized_cost_label = ''
    for word in split_label:
        capitalized_cost_label += word.capitalize() + ' '
    f.write('Objective: %s\n' % capitalized_cost_label)
    f.write('Units: %s\n' % the_model.cost.units)
    f.write('----------------------------------------------------------\n')
    f.write('Number of iterations: %s\n' % the_number_of_iterations)
    f.write('Minimum number of piecewise-linear sections: %s\n' % the_min_num_of_linear_sections)
    f.write('Maximum number of piecewise-linear sections: %s\n' % the_max_num_of_linear_sections)
    f.write('Linearization tolerance: %s\n' % the_linearization_tolerance)
    f.write('----------------------------------------------------------\n')
    f.write('Nominal cost: %s\n' % nominal_solution['cost'])
    f.write('Average nominal solve time: %s\n' % nominal_solve_time)
    f.write('Nominal number of constraints: %s\n' % nominal_number_of_constraints)
    f.write('----------------------------------------------------------\n')

    for gamma in the_gammas:
        f.write('Gamma = %s:\n' % gamma)
        for method in methods:
            f.write('\t' + method['name'] + ':\n')
            for uncertainty_set in uncertainty_sets:
                f.write('\t\t' + uncertainty_set + ':\n')
                robust_model, robust_model_solution, robust_model_solve_time, simulation_results = \
                    simulate_robust_model(the_model, method, uncertainty_set, gamma, directly_uncertain_vars_subs,
                                          the_number_of_iterations, the_linearization_tolerance,
                                          the_min_num_of_linear_sections,
                                          the_max_num_of_linear_sections, the_verbosity, nominal_solution,
                                          the_number_of_time_average_solves)
                print_simulation_results(robust_model, robust_model_solution, robust_model_solve_time,
                                         nominal_solve_time, nominal_number_of_constraints, nominal_solution['cost'],
                                         simulation_results, f)
    f.close()


def generate_variable_piecewiselinearsections_results(the_model, the_model_name, the_gamma, the_number_of_iterations,
                                                      the_numbers_of_linear_sections, the_linearization_tolerance,
                                                      the_verbosity, the_file_name, the_number_of_time_average_solves,
                                                      methods, uncertainty_sets, nominal_solution, nominal_solve_time,
                                                      nominal_number_of_constraints, directly_uncertain_vars_subs):

    f = open(the_file_name, 'w')
    f.write(the_model_name + ' Results: variable piecewise-linear sections\n')
    f.write('----------------------------------------------------------\n')
    cost_label = the_model.cost.str_without()
    split_label = cost_label.split(' ')
    capitalized_cost_label = ''
    for word in split_label:
        capitalized_cost_label += word.capitalize() + ' '
    f.write('Objective: %s\n' % capitalized_cost_label)
    f.write('Units: %s\n' % the_model.cost.units)
    f.write('----------------------------------------------------------\n')
    f.write('Number of iterations: %s\n' % the_number_of_iterations)
    f.write('gamma: %s\n' % the_gamma)
    f.write('Linearization tolerance: %s\n' % the_linearization_tolerance)
    f.write('----------------------------------------------------------\n')
    f.write('Nominal cost: %s\n' % nominal_solution['cost'])
    f.write('Average nominal solve time: %s\n' % nominal_solve_time)
    f.write('Nominal number of constraints: %s\n' % nominal_number_of_constraints)
    f.write('----------------------------------------------------------\n')

    for number_of_linear_sections in the_numbers_of_linear_sections:
        f.write('number of piecewise-linear sections = %s:\n' % number_of_linear_sections)
        for method in methods:
            f.write('\t' + method['name'] + ':\n')
            for uncertainty_set in uncertainty_sets:
                f.write('\t\t' + uncertainty_set + ':\n')
                robust_model, robust_model_solution, robust_model_solve_time, simulation_results = \
                    simulate_robust_model(the_model, method, uncertainty_set, the_gamma, directly_uncertain_vars_subs,
                                          the_number_of_iterations, the_linearization_tolerance,
                                          number_of_linear_sections,
                                          number_of_linear_sections, the_verbosity, nominal_solution,
                                          the_number_of_time_average_solves)
                print_simulation_results(robust_model, robust_model_solution, robust_model_solve_time,
                                         nominal_solve_time, nominal_number_of_constraints, nominal_solution['cost'],
                                         simulation_results, f)
    f.close()


def generate_model_properties(the_model, the_number_of_time_average_solves, the_number_of_iterations):
    try:
        nominal_solution = the_model.solve(verbosity=0)
        nominal_solve_time = nominal_solution['soltime']
        for i in xrange(the_number_of_time_average_solves-1):
            nominal_solve_time += the_model.solve(verbosity=0)['soltime']
    except InvalidGPConstraint:
        nominal_solution = the_model.localsolve(verbosity=0)
        nominal_solve_time = nominal_solution['soltime']
        for i in xrange(the_number_of_time_average_solves-1):
            nominal_solve_time += the_model.localsolve(verbosity=0)['soltime']
    nominal_solve_time = nominal_solve_time / the_number_of_time_average_solves

    directly_uncertain_vars_subs = [{k: np.random.uniform(v - k.key.pr * v / 100.0, v + k.key.pr * v / 100.0)
                                     for k, v in the_model.substitutions.items()
                                     if k in the_model.varkeys and RobustGPTools.is_directly_uncertain(k)}
                                    for _ in xrange(the_number_of_iterations)]
    nominal_number_of_constraints = len([cs for cs in the_model.flat(constraintsets=False)])

    return nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs

if __name__ == '__main__':
    pass

from gpkit import Variable, Model, SignomialsEnabled, units
from robust.simulations import simulate, read_simulation_data
import os

from robust.testing.models import simple_ac

if __name__ == '__main__':
    model = simple_ac()
    number_of_time_average_solves = 3  # 100
    number_of_iterations = 15  # 1000
    nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
        simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)
    model_name = 'Signomial Simple Flight'
    gammas = [0.0, 0.3, 1]  # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    min_num_of_linear_sections = 3
    max_num_of_linear_sections = 99
    linearization_tolerance = 1e-3
    verbosity = 1

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True}]
    uncertainty_sets = ['box', 'elliptical']

    model = simple_ac()

    variable_gamma_file_name = os.path.dirname(__file__) + '/simulation_data_variable_gamma.txt'
    simulate.print_variable_gamma_results(model, model_name, gammas, number_of_iterations,
                                             min_num_of_linear_sections,
                                             max_num_of_linear_sections, verbosity, linearization_tolerance,
                                             variable_gamma_file_name, number_of_time_average_solves, methods,
                                             uncertainty_sets, nominal_solution, nominal_solve_time,
                                             nominal_number_of_constraints, directly_uncertain_vars_subs)

    gamma = 1
    numbers_of_linear_sections = [40, 60]  # [12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 36, 44, 52, 60, 70, 80]

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False}]
    uncertainty_sets = ['box', 'elliptical']

    variable_pwl_file_name = os.path.dirname(__file__) + '/simulation_data_variable_pwl.txt'
    simulate.print_variable_pwlsections_results(model, model_name, gamma, number_of_iterations,
                                                               numbers_of_linear_sections, linearization_tolerance,
                                                               verbosity, variable_pwl_file_name,
                                                               number_of_time_average_solves, methods, uncertainty_sets,
                                                               nominal_solution, nominal_solve_time,
                                                               nominal_number_of_constraints,
                                                               directly_uncertain_vars_subs)

    file_path_gamma = os.path.dirname(__file__) + '/simulation_data_variable_gamma.txt'
    file_path_pwl = os.path.dirname(__file__) + '/simulation_data_variable_pwl.txt'
    read_simulation_data.generate_all_plots(file_path_gamma, file_path_pwl)

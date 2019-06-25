from gassolar.solar.solar import Mission as solar_mike
import os

from robust.simulations import simulate, read_simulation_data


def mike_solar_model(lat):
    model = solar_mike(latitude=lat)
    model.cost = model["W_{total}"]
    return model


if __name__ == '__main__':
    model = mike_solar_model(20)
    number_of_time_average_solves = 30
    number_of_iterations = 300
    nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
        simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)
    model_name = 'Solar Model'
    gammas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.29, 0.33, 0.39, 0.45, 0.6, 0.87, 1]
    gammas = [0.7 * i for i in gammas]
    min_num_of_linear_sections = 10
    max_num_of_linear_sections = 50
    linearization_tolerance = 1e-3
    verbosity = 0

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linear. Perts.', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Cons.', 'twoTerm': False, 'boyd': False, 'simpleModel': True},
               {'name': 'Two Term', 'twoTerm': False, 'boyd': True, 'simpleModel': False}]
    uncertainty_sets = ['box', 'elliptical']

    model = mike_solar_model(20)

    variable_gamma_file_name = os.path.dirname(__file__) + '/simulation_data_variable_gamma.txt'
    simulate.print_variable_gamma_results(model, model_name, gammas, number_of_iterations,
                                             min_num_of_linear_sections, max_num_of_linear_sections, verbosity,
                                             linearization_tolerance, variable_gamma_file_name,
                                             number_of_time_average_solves, methods, uncertainty_sets, nominal_solution,
                                             nominal_solve_time, nominal_number_of_constraints,
                                             directly_uncertain_vars_subs)

    gamma = 0.7 * 1
    numbers_of_linear_sections = [12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 36, 44, 52, 60, 70, 80]

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linear. Perts.', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Two Term', 'twoTerm': False, 'boyd': True, 'simpleModel': False}]
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

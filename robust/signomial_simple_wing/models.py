from gpkit import Variable, Model, SignomialsEnabled, VarKey, units, Vectorize
import numpy as np
from robust.simulations import simulate, read_simulation_data
import os

from gpkitmodels.SP.SimPleAC.SimPleAC_mission import SimPleAC, Mission

def example_sp():
    x = Variable('x')
    y = Variable('y')
    a = Variable('a', 1, pr=10)
    b = Variable('b', 1, pr=10)
    constraints = []
    with SignomialsEnabled():
        constraints = constraints + [x >= 1 - a * y, b * y <= 0.1]
    return Model(x, constraints)

def simple_wing_sp():
    model = Mission(SimPleAC(), 4)
    model.substitutions.update({
        'h_{cruise_m}'   :5000*units('m'),
        'Range_m'        :3000*units('km'),
        'W_{p_m}'        :6250*units('N'),
        'C_m'            :120*units('1/hr'),
        'V_{min_m}'      :25*units('m/s'),
        'T/O factor_m'   :2,
    })
    c = Variable('c','-','model cost')
    model = Model(c, [model, c >= model['W_{f_m}']*units('1/N') + model['C_m']*model['t_m']])
    return model

if __name__ == '__main__':
    model = simple_wing_sp()
    number_of_time_average_solves = 100
    number_of_iterations = 1000
    nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
        simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)
    model_name = 'Signomial Simple Flight'
    gammas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    min_num_of_linear_sections = 3
    max_num_of_linear_sections = 99
    linearization_tolerance = 1e-3
    verbosity = 0

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True}]
    uncertainty_sets = ['box', 'elliptical']

    model = simple_wing_sp()

    variable_gamma_file_name = os.path.dirname(__file__) + '/simulation_data_variable_gamma.txt'
    simulate.generate_variable_gamma_results(model, model_name, gammas, number_of_iterations,
                                             min_num_of_linear_sections,
                                             max_num_of_linear_sections, verbosity, linearization_tolerance,
                                             variable_gamma_file_name, number_of_time_average_solves, methods,
                                             uncertainty_sets, nominal_solution, nominal_solve_time,
                                             nominal_number_of_constraints, directly_uncertain_vars_subs)

    gamma = 1
    numbers_of_linear_sections = [12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 36, 44, 52, 60, 70, 80]

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False}]
    uncertainty_sets = ['box', 'elliptical']

    variable_pwl_file_name = os.path.dirname(__file__) + '/simulation_data_variable_pwl.txt'
    simulate.generate_variable_piecewiselinearsections_results(model, model_name, gamma, number_of_iterations,
                                                               numbers_of_linear_sections, linearization_tolerance,
                                                               verbosity, variable_pwl_file_name,
                                                               number_of_time_average_solves, methods, uncertainty_sets,
                                                               nominal_solution, nominal_solve_time,
                                                               nominal_number_of_constraints, directly_uncertain_vars_subs)

    file_path_gamma = os.path.dirname(__file__) + '/simulation_data_variable_gamma.txt'
    file_path_pwl = os.path.dirname(__file__) + '/simulation_data_variable_pwl.txt'
    read_simulation_data.generate_all_plots(file_path_gamma, file_path_pwl)

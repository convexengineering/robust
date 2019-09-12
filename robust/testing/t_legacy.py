"""Testing legacy code"""
import numpy as np
import os

import unittest
from gpkit import Model
from gpkit.tests.helpers import run_tests
from gpkit.constraints.bounded import Bounded

from robust.testing.models import simple_wing
from robust.simulations import simulate
from robust.simulations import read_simulation_data

class TestLegacy(unittest.TestCase):
    def test_simple_wing(self):
        model = Model(simple_wing().cost, Bounded(simple_wing()), simple_wing().substitutions)
        number_of_time_average_solves = 3
        number_of_iterations = 10
        nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
            simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)
        model_name = 'Simple_Wing'
        gammas = np.linspace(0.0, 1.0, 3)
        min_num_of_linear_sections = 3
        max_num_of_linear_sections = 99
        linearization_tolerance = 1e-4
        verbosity = 0

        methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
                   {'name': 'Linear. Perts.', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
                   {'name': 'Simple Cons.', 'twoTerm': False, 'boyd': False, 'simpleModel': True},
                   {'name': 'Two Term', 'twoTerm': False, 'boyd': True, 'simpleModel': False}]
        uncertainty_sets = ['box', 'elliptical']

        variable_gamma_file_name = os.path.dirname(__file__) + '/' + \
                                   model_name + '_simulation_data_variable_gamma.txt'
        simulate.print_variable_gamma_results(model, model_name, gammas, number_of_iterations,
                                                 min_num_of_linear_sections,
                                                 max_num_of_linear_sections, verbosity, linearization_tolerance,
                                                 variable_gamma_file_name, number_of_time_average_solves, methods,
                                                 uncertainty_sets, nominal_solution, nominal_solve_time,
                                                 nominal_number_of_constraints, directly_uncertain_vars_subs)

        gamma = 1.
        numbers_of_linear_sections = [12, 20, 30, 44, 60, 80]

        methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
                   {'name': 'Linear. Perts.', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
                   {'name': 'Two Term', 'twoTerm': False, 'boyd': True, 'simpleModel': False}]
        uncertainty_sets = ['box', 'elliptical']

        variable_pwl_file_name = os.path.dirname(__file__) + '/' + \
                                 model_name + '_simulation_data_variable_pwl.txt'
        simulate.print_variable_pwlsections_results(model, model_name, gamma, number_of_iterations,
                                                                   numbers_of_linear_sections, linearization_tolerance,
                                                                   verbosity, variable_pwl_file_name,
                                                                   number_of_time_average_solves, methods, uncertainty_sets,
                                                                   nominal_solution, nominal_solve_time,
                                                                   nominal_number_of_constraints, directly_uncertain_vars_subs)

        read_simulation_data.generate_all_plots(variable_gamma_file_name,
                                                variable_pwl_file_name)

TESTS = [TestLegacy]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

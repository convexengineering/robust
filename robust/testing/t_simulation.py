import numpy as np

import unittest
from gpkit.tests.helpers import run_tests

from robust.simulations import simulate
from robust.testing.models import gp_test_model

class TestSimulation(unittest.TestCase):
    def test_simulate(self):
        model = gp_test_model()
        number_of_time_average_solves = 3
        number_of_iterations = 10
        uncertainty_sets = ['box']
        methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False}]
        nGammas = 3
        gammas = np.linspace(0, 1.0, nGammas)
        min_num_of_linear_sections = 3
        max_num_of_linear_sections = 99
        linearization_tolerance = 1e-4
        verbosity = 0
        parallel = False

        nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
            simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)

        solutions, solve_times, simulation_results, number_of_constraints = simulate.variable_gamma_results(
            model, methods, gammas, number_of_iterations,
            min_num_of_linear_sections,
            max_num_of_linear_sections, verbosity, linearization_tolerance,
            number_of_time_average_solves,
            uncertainty_sets, nominal_solution, directly_uncertain_vars_subs, parallel=parallel)

        # Checking probability of failure is 0 for gamma=1
        robustifiedResult = simulation_results[sorted(simulation_results.keys())[-1]]
        self.assertEqual(robustifiedResult[0], 0.)

TESTS = [TestSimulation]

def test():
    run_tests(TESTS)

if __name__ == '__main__':
    test()

import numpy as np
import unittest
from gpkit.tests.helpers import run_tests
from gpkit import Model
from gpkit.constraints.bounded import Bounded

from robust.simulations import simulate
from robust.testing.models import simple_ac

class TestSimulation(unittest.TestCase):
    def test_simulate(self):
        model = simple_ac()
        number_of_time_average_solves = 3
        number_of_iterations = 20
        uncertainty_sets = ['elliptical']
        methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False}]
        nGammas = 3
        gammas = np.linspace(0, 1.0, nGammas)
        min_num_of_linear_sections = 3
        max_num_of_linear_sections = 99
        linearization_tolerance = 1e-4
        verbosity = 0

        nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
            simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)

        _, _, simulation_results, _ = simulate.variable_gamma_results(
            model, methods, gammas, number_of_iterations,
            min_num_of_linear_sections,
            max_num_of_linear_sections, verbosity, linearization_tolerance,
            number_of_time_average_solves,
            uncertainty_sets, nominal_solution, directly_uncertain_vars_subs, parallel=False)

        # Checking probability of failure is 0 for gamma=1
        keys = sorted(simulation_results.keys())
        self.assertEqual(simulation_results[keys[-1]][0], 0.)

        # Then test in parallel, and compare time results
        # _, _, parallel_simulation_results, _ = simulate.variable_gamma_results(
        #     model, methods, gammas, number_of_iterations,
        #     min_num_of_linear_sections,
        #     max_num_of_linear_sections, verbosity, linearization_tolerance,
        #     number_of_time_average_solves,
        #     uncertainty_sets, nominal_solution, directly_uncertain_vars_subs, parallel=True)

        # Checking mean of simulation results is equal for gamma=0
        # self.assertAlmostEqual(simulation_results[keys[0]][1], parallel_simulation_results[keys[0]][1])

TESTS = [TestSimulation]

def test():
    run_tests(TESTS)

if __name__ == '__main__':
    test()

import numpy as np
import os
import unittest
from gpkit.tests.helpers import run_tests
from gpkit.small_scripts import mag

from robust.margin import MarginModel
from robust.robust import RobustModel
from robust.simulations import simulate
from robust.testing.models import simple_ac

class TestSimulation(unittest.TestCase):
    def test_simulate(self):
        model = simple_ac()
        model.cost = model['c']
        number_of_time_average_solves = 3
        number_of_iterations = 10
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

    def test_table_diff(self):
        m = simple_ac()
        m.cost = m['c']
        sol = m.localsolve(verbosity=0)
        gamma = 0.5

        # Model with margins
        mm = MarginModel(m, gamma=gamma)
        msol = mm.localsolve(verbosity=0)
        # Model with box uncertainty
        bm = RobustModel(m, 'box', gamma=gamma, twoTerm = True, boyd = False, simpleModel = False)
        bsol = bm.robustsolve(verbosity=0)
        # Model with elliptical uncertainty
        em = RobustModel(m, 'elliptical', gamma=gamma, twoTerm = True, boyd = False, simpleModel = False)
        esol = em.robustsolve(verbosity=0)

        soltab = [sol, msol, bsol, esol]

        origfilename = os.path.dirname(__file__) + '/' + 'diffs/test_table.txt'
        filename = os.path.dirname(__file__) + '/' + 'diffs/test_table_diff.txt'
        f = open(filename, 'w+')

        for i in ['L/D', 'A', 'Re', 'S', 'V', 't_s', 'W_w', 'W_{w_{strc}}', 'W_{w_{surf}}',
              'W_{fuse}','V_{f_{avail}}', 'V_{f_{fuse}}', 'V_{f_{wing}}']:
            f.write(i)
            if i in ['L/D', 'Re', 'V']:
                a = [mag(np.mean(s(i))) for s in soltab]
            elif i in ['t_s']:
                a = [mag(np.sum(s(i))) for s in soltab]
            else:
                a = [mag(s(i))  for s in soltab]
            for j in range(len(a)):
                if a[j] <= 1e-5:
                    a[j] = 0.
            f.write(''.join([" & " + str(np.format_float_scientific(j, precision=2)) for j in a]))
            f.write('\n')
        f.write('cost ')
        f.write(' '.join(["& " + str(np.format_float_scientific(i['cost'], precision=2)) for i in soltab]))
        f.write('\n')
        f.close()

        self.assertEqual(open(origfilename, 'r').readlines(), open(filename, 'r').readlines())

TESTS = [TestSimulation]

def test():
    run_tests(TESTS)

if __name__ == '__main__':
    test()

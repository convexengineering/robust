from RobustGP import RobustGPModel
from gpkit import Model
import numpy as np


class RobustSPModel:
    sequence_of_rgps = []
    number_of_rgp_approximations = None
    original_model = None
    gamma = None
    sp_rel_tol = None
    initial_guess = None
    initial_cost = None
    type_of_uncertainty_set = None
    minimum_number_of_pwl_functions = None
    linearization_tolerance = None
    simple_model = None
    number_of_regression_points = None
    linearize_two_term = None
    enable_sp = None
    two_term = None
    simple_two_term = None
    maximum_number_of_permutations = None

    def __init__(self, model, gamma, type_of_uncertainty_set, minimum_number_of_pwl_functions=5,
                 linearization_tolerance=0.001, simple_model=False, number_of_regression_points=2,
                 linearize_two_term=True, enable_sp=True, two_term=None, simple_two_term=True,
                 maximum_number_of_permutations=30, sp_rel_tol=1e-4):
        self.original_model = model
        self.gamma = gamma
        self.sp_rel_tol = sp_rel_tol
        self.type_of_uncertainty_set = type_of_uncertainty_set
        self.minimum_number_of_pwl_functions = minimum_number_of_pwl_functions
        self.linearization_tolerance = linearization_tolerance
        self.simple_model = simple_model
        self.number_of_regression_points = number_of_regression_points
        self.linearize_two_term = linearize_two_term
        self.enable_sp = enable_sp
        self.two_term = two_term
        self.simple_two_term = simple_two_term
        self.maximum_number_of_permutations = maximum_number_of_permutations

        try:
            nominal_solution = model.localsolve(verbosity=0)
        except:
            nominal_solution = model.solve(verbosity=0)
        self.initial_guess = nominal_solution.get('variables')
        self.initial_cost = nominal_solution['cost']

    def localsolve(self, verbosity=0):
        try:
            old_cost = self.initial_cost.m
        except:
            old_cost = self.initial_cost
        solution = self.initial_guess

        rgp_solve = None

        new_cost = old_cost * (1 + 2 * self.sp_rel_tol)

        while (np.abs(old_cost - new_cost) / old_cost) > self.sp_rel_tol:
            local_gp_approximation = Model(self.original_model.cost, self.original_model.as_gpconstr(solution))

            robust_local_approximation = RobustGPModel(local_gp_approximation, self.gamma, self.type_of_uncertainty_set,
                                                       self.minimum_number_of_pwl_functions, self.linearization_tolerance,
                                                       self.simple_model, self.number_of_regression_points,
                                                       self.linearize_two_term, self.enable_sp, False, self.two_term,
                                                       self.simple_two_term, self.maximum_number_of_permutations)

            self.sequence_of_rgps.append(robust_local_approximation)

            try:
                rgp_solve = robust_local_approximation.solve(verbosity=verbosity)
            except:
                rgp_solve = robust_local_approximation.localsolve(verbosity=verbosity)

            solution = rgp_solve.get('variables')
            old_cost = new_cost

            try:
                new_cost = rgp_solve['cost'].m
            except:
                new_cost = rgp_solve['cost']
        self.number_of_rgp_approximations = len(self.sequence_of_rgps)
        return rgp_solve

from RobustGP import RobustGPModel
from gpkit import Model

import numpy as np


class RobustSPModel:
    sequence_of_rgps = None
    number_of_rgp_approximations = None
    original_model = None
    gamma = None
    sp_rel_tol = None
    initial_cost = None
    type_of_uncertainty_set = None
    simple_model = None
    number_of_regression_points = None
    boyd = None
    linearize_two_term = None
    enable_sp = None
    two_term = None
    simple_two_term = None
    maximum_number_of_permutations = None
    nominal_solution = None
    smart_two_term_choose = None

    r_min = None
    r_max = None
    linearization_tolerance = None

    r = None
    solve_time = None

    lower_approximation_used = False

    def __init__(self, model, gamma, type_of_uncertainty_set, simple_model=False, number_of_regression_points=2,
                 linearize_two_term=True, enable_sp=True, boyd=False, two_term=False, simple_two_term=False,
                 smart_two_term_choose=False, maximum_number_of_permutations=30, sp_rel_tol=1e-4):
        self.original_model = model
        self.gamma = gamma
        self.sp_rel_tol = sp_rel_tol
        self.type_of_uncertainty_set = type_of_uncertainty_set
        self.simple_model = simple_model
        self.number_of_regression_points = number_of_regression_points
        self.boyd = boyd
        self.linearize_two_term = linearize_two_term
        self.enable_sp = enable_sp
        self.two_term = two_term
        self.simple_two_term = simple_two_term
        self.smart_two_term_choose = smart_two_term_choose
        self.maximum_number_of_permutations = maximum_number_of_permutations

        nominal_solve = model.localsolve(verbosity=0)

        self.nominal_solution = nominal_solve.get('variables')
        self.nominal_cost = nominal_solve['cost']

    def localsolve(self, verbosity=0, r_min=12, r_max=20, linearization_tolerance=0.01):
        try:
            old_cost = self.nominal_cost.m
        except:
            old_cost = self.nominal_cost
        solution = self.nominal_solution

        rgp_solve = None

        new_cost = old_cost * (1 + 2 * self.sp_rel_tol)

        self.sequence_of_rgps = []

        while (np.abs(old_cost - new_cost) / old_cost) > self.sp_rel_tol:
            local_gp_approximation = Model(self.original_model.cost, self.original_model.as_gpconstr(solution))

            robust_local_approximation = RobustGPModel(local_gp_approximation, self.gamma, self.type_of_uncertainty_set,
                                                       self.simple_model, self.number_of_regression_points,
                                                       self.linearize_two_term, self.enable_sp, self.boyd, self.two_term,
                                                       self.simple_two_term, self.smart_two_term_choose,
                                                       self.maximum_number_of_permutations)

            rgp_solve = robust_local_approximation.solve(verbosity, r_min, r_max, linearization_tolerance)
            r_min = robust_local_approximation.r
            self.sequence_of_rgps.append(robust_local_approximation.robust_model)

            solution = rgp_solve.get('variables')
            old_cost = new_cost

            try:
                new_cost = rgp_solve['cost'].m
            except:
                new_cost = rgp_solve['cost']
        self.number_of_rgp_approximations = len(self.sequence_of_rgps)
        return rgp_solve

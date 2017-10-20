from RobustGP import RobustGPModel, RobustGPSetting
from gpkit import Model

import numpy as np


class RobustSPSetting(RobustGPSetting):
    def __init__(self, **options):
        if 'SPRelativeTolerance' in options:
            RobustGPSetting.__init__(self, **options)
        else:
            RobustGPSetting.__init__(self, SPRelativeTolerance=1e-4, **options)


class RobustSPModel:

#    ready_gp_constraints = []
#    tractable_posynomials = []
#    to_linearize_posynomials = []
#    large_posynomials = []

    lower_approximation_used = False

    def __init__(self, model, type_of_uncertainty_set, **options):
        self.original_model = model
        self.type_of_uncertainty_set = type_of_uncertainty_set
        self.setting = RobustSPSetting(**options)
        self.options = options
        nominal_solve = model.localsolve(verbosity=0)

        self.nominal_solution = nominal_solve.get('variables')
        self.nominal_cost = nominal_solve['cost']

        self.sequence_of_rgps = []
        self.number_of_rgp_approximations = None
        self. r = None
        # self.solve_time = None

    def localsolve(self, verbosity=0, **options):
        try:
            old_cost = self.nominal_cost.m
        except:
            old_cost = self.nominal_cost
        solution = self.nominal_solution

        rgp_solve = None

        new_cost = old_cost * (1 + 2 * self.setting.get('SPRelativeTolerance'))

        self.sequence_of_rgps = []

        while (np.abs(old_cost - new_cost) / old_cost) > self.setting.get('SPRelativeTolerance'):
            # print("---------------------------------------------------------------------")
            local_gp_approximation = Model(self.original_model.cost, self.original_model.as_gpconstr(solution))

            robust_local_approximation = RobustGPModel.\
                construct(local_gp_approximation, self.type_of_uncertainty_set, **self.options)
            rgp_solve = robust_local_approximation.solve(verbosity=verbosity, **options)
            options['minNumOfLinearSections'] =  robust_local_approximation.r
            self.sequence_of_rgps.append(robust_local_approximation )

            solution = rgp_solve.get('variables')
            old_cost = new_cost

            try:
                new_cost = rgp_solve['cost'].m
            except:
                new_cost = rgp_solve['cost']
        self.number_of_rgp_approximations = len(self.sequence_of_rgps)
        return rgp_solve

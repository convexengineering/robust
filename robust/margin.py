import numpy as np

from gpkit import Model
from gpkit.small_scripts import mag

from .robust import RobustModel
from.robust_gp_tools import RobustGPTools

class MarginSetting(object):
    def __init__(self, **options):
        self._options = {'gamma': 1}

        for key, value in options.items():
            self._options[key] = value

    def get(self, option_name):
        return self._options[option_name]

    def set(self, option_name, value):
        self._options[option_name] = value

class MarginModel(Model):
    """
    MarginModel extends gpkit.Model by adding margins to
    Model fixed variables with pr.
    It uses the local sensitivities of the Model solution to
    determine the direction of perturbation of fixed variables.
    """
    def setup(self, model, **options):
        self.nominal_model = model
        self.setting = MarginSetting(**options)

        # Solving the nominal model
        if model.solution:
            self.nominal_solve = model.solution
        else:
            self.nominal_solve = RobustModel.internalsolve(model, verbosity=0)
        self.nominal_solution = self.nominal_solve.get('variables')
        self.nominal_cost = self.nominal_solve['cost']

        # Determining margins
        self.substitutions = {k: v + self.setting.get("gamma") *
                                     np.sign(mag(self.nominal_solve['sensitivities']['constants'][k.key]))*k.key.pr * v / 100.0
                                     for k, v in model.substitutions.items()
                                     if k in model.varkeys and RobustGPTools.is_directly_uncertain(k)}
        self.cost = model.cost
        return Model(self.cost, model, self.substitutions)

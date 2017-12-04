import GPModels as models
from gpkit import Model, Variable, ConstraintSet


class trial(Model):
    def setup(self, model, solution):
        self.cost = model.cost
        subs = {k: v for k, v in solution['freevariables'].items() if not hasattr(v, "__call__")}
        slack = Variable
        ConstraintSet([slack <= 10, slack <= 10])
        return [model, ConstraintSet([slack <= 10, slack >= 1])], subs

solar = models.mike_solar_model()
solution = solar.solve()
new_solar = trial(solar, solution)
new_solution = new_solar.solve()

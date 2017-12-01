import GPModels as models
from gpkit import Model, Variable


class trial(Model):
    def setup(self, model, solution):
        self.cost = model.cost
        subs = {k: v for k, v in solution['freevariables'].items() if k.key.shape is None}
        return [model], subs

solar = models.mike_solar_model()
solution = solar.solve()
new_solar = trial(solar, solution)
new_solution = new_solar.solve()

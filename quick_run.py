from robust.robust import RobustModel
from robust.signomial_simple_wing.models import simple_ac
from robust.simulations.simulate import generate_model_properties
from robust.robust_gp_tools import RobustGPTools

model = simple_ac()
model_solution, _, _, directly_uncertain_vars_subs = generate_model_properties(model, 1, 1)
robust_model = RobustModel(model, 'box')
robust_model_solution = robust_model.robustsolve(verbosity=0)

designed_model = RobustGPTools.DesignedModel(model, robust_model_solution, directly_uncertain_vars_subs[0])
designed_model.localsolve()
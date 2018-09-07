from robust.robust import RobustModel
from robust.signomial_simple_wing.models import simple_ac
from robust.simulations.simulate import generate_model_properties
from robust.robust_gp_tools import RobustGPTools

model = simple_ac()  # the SimPleAC model
model_solution, _, _, directly_uncertain_vars_subs = generate_model_properties(model, 1, 1)  # solving the SimPleAC
# and generating a realization of the uncertain parameters
robust_model = RobustModel(model, 'box')  # generates the robust model
robust_model_solution = robust_model.robustsolve(verbosity=0)  # solve the robust model

# Below, we are testing the SimPleAC model against failure. The free variables with attribute fixed=True are fixed using
# the robust solution, and the uncertain parameters are replaced by the realizations generated above. The model is then
# solved to check if a feasible solution exists.
designed_model = RobustGPTools.DesignedModel(model, robust_model_solution, directly_uncertain_vars_subs[0])
designed_model.localsolve()
# The designed model is not converging although it has the same set of constraints as the original model by with
# different substitutions
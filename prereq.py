from signomial_simple_wing.models import simple_wing_sp
from robust_gp_tools import RobustGPTools
from simulations.simulate import generate_model_properties

simp = simple_wing_sp()
_, _, _, uncertain_vars = generate_model_properties(simp, 1, 1)
sol = simp.localsolve(verbosity=0)
new_model = RobustGPTools.DesignedModel(simp, sol, uncertain_vars[0])
try:
    new_model.localsolve()
except ValueError:
    pass
_ = simp.localsolve()

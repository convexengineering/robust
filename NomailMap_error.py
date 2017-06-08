from RobustGP import RobustGPModel
from gpkit import Model
import numpy as np
import GPModels

simp = GPModels.simpleWingSP()
solution = simp.localsolve().get('variables')
print(solution)
local_gp_approximation = Model(simp.cost, simp.as_gpconstr(solution))

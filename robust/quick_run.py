from robust import RobustModel
from solar_model.models import mike_solar_model

model = mike_solar_model(20)
robust_model = RobustModel(model, 'elliptical', twoTerm=False, gamma=0.8)
robust_model_solution = robust_model.robustsolve(minNumOfLinearSections=30, maxNumOfLinearSections=30)

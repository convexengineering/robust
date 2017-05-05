import GPModels as GPM
import RobustGP as RGP

model = GPM.testModel()
robustModel = RGP.robustModelEllipticalUncertainty(model, linearizeTwoTerm=False, enableSP=True, numberOfRegressionPoints=2)[0]
GPM.solveModel(robustModel);
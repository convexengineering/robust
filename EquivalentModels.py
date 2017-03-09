from gpkit import  Model
import TwoTermApproximation as TTA
import LinearizeTwoTermPosynomials as LTTP
import EquivalentPosynomial as EP

def uncertainModelVariables(model):
    subsVars = list(model.substitutions.keys())
    uncertainVars = [var for var in subsVars if var.key.pr != None]
    return uncertainVars

def equivalentModel(model, dependentUncertainties, coupled = True):
    dataConstraints = []
    noDataConstraints = []
    uncertainVars = uncertainModelVariables(model)
    for i, p in enumerate(model.as_posyslt1()):
        (noData, data) = EP.equivalentPosynomial(p,uncertainVars,i,coupled,dependentUncertainties)
        dataConstraints = dataConstraints + data
        noDataConstraints = noDataConstraints + noData
    numberOfNoDataConstraints = len(noDataConstraints)
    return Model(model.cost,[noDataConstraints, dataConstraints]), numberOfNoDataConstraints
    
def twoTermModel(model,dependentUncertainties):
    equiModel, numberOfNoDataConstraints = equivalentModel(model,dependentUncertainties,True)
    dataConstraints = []
    noDataConstraints = []
    uncertainVars = uncertainModelVariables(model)
    for i, p in enumerate(equiModel.as_posyslt1()):   
        if i < numberOfNoDataConstraints:
            noDataConstraints =  noDataConstraints + [p <= 1]
        else:
            (noData, data) = TTA.twoTermExpApproximation(p,uncertainVars,i)#data = twoTermExpApproximationBoyd(p,i)#
            dataConstraints = dataConstraints + data
            noDataConstraints = noDataConstraints + noData
    numberOfNoDataConstraints = len(noDataConstraints)
    return Model(equiModel.cost,[noDataConstraints, dataConstraints]), numberOfNoDataConstraints

def twoTermBoydModel(model):
    constraints = []
    for i, p in enumerate(model.as_posyslt1()):
        constraints.append(TTA.twoTermExpApproximationBoyd(p,i))
    return Model(model.cost,constraints)
    
def tractableModel(model,r = 3,tol = 0.001, coupled = True, dependentUncertainties = False, twoTerm = True):
    dataConstraints = []
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    if dependentUncertainties == False and coupled == True and twoTerm == True:
        safeModel, numberOfNoDataConstraints = twoTermModel(model,dependentUncertainties)
    else:
        safeModel, numberOfNoDataConstraints = equivalentModel(model,dependentUncertainties,coupled)
    for i, p in enumerate(safeModel.as_posyslt1()):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [p <= 1]            
        else:
            if len(p.exps) == 2:
                noDataUpper, noDataLower, data = LTTP.linearizeTwoTermExp(p, i, r, tol)
                noDataConstraintsUpper = noDataConstraintsUpper + noDataUpper
                noDataConstraintsLower = noDataConstraintsLower + noDataLower
                dataConstraints = dataConstraints + data
            else:
                dataConstraints = dataConstraints + [p <= 1]
    numberOfNoDataConstraints = len(noDataConstraintsUpper)
    return Model(safeModel.cost,[noDataConstraintsUpper,dataConstraints]), Model(safeModel.cost,[noDataConstraintsLower,dataConstraints]), numberOfNoDataConstraints     

def tractableBoydModel(model,r=3,tol=0.001):
    dataConstraints = []
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    twoTerm = twoTermBoydModel(model)
    for i, p in enumerate(twoTerm.as_posyslt1()):
        if len(p.exps) == 2:
            noDataUpper, noDataLower, data = LTTP.linearizeTwoTermExp(p, i, r, tol)
            noDataConstraintsUpper = noDataConstraintsUpper + noDataUpper
            noDataConstraintsLower = noDataConstraintsLower + noDataLower
            dataConstraints = dataConstraints + data
        else:
            dataConstraints = dataConstraints + [p <= 1]
    numberOfNoDataConstraints = len(noDataConstraintsUpper)
    return Model(twoTerm.cost,[noDataConstraintsUpper,dataConstraints]), Model(twoTerm.cost,[noDataConstraintsLower,dataConstraints]), numberOfNoDataConstraints  

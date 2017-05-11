from gpkit import  Model
import TwoTermApproximation as TTA
import LinearizeTwoTermPosynomials as LTTP
import EquivalentPosynomial as EP

def uncertainModelVariables(model):
    subsVars = list(model.substitutions.keys())
    uncertainVars = [var for var in subsVars if var.key.pr != None]
    return uncertainVars

def sameModel(model):
    constraints = []
    for i, p in enumerate(model.as_posyslt1()):
        constraints.append(0.9999*p<=1)
        output = Model(model.cost,constraints)
        output.substitutions.update(model.substitutions)
    return output
    
def equivalentModel(model, dependentUncertainties = False, coupled = True):
    dataConstraints = []
    noDataConstraints = []
    uncertainVars = uncertainModelVariables(model)
    for i, p in enumerate(model.as_posyslt1()):
        #print(p)
        (noData, data) = EP.equivalentPosynomial(p,uncertainVars,i,coupled,dependentUncertainties)
        dataConstraints = dataConstraints + data
        noDataConstraints = noDataConstraints + noData
    numberOfNoDataConstraints = len(noDataConstraints)
    output = Model(model.cost,[noDataConstraints, dataConstraints])
    output.substitutions.update(model.substitutions)
    return output, numberOfNoDataConstraints
    
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
    output = Model(equiModel.cost,[noDataConstraints, dataConstraints])
    output.substitutions.update(model.substitutions)    
    return output, numberOfNoDataConstraints

def twoTermBoydModel(model):
    constraints = []
    for i, p in enumerate(model.as_posyslt1()):
        constraints.append(TTA.twoTermExpApproximationBoyd(p,i))
    output = Model(model.cost,constraints)
    output.substitutions.update(model.substitutions)    
    return output
    
def tractableModel(model,r = 3,tol = 0.001, coupled = True, dependentUncertainties = False, twoTerm = True, linearizeTwoTerm = True):
    dataConstraints = []
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    if (dependentUncertainties == False and coupled == True and twoTerm) or twoTerm == True:
        safeModel, numberOfNoDataConstraints = twoTermModel(model,dependentUncertainties)
    else:
        safeModel, numberOfNoDataConstraints = equivalentModel(model,dependentUncertainties,coupled)
    for i, p in enumerate(safeModel.as_posyslt1()):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [p <= 1]            
        else:
            if len(p.exps) == 2 and linearizeTwoTerm:
                uncertainSubsVars = uncertainModelVariables(model)                
                minVars = len(uncertainSubsVars)
                maxVars = 0
                pUncertainVars = []
                for i in xrange(len(p.exps)):
                    mUncertainVars = [var for var in p.exps[i].keys() if var in uncertainSubsVars]
                    minVars = min(minVars,len(mUncertainVars))
                    maxVars = max(maxVars,len(mUncertainVars))
                    for var in mUncertainVars:
                        if var not in pUncertainVars:
                            pUncertainVars.append(var)
                noDataUpper, noDataLower, data = LTTP.linearizeTwoTermExp(p, i, r, tol)
                noDataConstraintsUpper = noDataConstraintsUpper + noDataUpper
                noDataConstraintsLower = noDataConstraintsLower + noDataLower
                dataConstraints = dataConstraints + data
            else:
                dataConstraints = dataConstraints + [p <= 1]
    numberOfNoDataConstraints = len(noDataConstraintsUpper)
    outputUpper = Model(safeModel.cost,[noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions)    
    outputLower = Model(safeModel.cost,[noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions)
    return outputUpper, outputLower, numberOfNoDataConstraints     

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
    outputUpper = Model(twoTerm.cost,[noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions)    
    outputLower = Model(twoTerm.cost,[noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions)
    return outputUpper, outputLower, numberOfNoDataConstraints  

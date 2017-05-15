import numpy as np
import EquivalentModels as EM
import EquivalentPosynomial as EP
from gpkit import  Model

def uncertainVariablesExponents (dataMonomials, uncertainVars):
    RHS_Coeff_Uncertain = np.array([[-p.exps[0].get(var.key, 0) for var in uncertainVars]
                          for p in dataMonomials])
    return  RHS_Coeff_Uncertain

def normalizePerturbationVector(uncertainVars):
    prs = np.array([var.key.pr for var in uncertainVars])
    etaMax = np.log(1 + prs/100.0)
    etaMin = np.log(1 - prs/100.0)
    centeringVector = (etaMin + etaMax)/2.0
    scalingVector = etaMax - centeringVector
    return centeringVector, scalingVector

def constructRobustMonomailCoeffiecientsBoxUncertainty(RHS_Coeff_Uncertain, Gamma, centeringVector, scalingVector):
    b_purt = (RHS_Coeff_Uncertain * scalingVector[np.newaxis])       
    coefficient = []
    for i in xrange(RHS_Coeff_Uncertain.shape[0]):
        oneNorm = 0
        centering = 0
        for j in range(RHS_Coeff_Uncertain.shape[1]):
            oneNorm = oneNorm + np.abs(b_purt[i][j])
            centering = centering + RHS_Coeff_Uncertain[i][j] * centeringVector[j]
        coefficient.append([np.exp(Gamma*oneNorm)/np.exp(centering)])
    return coefficient

def constructRobustMonomailCoeffiecientsEllipticalUncertainty(expsOfUncertainVars, Gamma, centeringVector, scalingVector):
    b_purt = (expsOfUncertainVars * scalingVector[np.newaxis])                  
    coefficient = []
    for i in xrange(expsOfUncertainVars.shape[0]):
        twoNorm = 0
        centering = 0
        for j in range(expsOfUncertainVars.shape[1]):
            twoNorm = twoNorm + b_purt[i][j]**2
            centering = centering + expsOfUncertainVars[i][j] * centeringVector[j]
        coefficient.append([np.exp(Gamma*np.sqrt(twoNorm))/np.exp(centering)])
    return coefficient
    
def constructRobustMonomailCoeffiecientsRhombalUncertainty(expsOfUncertainVars, centeringVector, scalingVector):
    b_purt = (expsOfUncertainVars * scalingVector[np.newaxis])                  
    coefficient = []
    for i in xrange(expsOfUncertainVars.shape[0]):
        twoNorm = []
        centering = 0
        numberOfUncertainPars = 0
        for j in range(expsOfUncertainVars.shape[1]):
            twoNorm.append(np.abs(b_purt[i][j]))
            if b_purt[i][j] != 0:
                numberOfUncertainPars = numberOfUncertainPars + 1
            centering = centering + expsOfUncertainVars[i][j] * centeringVector[j]
        #print(numberOfUncertainPars)
        coefficient.append([np.exp(np.sqrt(numberOfUncertainPars)*max(twoNorm))/np.exp(centering)])
    return coefficient
    
def robustModelBoxUncertaintyUpperLower(model, Gamma,r,tol, numberOfRegressionPoints = 4, coupled = True, twoTerm = True, linearizeTwoTerm = True, enableSP = True):
    simplifiedModelUpper, simplifiedModelLower, numberOfNoDataConstraints = EM.tractableModel(model,r,tol,coupled, False, twoTerm, linearizeTwoTerm)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    uncertainVars = EM.uncertainModelVariables(model)
    posynomialsUpper = simplifiedModelUpper.as_posyslt1()
    posynomialsLower = simplifiedModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        #print(i)
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [posynomialsLower[i] <= 1]
        else:
            if len(p.exps) > 1:
                dataConstraints.append(EP.safePosynomialBoxUncertainty(p, uncertainVars, i, enableSP, numberOfRegressionPoints))
            else:
                dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    #for i in xrange(len(uncertainVars)):
        #print(uncertainVars[i].key.descr.get("name"))
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = normalizePerturbationVector(uncertainVars)
        coefficient = constructRobustMonomailCoeffiecientsBoxUncertainty(expsOfUncertainVars, Gamma, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def robustModelBoxUncertainty(model, Gamma, tol=0.001, numberOfRegressionPoints = 4, coupled = True, twoTerm = True, linearizeTwoTerm = True, enableSP = True):
    r = 2
    error = 1
    sol = 0
    flag = 0 
    while r <= 20 and error > 0.01:
        flag = 0
        #print(r)
        modelUpper, modelLower = robustModelBoxUncertaintyUpperLower(model, Gamma,r,tol, numberOfRegressionPoints, coupled, twoTerm, linearizeTwoTerm, False)
        try:
            solUpper = modelUpper.solve(verbosity = 0)
            sol = solUpper
        except:
            flag = 1
        try:
            solLower = modelLower.solve(verbosity = 0)
        except:
            print("infeasible")
            r=20
            sol = model.solve(verbosity = 0)
            break
        if flag != 1:    
            try:
                error = (solUpper.get('cost').m - solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
            except:
                error = (solUpper.get('cost') - solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    initialGuess = sol.get("variables")
    #if enableSP:
    #    modelUpper, modelLower = robustModelBoxUncertaintyUpperLower(model,r-1,tol, numberOfRegressionPoints, coupled, False, linearizeTwoTerm, True)
    #    subsVars = modelUpper.substitutions.keys()
    #    for i in xrange(len(subsVars)):
    #        del initialGuess[subsVars[i].key]
    return modelUpper, initialGuess, r

def boydRobustModelBoxUncertaintyUpperLower(model,r=3,tol=0.001):
    tracModelUpper, tracModelLower, numberOfNoDataConstraints = EM.tractableBoydModel(model,r,tol)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    posynomialsUpper = tracModelUpper.as_posyslt1()
    posynomialsLower = tracModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [posynomialsLower[i] <= 1]
        else:
            dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = normalizePerturbationVector(uncertainVars)
        coefficient = constructRobustMonomailCoeffiecientsBoxUncertainty(expsOfUncertainVars, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def boydRobustModelBoxUncertainty(model, tol=0.001):
    r = 3
    error = 1
    while r <= 20 and error > 0.01:
        modelUpper, modelLower = boydRobustModelBoxUncertaintyUpperLower(model,r,tol)
        solUpper = modelUpper.solve(verbosity = 0)
        solLower = modelLower.solve(verbosity = 0)
        try:
            error = (solUpper.get('cost').m - solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
        except:
            error = (solUpper.get('cost') - solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    return modelUpper,r
    
def boydRobustModelEllipticalUncertaintyUpperLower(model,r=3,tol=0.001):
    tracModelUpper, tracModelLower, numberOfNoDataConstraints = EM.tractableBoydModel(model,r,tol)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    posynomialsUpper = tracModelUpper.as_posyslt1()
    posynomialsLower = tracModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [posynomialsLower[i] <= 1]
        else:
            dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = normalizePerturbationVector(uncertainVars)
        coefficient = constructRobustMonomailCoeffiecientsEllipticalUncertainty(expsOfUncertainVars, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def boydRobustModelEllipticalUncertainty(model, tol=0.001):
    r = 3
    error = 1
    while r <= 20 and error > 0.01:
        modelUpper, modelLower = boydRobustModelEllipticalUncertaintyUpperLower(model,r,tol)
        solUpper = modelUpper.solve(verbosity = 0)
        solLower = modelLower.solve(verbosity = 0)
        try:
            error = (solUpper.get('cost').m - solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
        except:
            error = (solUpper.get('cost') - solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    return modelUpper,r   
    
def robustModelEllipticalUncertaintyUpperLower(model, Gamma,r,tol, numberOfRegressionPoints = 4, coupled = True,dependentUncertainties = True, twoTerm = False, linearizeTwoTerm = True, enableSP = True):
    simplifiedModelUpper, simplifiedModelLower, numberOfNoDataConstraints = EM.tractableModel(model,r,tol,coupled,dependentUncertainties, twoTerm, linearizeTwoTerm)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    uncertainVars = EM.uncertainModelVariables(model)
    posynomialsUpper = simplifiedModelUpper.as_posyslt1()
    posynomialsLower = simplifiedModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [posynomialsLower[i] <= 1]
        else:
            if len(p.exps) > 1:
                dataConstraints.append(EP.safePosynomialEllipticalUncertainty(p, uncertainVars, i, enableSP,numberOfRegressionPoints))
            else:
                dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = normalizePerturbationVector(uncertainVars)
        coefficient = constructRobustMonomailCoeffiecientsEllipticalUncertainty(expsOfUncertainVars, Gamma, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def robustModelEllipticalUncertainty(model, Gamma,tol = 0.001,numberOfRegressionPoints = 4 ,coupled = True,dependentUncertainties = True, twoTerm = False, linearizeTwoTerm = True, enableSP = True):
    r = 2
    error = 1
    sol = 0
    flag = 0 
    while r <= 20 and error > 0.00001:
        flag = 0
        #print(r)
        #print(error)
        modelUpper, modelLower = robustModelEllipticalUncertaintyUpperLower(model, Gamma,r,tol, numberOfRegressionPoints, coupled,dependentUncertainties, twoTerm, linearizeTwoTerm, False)
        try:
            solUpper = modelUpper.solve(verbosity = 0)
            sol = solUpper
        except:
            flag = 1
        try:
            solLower = modelLower.solve(verbosity = 0)
        except:
            print("infeasible")
            r=20
            sol = model.solve(verbosity = 0)
            break
        if flag != 1:
            try:
                error = (solUpper.get('cost').m - solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
            except:
                error = (solUpper.get('cost') - solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    initialGuess = sol.get("variables")
    #if enableSP:
    #    modelUpper, modelLower = robustModelEllipticalUncertaintyUpperLower(model,r-1,tol, numberOfRegressionPoints, coupled,dependentUncertainties, False, linearizeTwoTerm, True)
    #    subsVars = modelUpper.substitutions.keys()
    #    for i in xrange(len(subsVars)):
    #        del initialGuess[subsVars[i].key]
    return modelUpper, initialGuess, r
    
def robustModelRhombalUncertaintyUpperLower(model,r,tol, numberOfRegressionPoints = 4, coupled = True, dependentUncertainties = True, twoTerm = True, linearizeTwoTerm = True, enableSP = True):
    simplifiedModelUpper, simplifiedModelLower, numberOfNoDataConstraints = EM.tractableModel(model,r,tol,coupled,dependentUncertainties, twoTerm, linearizeTwoTerm)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    uncertainVars = EM.uncertainModelVariables(model)
    posynomialsUpper = simplifiedModelUpper.as_posyslt1()
    posynomialsLower = simplifiedModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [posynomialsLower[i] <= 1]
        else:
            if len(p.exps) > 1:
                dataConstraints.append(EP.safePosynomialRhombalUncertainty(p, uncertainVars, i, enableSP))
            else:
                dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = normalizePerturbationVector(uncertainVars)
        coefficient = constructRobustMonomailCoeffiecientsRhombalUncertainty(expsOfUncertainVars, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def robustModelRhombalUncertainty(model, tol=0.001, numberOfRegressionPoints = 4, coupled = True, dependentUncertainties = True, twoTerm = True, linearizeTwoTerm = True, enableSP = True):
    r = 2
    error = 1
    sol = 0
    flag = 0 
    while r <= 20 and error > 0.01:
        flag = 0
        print(r)
        print(error)
        modelUpper, modelLower = robustModelRhombalUncertaintyUpperLower(model,r,tol, numberOfRegressionPoints, coupled,dependentUncertainties, twoTerm, linearizeTwoTerm, False)
        try:
            solUpper = modelUpper.solve(verbosity = 0)
            sol = solUpper
        except:
            flag = 1
        try:
            solLower = modelLower.solve(verbosity = 0)
        except:
            print("infeasible")
            r=20
            sol = model.solve(verbosity = 0)
            break
        if flag != 1:
            try:
                error = (solUpper.get('cost').m - solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
            except:
                error = (solUpper.get('cost') - solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    initialGuess = sol.get("variables")
    if enableSP:
        modelUpper, modelLower = robustModelRhombalUncertaintyUpperLower(model,r-1,tol, numberOfRegressionPoints, coupled,dependentUncertainties, False, linearizeTwoTerm, True)
        subsVars = modelUpper.substitutions.keys()
        for i in xrange(len(subsVars)):
            del initialGuess[subsVars[i].key]
    return modelUpper, initialGuess, r

def solveRobustSPBox(model,Gamma,relTol = 1e-5):
    sol = model.localsolve(verbosity=0)
    initSol = sol.get('variables')
    initCost = sol['cost']
    newCost = initCost*(1 + 2*relTol)
    while (np.abs(initCost - newCost)/initCost) > relTol:
        apprModel = Model(model.cost,model.as_gpconstr(initSol))
        robModel = robustModelBoxUncertainty(apprModel,Gamma)[0]
        sol = robModel.solve(verbosity=0)
        initSol = sol.get('variables')
        initCost = newCost
        newCost = sol['cost']
        print(newCost)
    return sol

def solveRobustSPEll(model,Gamma,relTol = 1e-5):
    sol = model.localsolve(verbosity=0)
    initSol = sol.get('variables')
    initCost = sol['cost']
    newCost = initCost*(1 + 2*relTol)
    while (np.abs(initCost - newCost)/initCost) > relTol:
        apprModel = Model(model.cost,model.as_gpconstr(initSol))
        robModel = robustModelEllipticalUncertainty(apprModel,Gamma)[0]
        sol = robModel.solve(verbosity=0)
        initSol = sol.get('variables')
        initCost = newCost
        newCost = sol['cost']
        print(newCost)
    return sol

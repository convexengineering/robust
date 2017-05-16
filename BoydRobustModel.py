class BoydRobustModel(EquivalentModel):
    def boydRobustModelBoxUncertaintyUpperLower(model,r=3,tol=0.001):
        tracModelUpper, tracModelLower, numberOfNoDataConstraints = \
                                            EM.tractableBoydModel(model,r,tol)
        noDataConstraintsUpper = []
        noDataConstraintsLower = []
        dataConstraints = []
        dataMonomails = []
        posynomialsUpper = tracModelUpper.as_posyslt1()
        posynomialsLower = tracModelLower.as_posyslt1()
        for i,p in enumerate(posynomialsUpper):
            if i < numberOfNoDataConstraints:
                noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
                noDataConstraintsLower = noDataConstraintsLower + \
                                                [posynomialsLower[i] <= 1]
            else:
                dataMonomails.append(p)
        uncertainVars = EM.uncertainModelVariables(model)
        expsOfUncertainVars = uncertainVariablesExponents (dataMonomails,
                                                           uncertainVars)
        if expsOfUncertainVars.size > 0:
            centeringVector, scalingVector = \
                                normalizePerturbationVector(uncertainVars)
            coefficient = \
                constructRobustMonomailCoeffiecientsBoxUncertainty
                (expsOfUncertainVars, centeringVector, scalingVector)
            for i in xrange(len(dataMonomails)):
                dataConstraints = dataConstraints + \
                                        [coefficient[i][0]*dataMonomails[i] <= 1]
        outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
        outputUpper.substitutions.update(model.substitutions)
        outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
        outputLower.substitutions.update(model.substitutions)
        return outputUpper, outputLower

def boydRobustModelBoxUncertainty(model, tol=0.001):
    r = 3
    error = 1
    while r <= 20 and error > 0.01:
        modelUpper, modelLower = boydRobustModelBoxUncertaintyUpperLower(model,
                                                                         r,
                                                                         tol)
        solUpper = modelUpper.solve(verbosity = 0)
        solLower = modelLower.solve(verbosity = 0)
        try:
            error = \
                (solUpper.get('cost').m -
                 solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
        except:
            error = (solUpper.get('cost') -
                     solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    return modelUpper,r

def boydRobustModelEllipticalUncertaintyUpperLower(model,r=3,tol=0.001):
    tracModelUpper, tracModelLower, numberOfNoDataConstraints = \
                                        EM.tractableBoydModel(model,r,tol)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    posynomialsUpper = tracModelUpper.as_posyslt1()
    posynomialsLower = tracModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + \
                                            [posynomialsLower[i] <= 1]
        else:
            dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails,
                                                       uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = \
                                normalizePerturbationVector(uncertainVars)
        coefficient = \
            constructRobustMonomailCoeffiecientsEllipticalUncertainty
            (expsOfUncertainVars, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + \
                                    [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions)
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions)
    return outputUpper, outputLower

def boydRobustModelEllipticalUncertainty(model, tol=0.001):
    r = 3
    error = 1
    while r <= 20 and error > 0.01:
        modelUpper, modelLower = \
                boydRobustModelEllipticalUncertaintyUpperLower(model,r,tol)
        solUpper = modelUpper.solve(verbosity = 0)
        solLower = modelLower.solve(verbosity = 0)
        try:
            error = \
                (solUpper.get('cost').m -
                 solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
        except:
            error = (solUpper.get('cost') -
                     solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    return modelUpper,r

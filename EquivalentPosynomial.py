from gpkit import Variable, Monomial, SignomialsEnabled
import numpy as np
from sklearn import linear_model

def iterateMergeIntesectedLists(coupledPartition):
    i = 0
    while (i < len(coupledPartition)-1):
        j = i+1
        while (j < len(coupledPartition)):
            if not list(set(coupledPartition[i]) & set(coupledPartition[j])):
                j = j + 1
            else:
                coupledPartition[i] = list(set(coupledPartition[i]) | set(coupledPartition[j]))
                coupledPartition.pop(j)
        i = i + 1
    return coupledPartition
    
def mergeIntersectedLists (coupledPartition):
    l = len(coupledPartition) + 1
    while (l > len(coupledPartition)):
        l = len(coupledPartition)
        coupledPartition = iterateMergeIntesectedLists(coupledPartition)  
    return coupledPartition

def sameSign(a):
    for i in xrange(len(a) - 1):
        if a[0] * a[i + 1] < 0:
            return False
    return True

def correlatedMonomials(p,subsVars,m,dependentUncertainties):
    l = len(p.exps)
    pSubsVars = [var for var in p.varkeys if var in subsVars]
    coupledPartition = []
    if dependentUncertainties:
        for j in xrange(l):
            for var in pSubsVars:
                if var.key in p.exps[j]:
                    coupledPartition.append(j)
                    break
        return [coupledPartition]
    else:
        coupledPartitionCounter = 0
        for var in pSubsVars:
            coupledPartition.append([])
            checkSign = []
            for j in xrange(l):
                if var.key in p.exps[j]:
                    coupledPartition[coupledPartitionCounter].append(j)
                    checkSign.append(p.exps[j].get(var.key))
            coupledPartitionCounter = coupledPartitionCounter + 1
            if sameSign(checkSign):
                coupledPartitionCounter = coupledPartitionCounter - 1
                coupledPartition.pop(coupledPartitionCounter)
        mergeIntersectedLists (coupledPartition)
        return coupledPartition

def checkIfInListOfLists(element, listOfLists):
    for i in xrange(len(listOfLists)):
        if element in listOfLists[i]:
            return True
    return False

def checkIfNoData(uncertainVars, monomial):
    intersection = [var for var in uncertainVars if var.key in monomial]
    if not intersection:
        return True
    else:
        return False

def equivalentPosynomial(p,uncertainSubsVars,m,coupled,dependentUncertainties):
    if len(p.exps) == 1:
        if len(p.exps[0]) == 0:
            return [],[]
        else:
            if checkIfNoData(uncertainSubsVars, p.exps[0]):
                return [p <= 1] , []
            else:
                return [], [p <= 1]
    if coupled:
        coupledPartitions = correlatedMonomials(p,uncertainSubsVars,m,dependentUncertainties)
    else:
        coupledPartitions = []
    if len(coupledPartitions) != 0:
        if len(coupledPartitions[0]) == len(p.exps):
            return [], [p <= 1]
    ts = []
    dataConstraints = []
    noDataConstraint = []
    elements = list(range(len(p.exps)))
    uncoupled = [element for element in elements if not checkIfInListOfLists(element, coupledPartitions)]
    superScript = 0
    for i in uncoupled:
        if checkIfNoData(uncertainSubsVars, p.exps[i]):
            ts.append(Monomial(p.exps[i],p.cs[i]))
        else:
            t = Variable('t_%s^%s'%(m,superScript))
            superScript = superScript + 1
            ts.append(t)
            dataConstraints = dataConstraints +[Monomial(p.exps[i],p.cs[i]) <= t]
    for i in xrange(len(coupledPartitions)):
        posynomial = 0
        t = Variable('t_%s^%s'%(m,superScript))
        superScript = superScript + 1
        ts.append(t)
        for j in coupledPartitions[i]:
            posynomial = posynomial + Monomial(p.exps[j],p.cs[j])
        dataConstraints = dataConstraints + [posynomial <= t]
    noDataConstraint = noDataConstraint + [sum(ts) <= 1]
    return noDataConstraint, dataConstraints

def linearizePurturbationsOld (p, uncertainVars):
    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
    center = []
    scale = []
    meanVector = []
    for i in xrange(len(pUncertainVars)):
        pr = pUncertainVars[i].key.pr
        center.append(np.sqrt(1 - pr**2/10000.0))
        scale.append(0.5*np.log((1 + pr/100.0)/(1 - pr/100.0)))
    perturbationMatrix = []
    for i in xrange(len(p.exps)):
        perturbationMatrix.append([])
        monUncertainVars = [var for var in pUncertainVars if var in p.exps[i]]
        mean = 1
        for j,var in enumerate(pUncertainVars):
            if var.key in monUncertainVars:
                mean = mean*center[j]**(p.exps[i].get(var.key))
        meanVector.append(mean)
        for j,var in enumerate(pUncertainVars):
            if var.key in monUncertainVars:
                perturbationMatrix[i].append(p.exps[i].get(var.key)*scale[j])
            else:
                perturbationMatrix[i].append(0)
    return perturbationMatrix, meanVector

def mergeMeshGrid(array,n):
    if n == 1:
        return [array]
    else:
        output = []
        for i in xrange(len(array)):
            output = output + mergeMeshGrid(array[i],n/(len(array) + 0.0))
        return output 

def perturbationFunction(perturbationVector,numberOfPoints = 3):
    dim = len(perturbationVector)
    x = np.meshgrid(*[np.linspace(-1,1,numberOfPoints)]*dim)
    result = []
    inputList = []
    for i in xrange(numberOfPoints**dim):
        inputList.append([])
    for i in xrange(dim):
        temp = mergeMeshGrid(x[i],numberOfPoints**dim)
        for j in xrange(numberOfPoints**dim):
            inputList[j].append(temp[j])
    for i in xrange(numberOfPoints**dim):
        output = 1
        for j in xrange(dim):
            if perturbationVector[j] != 0:
                output = output*perturbationVector[j]**inputList[i][j]
        result.append(output)
    clf = linear_model.LinearRegression()
    clf.fit(inputList,result)
    return clf.coef_, clf.intercept_

def linearizePurturbations (p, uncertainVars, numberOfPoints = 3):
    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
    center = []
    scale = []
    meanVector = []
    coeff = []
    intercept = []
    for i in xrange(len(pUncertainVars)):
        pr = pUncertainVars[i].key.pr
        center.append(np.sqrt(1 - pr**2/10000.0))
        scale.append(0.5*np.log((1 + pr/100.0)/(1 - pr/100.0)))
    perturbationMatrix = []
    for i in xrange(len(p.exps)):
        perturbationMatrix.append([])
        monUncertainVars = [var for var in pUncertainVars if var in p.exps[i]]
        mean = 1
        for j,var in enumerate(pUncertainVars):
            if var.key in monUncertainVars:
                mean = mean*center[j]**(p.exps[i].get(var.key))
        meanVector.append(mean)
        for j,var in enumerate(pUncertainVars):
            if var.key in monUncertainVars:
                perturbationMatrix[i].append(np.exp(p.exps[i].get(var.key)*scale[j]))
            else:
                perturbationMatrix[i].append(0)
        coeff.append([])
        intercept.append([])
        coeff[i],intercept[i] = perturbationFunction(perturbationMatrix[i], numberOfPoints)
    return coeff, intercept, meanVector

def noCoefficientMonomials (p, uncertainVars):
    monomials = []
    for i in xrange(len(p.exps)):
        monomials.append(Monomial(p.exps[i],p.cs[i]))
    return monomials

def safePosynomialEllipticalUncertainty(p, uncertainVars, m, enableSP = False, numberOfPoints = 3):
    perturbationMatrix, intercept, meanVector = linearizePurturbations (p, uncertainVars, numberOfPoints)
    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
    if not pUncertainVars:
        return [p <= 1]
    monomials = noCoefficientMonomials (p, uncertainVars)
    constraints = []
    s_main = Variable("s_%s"%(m))
    constraints = constraints + [sum([a*b for a,b in zip([a*b for a,b in zip(meanVector,intercept)],monomials)]) + s_main**0.5 <= 1]
    ss = []
    for i in xrange(len(perturbationMatrix[0])):
        positivePert = []
        negativePert = []
        positiveMonomials = []
        negativeMonomials = []
        s = Variable("s^%s_%s"%(i,m))
        ss.append(s)
        for j in xrange(len(perturbationMatrix)):
            if perturbationMatrix[j][i] > 0:
                positivePert.append(meanVector[j]*perturbationMatrix[j][i])
                positiveMonomials.append(monomials[j])
            elif perturbationMatrix[j][i] < 0:
                negativePert.append(-meanVector[j]*perturbationMatrix[j][i])
                negativeMonomials.append(monomials[j])
        if enableSP:
            with SignomialsEnabled():
                constraints = constraints + [(sum([a*b for a,b in zip(positivePert,positiveMonomials)]) 
                                             - sum([a*b for a,b in zip(negativePert,negativeMonomials)]))**2 <= s]
        else:       
            constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)])**2
                                         + sum([a*b for a,b in zip(negativePert,negativeMonomials)])**2 <= s]
    constraints.append(sum(ss) <= s_main)
    return constraints

#def safePosynomialEllipticalUncertainty(p, uncertainVars, m, enableSP = False):
#    perturbationMatrix, meanVector = linearizePurturbationsOld (p, uncertainVars)
#    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
#    if not pUncertainVars:
#        return [p <= 1]
#    monomials = noCoefficientMonomials (p, uncertainVars)
#    constraints = []
#    s_main = Variable("s_%s"%(m))
#    constraints = constraints + [sum([a*b for a,b in zip(meanVector,monomials)]) + s_main**0.5 <= 1]
#    ss = []
#    for i in xrange(len(perturbationMatrix[0])):
#        positivePert = []
#        negativePert = []
#        positiveMonomials = []
#        negativeMonomials = []
#        s = Variable("s^%s_%s"%(i,m))
#        ss.append(s)
#        for j in xrange(len(perturbationMatrix)):
#            if perturbationMatrix[j][i] > 0:
#                positivePert.append(meanVector[j]*perturbationMatrix[j][i])
#                positiveMonomials.append(monomials[j])
#            elif perturbationMatrix[j][i] < 0:
#                negativePert.append(-meanVector[j]*perturbationMatrix[j][i])
#                negativeMonomials.append(monomials[j])
#        if enableSP:
#            with SignomialsEnabled():
#                constraints = constraints + [(sum([a*b for a,b in zip(positivePert,positiveMonomials)]) 
#                                             - sum([a*b for a,b in zip(negativePert,negativeMonomials)]))**2 <= s]
#        else:       
#            constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)])**2
#                                         + sum([a*b for a,b in zip(negativePert,negativeMonomials)])**2 <= s]
#    constraints.append(sum(ss) <= s_main)
#    return constraints

#def safePosynomialBoxUncertainty(p, uncertainVars, m, enableSP = False):
#    perturbationMatrix, meanVector = linearizePurturbationsOld (p, uncertainVars)
#    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
#    if not pUncertainVars:
#        return [p <= 1]
#    monomials = noCoefficientMonomials (p, uncertainVars)
#    constraints = []
#    s_main = Variable("s_%s"%(m))
#    constraints = constraints + [sum([a*b for a,b in zip(meanVector,monomials)]) + s_main <= 1]
#    ss = []
#    for i in xrange(len(perturbationMatrix[0])):
#        positivePert = []
#        negativePert = []
#        positiveMonomials = []
#        negativeMonomials = []
#        s = Variable("s^%s_%s"%(i,m))
#        ss.append(s)
#        for j in xrange(len(perturbationMatrix)):
#            if perturbationMatrix[j][i] > 0:
#                positivePert.append(perturbationMatrix[j][i])
#                positiveMonomials.append(monomials[j])
#            elif perturbationMatrix[j][i] < 0:
#                negativePert.append(-perturbationMatrix[j][i])
#                negativeMonomials.append(monomials[j])
#        if enableSP:
#            with SignomialsEnabled():
#                if negativePert and not positivePert:
#                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
#                elif positivePert and not negativePert:
#                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
#                else:
#                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) 
#                                                 - sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
#                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) 
#                                                 - sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
#        else:
#            if positivePert:
#                constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) <= s]
#            if negativePert:
#                constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) <= s]
#    constraints.append(sum(ss) <= s_main)
#    return constraints
    
def safePosynomialBoxUncertainty(p, uncertainVars, m, enableSP = False, numberOfPoints = 3):
    perturbationMatrix, intercept, meanVector = linearizePurturbations (p, uncertainVars, numberOfPoints)
    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
    if not pUncertainVars:
        return [p <= 1]
    monomials = noCoefficientMonomials (p, uncertainVars)
    constraints = []
    s_main = Variable("s_%s"%(m))
    constraints = constraints + [sum([a*b for a,b in zip([a*b for a,b in zip(meanVector,intercept)],monomials)]) + s_main <= 1]
    ss = []
    for i in xrange(len(perturbationMatrix[0])):
        positivePert = []
        negativePert = []
        positiveMonomials = []
        negativeMonomials = []
        s = Variable("s^%s_%s"%(i,m))
        ss.append(s)
        for j in xrange(len(perturbationMatrix)):
            if perturbationMatrix[j][i] > 0:
                positivePert.append(meanVector[j]*perturbationMatrix[j][i])
                positiveMonomials.append(monomials[j])
            elif perturbationMatrix[j][i] < 0:
                negativePert.append(-meanVector[j]*perturbationMatrix[j][i])
                negativeMonomials.append(monomials[j])
        if enableSP:
            with SignomialsEnabled():
                if negativePert and not positivePert:
                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
                elif positivePert and not negativePert:
                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
                else:
                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) 
                                                 - sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) 
                                                 - sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
        else:
            if positivePert:
                constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) <= s]
            if negativePert:
                constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) <= s]
    constraints.append(sum(ss) <= s_main)
    return constraints

#def safePosynomialRhombalUncertainty(p, uncertainVars, m, enableSP = False):
#    perturbationMatrix, meanVector = linearizePurturbationsOld (p, uncertainVars)
#    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
#    if not pUncertainVars:
#        return [p <= 1]
#    monomials = noCoefficientMonomials (p, uncertainVars)
#    constraints = []
#    s = Variable("s_%s"%(m))
#    constraints = constraints + [sum([a*b for a,b in zip(meanVector,monomials)]) + s <= 1]
#    for i in xrange(len(perturbationMatrix[0])):
#        positivePert = []
#        negativePert = []
#        positiveMonomials = []
#        negativeMonomials = []
#        for j in xrange(len(perturbationMatrix)):
#            if perturbationMatrix[j][i] > 0:
#                positivePert.append(perturbationMatrix[j][i])
#                positiveMonomials.append(monomials[j])
#            elif perturbationMatrix[j][i] < 0:
#                negativePert.append(-perturbationMatrix[j][i])
#                negativeMonomials.append(monomials[j])
#        if enableSP:
#            with SignomialsEnabled():
#                if negativePert and not positivePert:
#                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
#                elif positivePert and not negativePert:
#                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
#                else:
#                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) 
#                                                 - sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
#                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) 
#                                                 - sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
#        else:
#            if positivePert:
#                constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) <= s]
#            if negativePert:
#                constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) <= s]
#    return constraints   

def safePosynomialRhombalUncertainty(p, uncertainVars, m, enableSP = False, numberOfPoints = 2):
    perturbationMatrix, intercept, meanVector = linearizePurturbations (p, uncertainVars, numberOfPoints)
    pUncertainVars = [var for var in p.varkeys if var in uncertainVars]
    if not pUncertainVars:
        return [p <= 1]
    monomials = noCoefficientMonomials (p, uncertainVars)
    constraints = []
    s = Variable("s_%s"%(m))
    constraints = constraints + [sum([a*b for a,b in zip([a*b for a,b in zip(meanVector,intercept)],monomials)]) + s <= 1]
    for i in xrange(len(perturbationMatrix[0])):
        positivePert = []
        negativePert = []
        positiveMonomials = []
        negativeMonomials = []
        for j in xrange(len(perturbationMatrix)):
            if perturbationMatrix[j][i] > 0:
                positivePert.append(meanVector[j]*perturbationMatrix[j][i])
                positiveMonomials.append(monomials[j])
            elif perturbationMatrix[j][i] < 0:
                negativePert.append(-meanVector[j]*perturbationMatrix[j][i])
                negativeMonomials.append(monomials[j])
        if enableSP:
            with SignomialsEnabled():
                if negativePert and not positivePert:
                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
                elif positivePert and not negativePert:
                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
                else:
                    constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) 
                                                 - sum([a*b for a,b in zip(negativePert,negativeMonomials)])<= s]
                    constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) 
                                                 - sum([a*b for a,b in zip(positivePert,positiveMonomials)])<= s]
        else:
            if positivePert:
                constraints = constraints + [sum([a*b for a,b in zip(positivePert,positiveMonomials)]) <= s]
            if negativePert:
                constraints = constraints + [sum([a*b for a,b in zip(negativePert,negativeMonomials)]) <= s]
    return constraints   
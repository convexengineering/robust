from gpkit import Variable, Monomial
import numpy as np

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
    pUncertainVars = []
    minVars = len(uncertainSubsVars)
    maxVars = 0
    for i in xrange(len(p.exps)):
        mUncertainVars = [var for var in p.exps[i].keys() if var in uncertainSubsVars]
        minVars = min(minVars,len(mUncertainVars))
        maxVars = max(maxVars,len(mUncertainVars))
        for var in mUncertainVars:
            if var not in pUncertainVars:
                pUncertainVars.append(var)
    if len(pUncertainVars) == maxVars and len(pUncertainVars) == minVars:
        dependentUncertainties = False
    #if len(p.exps) == 2:
    #    print(p)
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
    #print(uncoupled)
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
        if coupledPartitions[i]:
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

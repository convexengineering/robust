import numpy as np
from gpkit import Variable, Monomial

def twoTermExpApproximation (p, uncertainVars, m = 1, n= 0):
    l = len(p.exps)
    if (l <= 2):
        return [],[p <= 1]
    else:
        dataConstraints = []
        noDataConstraints =[]
        zs = []
        negative, positive = separateBadAndGood(p, uncertainVars)
        lengthOfNegative = len(negative)
        lengthOfPositive = len(positive)
        for i in xrange(min(lengthOfNegative, lengthOfPositive)):
            z = Variable("z^%s_(%s,%s)" % (i,m,n))
            zs.append(z)
            dataConstraints = dataConstraints + [Monomial(p.exps[negative[i]],p.cs[negative[i]]) +
                                                 Monomial(p.exps[positive[i]],p.cs[positive[i]]) <= z]
        additional = np.abs(lengthOfPositive - lengthOfNegative)
        for i in xrange (additional):
            z = Variable("z^%s_(%s,%s)" % (i+np.min(lengthOfPositive - lengthOfNegative),m,n))
            zs.append(z)
            if lengthOfPositive > lengthOfNegative:
                dataConstraints = dataConstraints + [Monomial(p.exps[positive[i + lengthOfNegative]],
                                                              p.cs[positive[i + lengthOfNegative]]) <= z]
            else:
                dataConstraints = dataConstraints + [Monomial(p.exps[negative[i + lengthOfPositive]],
                                                              p.cs[positive[i + lengthOfPositive]]) <= z]
        noDataConstraints = noDataConstraints + [sum(zs) <= 1]
        return noDataConstraints, dataConstraints

def twoTermExpApproximationCombinations (p, uncertainVars, m = 1, n= 0):
    l = len(p.exps)
    if (l <= 2):
        return [],[p <= 1]
    else:
        dataConstraints = []
        noDataConstraints =[]
        zs = []
        negative, positive = separateBadAndGood(p, uncertainVars)
        intersection = list(set(negative) & set(positive))
        for i in xrange(len(intersection)):
            negative.remove(intersection[i])
            positive.remove(intersection[i])
        lengthOfNegative = len(negative)
        lengthOfPositive = len(positive)
        lengthOfIntersection = len(intersection)
        for i in xrange(lengthOfNegative):
            for j in xrange(lengthOfPositive):
                z = Variable("z^%s_(%s,%s)" % (i*lengthOfPositive + j,m,n))
                zs.append(z)
                dataConstraints = dataConstraints + [Monomial(p.exps[negative[i]],p.cs[negative[i]])/(lengthOfPositive + lengthOfIntersection) +
                                                     Monomial(p.exps[positive[j]],p.cs[positive[j]])/(lengthOfNegative + lengthOfIntersection) <= z]
        for i in xrange(lengthOfIntersection):
            for j in xrange(lengthOfNegative):
                z = Variable("z^%s_(%s,%s)" % (lengthOfNegative*lengthOfPositive + i*lengthOfNegative + j,m,n))
                zs.append(z)
                dataConstraints = dataConstraints + [Monomial(p.exps[negative[j]],p.cs[negative[j]])/(lengthOfPositive + lengthOfIntersection) +
                                                     Monomial(p.exps[intersection[i]],p.cs[intersection[i]])/(lengthOfNegative + lengthOfPositive + lengthOfIntersection - 1) <= z]
        for i in xrange(lengthOfIntersection):
            for j in xrange(lengthOfPositive):
                z = Variable("z^%s_(%s,%s)" % (lengthOfNegative*lengthOfPositive + lengthOfNegative*lengthOfIntersection + i*lengthOfPositive + j,m,n))
                zs.append(z)
                dataConstraints = dataConstraints + [Monomial(p.exps[positive[j]],p.cs[positive[j]])/(lengthOfNegative + lengthOfIntersection) +
                                                     Monomial(p.exps[intersection[i]],p.cs[intersection[i]])/(lengthOfNegative + lengthOfPositive + lengthOfIntersection - 1) <= z]
        for i in xrange(lengthOfIntersection):
            for j in xrange (lengthOfIntersection - i - 1):
                z = Variable("z^%s_(%s,%s)" % (lengthOfNegative*lengthOfPositive + (lengthOfNegative + lengthOfPositive)*lengthOfIntersection + i*(lengthOfIntersection - i - 1) + j,m,n))
                zs.append(z)
                dataConstraints = dataConstraints + [Monomial(p.exps[intersection[i]],p.cs[intersection[i]])/(lengthOfNegative + lengthOfPositive + lengthOfIntersection - 1) +
                                                     Monomial(p.exps[intersection[j]],p.cs[intersection[j]])/(lengthOfNegative + lengthOfPositive + lengthOfIntersection - 1) <= z]
        noDataConstraints = noDataConstraints + [sum(zs) <= 1]
        return noDataConstraints, dataConstraints
        
def separateBadAndGood(p, uncertainVars):
    l = len(p.exps)
    pSubsVars = [var for var in p.varkeys if var in uncertainVars]
    negative = []
    positive = []
    for var in pSubsVars:
        tempNegative = []
        tempPositive = []
        for j in xrange(l):
            if var.key in p.exps[j]:
                if p.exps[j].get(var.key) < 0:
                    tempNegative.append(j)
                else:
                    tempPositive.append(j)
        if not (not tempNegative or not tempPositive):
            negative = negative + tempNegative
            positive = positive + tempPositive
    return negative, positive
        
def twoTermExpApproximationBoyd(p,m,n = 0):
    l = len(p.exps)
    if (l <= 2):
        return [p <= 1]
    else:
        dataConstraints = []
        z1 = Variable("z^1_(%s,%s)"%(m,n))
        dataConstraints = dataConstraints + [Monomial(p.exps[0],p.cs[0]) + z1 <= 1]
        for i in xrange(l-3):
            if (i > 0):
                z1 = Variable("z^%s_(%s,%s)" % (i+1,m,n))
            z2 = Variable("z^%s_(%s,%s)" % (i+2,m,n))
            dataConstraints = dataConstraints + [Monomial(p.exps[i+1],p.cs[i+1]) + z2/z1 <= 1]
        z2 = Variable("z^%s_(%s,%s)" % (l-2,m,n))
        dataConstraints = dataConstraints + [Monomial(p.exps[l-2],p.cs[l-2])/z2 +
                                             Monomial(p.exps[l-1],p.cs[l-1])/z2 <= 1]
        return dataConstraints

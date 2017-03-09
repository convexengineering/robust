import numpy as np
import scipy.optimize as op
from gpkit import Variable, Monomial

def tangentPointFunc(k,x,eps):
    return np.log(1 + np.exp(x)) - eps - np.log(1 + np.exp(k)) \
            - np.exp(k)*(x - k)/(1 + np.exp(k)) 
     
def intersectionPointFunc(x,a,b,eps):
    return a*x + b - np.log(1+np.exp(x)) + eps

def iterateTwoTermExpLinearizationCoeff(r,eps):
    a = []
    b = []
    xFirst = np.log(np.exp(eps) - 1)
    xNew = xFirst
    for i in xrange(r-2):
        xOld = xNew
        xTangent = op.newton(tangentPointFunc, xOld + 1, args = (xOld, eps))
        a.append(np.exp(xTangent)/(1 + np.exp(xTangent)))
        b.append(-np.exp(xTangent)*xTangent/(1 + np.exp(xTangent)) + np.log(1 + np.exp(xTangent)))
        xNew = op.newton(intersectionPointFunc, xTangent + 1, args = (a[i],b[i],eps))
    return (a,b,xNew)
    
def twoTermExpLinearizationCoeff(r, tol = 0.001):
    eps_min = 0
    eps_max = np.log(2)
    delta = 100
    while(delta > tol):
        eps = (eps_max + eps_min)/2
        xFinalTheoritical = -np.log(np.exp(eps) - 1)
        try:
            (a,b,xFinalActual) = iterateTwoTermExpLinearizationCoeff(r,eps)
        except:
            xFinalActual = xFinalTheoritical + 2*tol    
        if (xFinalActual < xFinalTheoritical):
            eps_min = eps
        else:
            eps_max = eps
        delta = np.abs(xFinalActual - xFinalTheoritical)
    return (a,b,eps)
    
def linearizeTwoTermExp(p, m, r, tol = 0.001):
    if len(p.exps) != 2:
        raise Exception('The Posynomial is not a two term posynomial')
    (a,b,eps) = twoTermExpLinearizationCoeff(r, tol = 0.001)
    dataConstraints = []
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    w = Variable('w_%s'%(m))
    noDataConstraintsUpper = dataConstraints + [w*np.exp(eps) <= 1]
    noDataConstraintsLower = dataConstraints + [w <= 1]    
    monomialOne = Monomial(p.exps[0],p.cs[0])
    monomialTwo = Monomial(p.exps[1],p.cs[1])
    dataConstraints = dataConstraints + [monomialOne <= w]
    for i in xrange(r-2):
        dataConstraints = dataConstraints + [monomialOne**a[r-3-i]*
                                             monomialTwo**a[i]*np.exp(b[i]) <= w]
    dataConstraints = dataConstraints + [monomialTwo <= w]
    return noDataConstraintsUpper, noDataConstraintsLower,dataConstraints

# coding: utf-8

# In[2]:

import GPModels as GPM
import RobustGP as RGP
from numpy import *
from matplotlib.pyplot import *

model = GPM.simpleWingTwoDimensionalUncertianty()
robustModel = RGP.robustModelEllipticalUncertainty(model, linearizeTwoTerm=True,
                                                   enableSP=True,
                                                   numberOfRegressionPoints=2)[0]
#print model.solve().summary()
try:
    print robustModel.localsolve().summary()
except:
    print robustModel.solve().summary()

# In[4]:

model


# In[5]:

1/(0.467*0.467), robustModel.solution["cost"]


# In[6]:

get_ipython().magic(u'pylab')


# In[15]:

from gpkit import Model, Variable

figure(figsize=(6,6))
for rad in linspace(0, 2*pi, 50):
    a_ = 1 * 15**cos(rad)
    b_ = 1.17 * 15**sin(rad)
    s = Variable("s")
    astar = Variable("a^*", a_)
    bstar = Variable("b^*", b_)
    a = model["toz"]
    b = model["k"]
    feasmodel = Model(s, [model, robustModel.cost <= robustModel.solution["cost"],
                          astar/s <= a, a <= s*astar, bstar/s <= b, b <= s*bstar, s >= 1])
    del feasmodel.substitutions["toz"]
    del feasmodel.substitutions["k"]
    feasmodel.solve(verbosity=0)
    loglog([a_, feasmodel.solution(a)], [b_, feasmodel.solution(b)], "k+-", alpha=0.25)
    xlabel("toz")
    ylabel("k")
    loglog([robustModel.substitutions["toz"]],
           [robustModel.substitutions["k"]], "k+")
    
th = linspace(0, 2*pi, 100)
loglog(1.15**sin(th), 1.17*1.1111**cos(th))
# xlim([0.3, 3])
# ylim([0.3, 3])


# In[23]:

feasmodel


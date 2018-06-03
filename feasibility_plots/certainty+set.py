
# coding: utf-8

# In[2]:

import GPModels as GPM
import RobustGP as RGP
from numpy import *
from matplotlib.pyplot import *


model = GPM.testModel()
robustModel = RGP.robustModelEllipticalUncertainty(model, linearizeTwoTerm=False,
                                                   enableSP=True,
                                                   numberOfRegressionPoints=2)[0]
print model.solve().summary()
print robustModel.solve().summary()


# In[3]:

model


# In[4]:

1/(0.467*0.467), robustModel.solution["cost"]


# In[5]:

get_ipython().magic(u'pylab')

# In[8]:

from gpkit import Model, Variable

figure(figsize=(6,6))
for rad in linspace(0, 2*pi, 100):
    a_ = 2**cos(rad)
    b_ = 2**sin(rad)
    s = Variable("s")
    astar = Variable("a^*", a_)
    bstar = Variable("b^*", b_)
    a = model["a"]
    b = model["b"]
    feasmodel = Model(s, [model, robustModel.cost <= robustModel.solution["cost"],
                          astar/s <= a, a <= s*astar, bstar/s <= b, b <= s*bstar, s >= 1])
    del feasmodel.substitutions["a"]
    del feasmodel.substitutions["b"]
    #print(feasmodel)
    feasmodel.solve(verbosity=0)
    loglog([a_, feasmodel.solution(a)], [b_, feasmodel.solution(b)], "k+-", alpha=0.25)
    xlabel("a")
    ylabel("b")
    loglog([robustModel.substitutions["a"]], [robustModel.substitutions["b"]], "k+")
    
th = linspace(0, 2*pi, 100)
loglog(1.1**sin(th), 1.1**cos(th))
xlim([0.5, 2])
ylim([0.5, 2])


# In[23]:

feasmodel


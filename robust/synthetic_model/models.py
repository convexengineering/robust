from __future__ import division
from builtins import range
import numpy as np
from gpkit import Variable, Model


def synthetic_model(number_of_constraints):
    constraints = []
    obj = 1
    number_of_gp_variables = int(number_of_constraints/2) + int(number_of_constraints*np.random.rand()) + 1
    gp_variables = []
    s = []  # Variable('s_relax_sm')
    for i in range(number_of_gp_variables):
        x = Variable('x_sm_%s' % i)
        gp_variables.append(x)
        constraints.append(x >= 0.01)
    number_of_uncertain_variables = int(50*np.random.rand()) + 1
    uncertain_variables = []
    for i in range(number_of_uncertain_variables):
        uncertain_variables.append(Variable('u_sm_%s' % i, 2*np.random.random(), pr=50*np.random.random()))

    for counter in range(number_of_constraints):
        number_of_monomials = int(15*np.random.random())+1
        vector_to_choose_from = [0, 0, 0, 1, -1]  # , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        m = number_of_monomials*[1]

        for j in range(number_of_monomials):
            for _ in range(int(number_of_gp_variables*np.random.rand()/2) + 1):
                m[j] *= (np.random.choice(gp_variables))**(10*np.random.random())  # -5)

        for i in range(number_of_uncertain_variables):
            neg_pos_neutral_powers = [vector_to_choose_from[int(len(vector_to_choose_from)*np.random.rand())] for _ in range(number_of_monomials)]
            for j in range(number_of_monomials):
                m[j] *= uncertain_variables[i]**(np.random.rand()*2*(neg_pos_neutral_powers[j]))
        s.append(Variable('s_relax_sm_%s' % counter))
        constraints.append(sum(m) <= s[counter])

    for x in gp_variables:
        obj += 1000*np.random.rand()*x**(-np.random.rand()*10)  # - 5)
    obj += sum([i**0.2 for i in s])
    m = Model(obj, constraints)
    return m


def test_synthetic_model():
    a = Variable("a", 1.17, "-", "form factor", pr=10)
    x = Variable('x')

    constraints = []
    constraints += [a*x + x/a + x**2/a**2 + a**3*x**1.3<= 1]

    return Model(1/x, constraints)

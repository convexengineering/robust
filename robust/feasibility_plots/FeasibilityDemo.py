

from gpkit import Variable, Model
import numpy as np

from robust.robust import RobustModel
from .plot_feasibilities import plot_feasibilities

k = Variable("k", 1.17, "-", "form factor")
e = Variable("e", 0.92, "-", "Oswald efficiency factor")
mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air")
# pi = Variable("\\pi", np.pi, "-", "half of the circle constant", pr= 0)
rho = Variable("\\rho", 1.23, "kg/m^3", "density of air")
tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio")
N_ult = Variable("N_{ult}", 3.3, "-", "ultimate load factor")
V_min = Variable("V_{min}", 25, "m/s", "takeoff speed")
C_Lmax = Variable("C_{L,max}", 1.6, "-", "max CL with flaps down")
S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.075, "-", "wetted area ratio")
W_W_coeff1 = Variable("W_{W_{coeff1}}", 12e-5, "1/m",
                      "Wing Weight Coefficent 1")
W_W_coeff2 = Variable("W_{W_{coeff2}}", 60, "Pa",
                      "Wing Weight Coefficent 2")
CDA0 = Variable("(CDA0)", 0.035, "m^2", "fuselage drag area")
W_0 = Variable("W_0", 6250, "N", "aircraft weight excluding wing")
dummy = Variable("dum", 1, "-")

# Free Variables
D = Variable("D", "N", "total drag force")
A = Variable("A", "-", "aspect ratio", fix=True)
S = Variable("S", "m^2", "total wing area", fix=True)
V = Variable("V", "m/s", "cruising speed")
W = Variable("W", "N", "total aircraft weight")
Re = Variable("Re", "-", "Reynold's number")
C_D = Variable("C_D", "-", "Drag coefficient of wing")
C_L = Variable("C_L", "-", "Lift coefficent of wing")
C_f = Variable("C_f", "-", "skin friction coefficient")
W_w = Variable("W_w", "N", "wing weight")

constraints = []

# Drag model
C_D_fuse = CDA0 / S
C_D_wpar = k * C_f * S_wetratio
C_D_ind = C_L ** 2 / (np.pi * A * e)
constraints += [C_D >= C_D_fuse * dummy + C_D_wpar / dummy + C_D_ind * dummy]

# Wing weight model
# W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
W_w_surf = W_W_coeff2 * S
constraints += [W_w >= W_w_surf + W_w_strc]

# and the rest of the models
constraints += [D >= 0.5 * rho * S * C_D * V ** 2,
                Re <= (rho / mu) * V * (S / A) ** 0.5,
                C_f >= 0.074 / Re ** 0.2,
                W <= 0.5 * rho * S * C_L * V ** 2,
                W <= 0.5 * rho * S * C_Lmax * V_min ** 2,
                W >= W_0 + W_w]

m = Model(D, constraints)
sol = m.solve()


def plot_feasibility_simple_Wing(type_of_uncertainty_set, x, y, str1, val1, str2, val2, design_feasibility):
    plot_feasibilities(x, y, m)
    x.key.descr[str1] = val1
    y.key.descr[str2] = val2
    RM = RobustModel(m, type_of_uncertainty_set, linearizationTolerance=1e-4)
    RMsol = RM.robustsolve(verbosity=0, minNumOfLinearSections=20, maxNumOfLinearSections=40)
    print("nominal: ", {k: v for k, v in list(sol["freevariables"].items())
                        if k in m.varkeys and k.key.fix is True})
    print("robust: ", {k: v for k, v in list(RMsol["freevariables"].items())
                       if k in m.varkeys and k.key.fix is True})
    print('cost', RMsol['cost'])
    plot_feasibilities(x, y, m, RM, numberofsweeps=120, design_feasibility=design_feasibility, skipfailures=True)
    del x.key.descr[str1]
    del y.key.descr[str2]


# plot_feasibility_simple_Wing('ellipsoidal', W_0, W_W_coeff2, 'r', 1.6, 'r', 1.66, True)
# plot_feasibility_simple_Wing('ellipsoidal', W_0, W_W_coeff2, 'r', 1.6, 'r', 1.66, False)
# plot_feasibility_simple_Wing('box', W_0, W_W_coeff2, 'pr', 60, 'pr', 66, True)
# plot_feasibility_simple_Wing('box', W_0, W_W_coeff2, 'pr', 60, 'pr', 66, False)

# plot_feasibility_simple_Wing('ellipsoidal', W_W_coeff1, W_W_coeff2, 'r', 1.6, 'r', 1.66, True)
# plot_feasibility_simple_Wing('ellipsoidal', W_W_coeff1, W_W_coeff2, 'r', 1.6, 'r', 1.66, False)
# plot_feasibility_simple_Wing('box', W_W_coeff1, W_W_coeff2, 'pr', 60, 'pr', 66, True)
# plot_feasibility_simple_Wing('box', W_W_coeff1, W_W_coeff2, 'pr', 60, 'pr', 66, False)

# plot_feasibility_simple_Wing('ellipsoidal', C_Lmax, V_min, 'r', 1.25, 'r', 1.2, True)
# plot_feasibility_simple_Wing('ellipsoidal', C_Lmax, V_min, 'r', 1.25, 'r', 1.2, False)
# plot_feasibility_simple_Wing('box', C_Lmax, V_min, 'pr', 25, 'pr', 20, True)
# plot_feasibility_simple_Wing('box', C_Lmax, V_min, 'pr', 25, 'pr', 20, False)

# plot_feasibility_simple_Wing('ellipsoidal', dummy, e, 'r', 1.12, 'r', 1.31, True)
# plot_feasibility_simple_Wing('ellipsoidal', dummy, e, 'r', 1.12, 'r', 1.31, False)
# plot_feasibility_simple_Wing('box', dummy, e, 'pr', 12, 'pr', 31, True)
# plot_feasibility_simple_Wing('box', dummy, e, 'pr', 12, 'pr', 31, False)

# plot_feasibility_simple_Wing('ellipsoidal', rho, dummy, 'r', 1.1, 'r', 1.12, True)
# plot_feasibility_simple_Wing('ellipsoidal', rho, dummy, 'r', 1.1, 'r', 1.12, False)
# plot_feasibility_simple_Wing('box', rho, dummy, 'pr', 10, 'pr', 12, True)
# plot_feasibility_simple_Wing('box', rho, dummy, 'pr', 10, 'pr', 12, False)

plot_feasibility_simple_Wing('ellipsoidal', rho, W_0, 'r', 1.1, 'r', 1.6, True)
plot_feasibility_simple_Wing('box', rho, W_0, 'r', 1.1, 'r', 1.6, False)

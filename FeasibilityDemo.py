import numpy as np
from gpkit import Variable, Model
from Robust import RobustModel
from plot_feasibilities import plot_feasibilities
from RobustGPTools import RobustGPTools
from gpkit.small_scripts import mag
import numpy as np

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
toz = Variable("toz", 1, "-")

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
constraints += [C_D >= C_D_fuse * toz + C_D_wpar / toz + C_D_ind / toz]

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

def plot_feasibility_simple_Wing(type_of_uncertainty_set, x, y, str1, val1, str2, val2):
    m = Model(D, constraints)
    _ = m.solve()
    plot_feasibilities(x, y, m)
    x.key.descr[str1] = val1
    y.key.descr[str2] = val2
    RM = RobustModel(m, type_of_uncertainty_set, linearizationTolerance=1e-4)
    RMsol = RM.robustsolve(verbosity=1, minNumOfLinearSections=20, maxNumOfLinearSections=40)
    print 'cost', RMsol['cost']
    plot_feasibilities(x, y, m, RM)
    del x.key.descr[str1]
    del y.key.descr[str2]


plot_feasibility_simple_Wing('elliptical', W_0, W_W_coeff1, 'r', 1.6, 'r', 1.6)
plot_feasibility_simple_Wing('box', W_0, W_W_coeff1, 'pr', 60, 'pr', 60)
# plot_feasibility_simple_Wing('elliptical', W_W_coeff1, W_W_coeff2, 'r', 1.6, 'r', 1.66)
# plot_feasibility_simple_Wing('box', W_W_coeff1, W_W_coeff2, 'pr', 60, 'pr', 66)
# plot_feasibility_simple_Wing('elliptical', C_Lmax, V_min, 'r', 1.25, 'r', 1.2)
# plot_feasibility_simple_Wing('box', C_Lmax, V_min, 'pr', 25, 'pr', 20)
# plot_feasibility_simple_Wing('elliptical', toz, k, 'r', 1.15, 'r', 1.31)
# plot_feasibility_simple_Wing('box', toz, k, 'pr', 15, 'pr', 31)
# plot_feasibility_simple_Wing('elliptical', W_W_coeff1, k, 'r', 1.6, 'r', 1.31)
# plot_feasibility_simple_Wing('box', W_W_coeff1, k, 'pr', 60, 'pr', 31)
# plot_feasibility_simple_Wing('elliptical', W_W_coeff1, C_Lmax, 'r', 1.6, 'r', 1.25)
# plot_feasibility_simple_Wing('box', W_W_coeff1, C_Lmax, 'pr', 60, 'pr', 25)
# plot_feasibility_simple_Wing('elliptical', W_W_coeff1, rho, 'r', 1.6, 'r', 1.1)
# plot_feasibility_simple_Wing('box', W_W_coeff1, rho, 'pr', 60, 'pr', 10)
# plot_feasibility_simple_Wing('elliptical', W_W_coeff1, toz, 'r', 1.6, 'r', 1.15)
# plot_feasibility_simple_Wing('box', W_W_coeff1, toz, 'pr', 60, 'pr', 15)

"""
m = Model(D, constraints)
nominalsol = m.solve()

C_Lmax.key.descr['r'] = 1.25
V_min.key.descr['r'] = 1.2

RM = RobustModel(m, 'elliptical', linearizationTolerance=1e-4)
RMsol = RM.robustsolve(verbosity=1, minNumOfLinearSections=20, maxNumOfLinearSections=40)

subs = {k: v for k, v in RMsol["freevariables"].items()
        if k.key.fix is True}
print subs

eta_min_x, eta_max_x = RobustGPTools.generate_etas(C_Lmax, 'elliptical', RM.number_of_stds, RM.setting)
eta_min_y, eta_max_y = RobustGPTools.generate_etas(V_min, 'elliptical', RM.number_of_stds, RM.setting)
center_x = (eta_min_x + eta_max_x) / 2.0
center_y = (eta_min_y + eta_max_y) / 2.0

xo = 1.6
yo = 25
x_center = np.log(xo) + center_x
y_center = np.log(yo) + center_y

print x_center
print y_center
print xo
print yo
print eta_max_x
print eta_max_y
print x_center
print y_center
print np.log(xo) + eta_max_x - x_center
print np.log(yo) + eta_max_y - y_center
print np.exp(x_center)
print np.exp(y_center)
print np.exp(np.sin(2))
print np.exp(np.sin(2)) ** (np.log(xo) + eta_max_x - x_center)
print np.exp(x_center) * np.exp(np.sin(2)) ** (np.log(xo) + eta_max_x - x_center)
print np.exp(np.cos(2))
print np.exp(np.cos(2)) ** (np.log(yo) + eta_max_y - y_center)
print np.exp(y_center) * np.exp(np.cos(2)) ** (np.log(yo) + eta_max_y - y_center)

subs[C_Lmax] = np.exp(x_center) * np.exp(np.sin(2)) ** (np.log(xo) + eta_max_x - x_center)
print np.exp(x_center) * np.exp(np.sin(2)) ** (np.log(xo) + eta_max_x - x_center)
subs[V_min] = np.exp(y_center) * np.exp(np.cos(2)) ** (np.log(yo) + eta_max_y - y_center)
print np.exp(y_center) * np.exp(np.cos(2)) ** (np.log(yo) + eta_max_y - y_center)
print subs

class new(Model):
    def setup(self):
        self.cost = m.cost
        return [m], subs

new = new()
print new.substitutions
sol = new.solve()
print sol['freevariables']
"""

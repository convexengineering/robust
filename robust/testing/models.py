import numpy as np
from gpkit import Variable, Model, SignomialsEnabled, units
from gpkitmodels.SP.SimPleAC.SimPleAC_mission import SimPleAC, Mission
from gpkitmodels.SP.SimPleAC.SimPleAC import SimPleAC as simpleWingSP

def simple_wing():
    # Uncertain parameters
    k = Variable("k", 1.17, "-", "form factor", pr=31.111111)
    e = Variable("e", 0.92, "-", "Oswald efficiency factor", pr=7.6086956)
    mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air", pr=4.225352)
    rho = Variable("\\rho", 1.23, "kg/m^3", "density of air", pr=10)
    tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio", pr=33.333333)
    N_ult = Variable("N_{ult}", 3.3, "-", "ultimate load factor", pr=33.333333)
    V_min = Variable("V_{min}", 25, "m/s", "takeoff speed", pr=20)
    C_Lmax = Variable("C_{L,max}", 1.6, "-", "max CL with flaps down", pr=25)
    S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.075, "-", "wetted area ratio", pr=3.6144578)
    W_W_coeff1 = Variable("W_{W_{coeff1}}", 12e-5, "1/m", "Wing Weight Coefficient 1", pr=60)
    W_W_coeff2 = Variable("W_{W_{coeff2}}", 60, "Pa", "Wing Weight Coefficient 2", pr=66)
    CDA0 = Variable("(CDA0)", 0.035, "m^2", "fuselage drag area", pr=42.857142)
    W_0 = Variable("W_0", 6250, "N", "aircraft weight excluding wing", pr=60)

    # Free Variables
    D = Variable("D", "N", "total drag force")
    A = Variable("A", "-", "aspect ratio", fix=True)
    S = Variable("S", "m^2", "total wing area", fix=True)
    V = Variable("V", "m/s", "cruising speed")
    W = Variable("W", "N", "total aircraft weight")
    Re = Variable("Re", "-", "Reynold's number")
    C_D = Variable("C_D", "-", "Drag coefficient of wing")
    C_L = Variable("C_L", "-", "Lift coefficient of wing")
    C_f = Variable("C_f", "-", "skin friction coefficient")
    W_w = Variable("W_w", "N", "wing weight")
    constraints = []

    # Drag Model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (np.pi * A * e)
    constraints += [C_D >= C_D_ind + C_D_fuse + C_D_wpar]

    # Wing Weight Model
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
    return m

def gp_test_model():
    x = Variable('x')
    y = Variable('y')

    a = Variable('a', 0.6, pr=10)
    b = Variable('b', 0.5, pr=10)

    constraints = [a * b * x + a * b * y <= 1,
                   b * x / y + b * x * y + a*b**2 * x ** 2 <= 1]
    return Model((x * y) ** -1, constraints)

def sp_test_model():
    x = Variable('x')
    y = Variable('y')

    a = Variable('a', 0.6, pr=10)
    b = Variable('b', 0.5, pr=10)

    with SignomialsEnabled():
        constraints = [a * b * x + a * b * y <= 1 + a*x**2 + 0.5*b*x*y,
                       b * x / y + b * x * y + a*b**2 * x ** 2 <= 1]
    return Model((x * y) ** -1, constraints)

def primitive_sp():
    x = Variable('x')
    y = Variable('y')
    a = Variable('a', 1, pr=10)
    b = Variable('b', 1, pr=10)
    constraints = []
    with SignomialsEnabled():
        constraints = constraints + [x >= 1 - a * y, b * y <= 0.1]
    return Model(x, constraints)

def simple_wing_sp():
    m = simpleWingSP()
    m.cost = m['W_f']
    return m


def simple_ac():
    m = Mission(SimPleAC(), 4)
    m.substitutions.update({
        'h_{cruise_m}': 5000 * units('m'),
        'Range_m': 3000 * units('km'),
        'W_{p_m}': 6250 * units('N'),
        '\\rho_{p_m}': 1500*units('kg/m^3'),
        'C_m': 120 * units('1/hr'),
        'V_{min_m}': 25 * units('m/s'),
        'T/O factor_m': 2,
    })
    m = Model(m['W_{f_m}'] * units('1/N') + m['C_m'] * m['t_m'], m, m.substitutions)
    return m



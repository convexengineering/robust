# coding=utf-8
from gpkit import Variable, Model, SignomialsEnabled, VarKey, units
import numpy as np
from RobustGPTools import SameModel


def simpleWing():
    k = Variable("k", 1.17, "-", "form factor", pr=11.111111, sigma=0.035)  # [1.04 - 1.3] -> [0.039 - 0.262] -> 1.1624 -> 74
    e = Variable("e", 0.92, "-", "Oswald efficiency factor",
                 pr=7.6086956)  # [0.85 - 0.99] -> [-0.1625 - −0.01] -> 0.9173 -> 88
    mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air",
                  pr=4.225352)  # [1.7e-5 - 1.85e-5] -> [-10.982297 - -10.897739] -> 1.773414 -> 0.3865
    # pi = Variable("\\pi", np.pi, "-", "half of the circle constant", pr= 0)
    rho = Variable("\\rho", 1.23, "kg/m^3", "density of air")  # [1.2 - 1.3] -> [0.1823 - 0.2623] -> 1.2489 -> 18
    tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio",
                   pr=33.333333)  # [0.08 - 0.16] -> [-2.5257 - -1.8325] -> 0.1131 -> 16
    N_ult = Variable("N_{ult}", 3.3, "-", "ultimate load factor",
                     pr=33.333333)  # [2.2 - 4.4] -> [0.7884 - 1.4816] -> 3.1112 -> 30.5
    V_min = Variable("V_{min}", 25, "m/s", "takeoff speed", pr=20)  # [20 - 30] -> [2.9957 - 3.4011] -> 24.4948 -> 6.33
    C_Lmax = Variable("C_{L,max}", 1.6, "-", "max CL with flaps down",
                      pr=25)  # [1.2 - 2] -> [0.1823, 0.6931] -> 1.5491 -> 58.4
    S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.075, "-", "wetted area ratio",
                          pr=3.6144578)  # [2 - 2.15] -> [0.6931 - 0.7654] -> 2.0736 -> 4.95
    W_W_coeff1 = Variable("W_{W_{coeff1}}", 12e-5, "1/m",
                          "Wing Weight Coefficent 1",
                         )  # [4e-5 - 20e-5] ->[-10.1266 - -8.5171 -> 8.9442 -> 8.63
    W_W_coeff2 = Variable("W_{W_{coeff2}}", 60, "Pa",
                          "Wing Weight Coefficent 2",)  # [20 - 100] ->[2.9957 - 4.6051] -> 44.7213 -> 21.2
    CDA0 = Variable("(CDA0)", 0.035, "m^2", "fuselage drag area",
                    pr=42.857142)  # [0.02 - 0.05] -> [-3.9120 - -2.9957] -> 0.0316 -> 13.3
    W_0 = Variable("W_0", 6250, "N", "aircraft weight excluding wing",
                   pr=60)  # [2500 - 10000] -> [7.8240 - 9.2103] -> 5000 -> 8.14
    toz = Variable("toz", 1, "-", pr=k*e/tau/N_ult)  # [0.85 - 1.15] -> [-0.1625 - 0.1397] -> 0.9886 -> 1328
    ejer = Variable("ejer", 10000, "N")
    # Free Variables
    D = Variable("D", "N", "total drag force")
    A = Variable("A", "-", "aspect ratio")
    S = Variable("S", "m^2", "total wing area")
    V = Variable("V", "m/s", "cruising speed")
    W = Variable("W", "N", "total aircraft weight")
    Re = Variable("Re", "-", "Reynold's number")
    C_D = Variable("C_D", "-", "Drag coefficient of wing")
    C_L = Variable("C_L", "-", "Lift coefficient of wing")
    C_f = Variable("C_f", "-", "skin friction coefficient")
    W_w = Variable("W_w", "N", "wing weight")

    constraints = []

    # Drag model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (np.pi * A * e)
    constraints += [C_D >= C_D_ind * toz + C_D_fuse * toz + C_D_wpar / toz]

    # Wing weight model
    W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_surf = W_W_coeff2 * S
    constraints += [W_w >= W_w_surf + W_w_strc]

    # and the rest of the models
    constraints += [D >= 0.5 * rho * S * C_D * V ** 2,
                    Re <= (rho / mu) * V * (S / A) ** 0.5,
                    C_f >= 0.074 / Re ** 0.2,
                    W <= 0.5 * rho * S * C_L * V ** 2,
                    W <= 0.5 * rho * S * C_Lmax * V_min ** 2,
                    W >= W_0 + W_w,
                    D <= ejer]
    # for key in subs.keys():

    m = Model(D, constraints)
    m.W1 = W_W_coeff1
    m.W2 = W_W_coeff2
    return m


def simpleWingSP():
    # Env. constants
    g = Variable("g", 9.81, "m/s^2", "gravitational acceleration")
    mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air", pr=4.)
    # mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air")
    rho = Variable("\\rho", 1.23, "kg/m^3", "density of air", pr=5.)
    rho_f = Variable("\\rho_f", 817, "kg/m^3", "density of fuel")

    # Non-dimensional constants
    C_Lmax = Variable("C_{L,max}", 1.6, "-", "max CL with flaps down", pr=5.)
    e = Variable("e", 0.92, "-", "Oswald efficiency factor", pr=3.)
    k = Variable("k", 1.17, "-", "form factor", pr=10.)
    N_ult = Variable("N_{ult}", 3.3, "-", "ultimate load factor", pr=15.)
    S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.075, "-", "wetted area ratio", pr=3.)
    tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio", pr=10.)
    W_W_coeff1 = Variable("W_{W_{coeff1}}", 2e-5, "1/m",
                          "Wing Weight Coefficient 1", pr=30.)  # orig  12e-5
    W_W_coeff2 = Variable("W_{W_{coeff2}}", 60., "Pa",
                          "Wing Weight Coefficient 2", pr=10.)
    # p_labor = Variable('p_{labor}', 1., '1/min', 'cost of labor', pr=20.)
    # Dimensional constants
    CDA0 = Variable("(CDA0)", "m^2", "fuselage drag area")  # 0.035 originally
    Range = Variable("Range", 3000, "km", "aircraft range")
    toz = Variable("toz", 1, "-", pr=15.)
    TSFC = Variable("TSFC", 0.6, "1/hr", "thrust specific fuel consumption")
    V_min = Variable("V_{min}", 25, "m/s", "takeoff speed", pr=20.)
    W_0 = Variable("W_0", 6250, "N", "aircraft weight excluding wing", pr=20.)
    ejer = Variable("ejer", 11500, "N")
    # Free Variables
    LoD = Variable('L/D', '-', 'lift-to-drag ratio')
    D = Variable("D", "N", "total drag force")
    V = Variable("V", "m/s", "cruising speed")
    W = Variable("W", "N", "total aircraft weight")
    Re = Variable("Re", "-", "Reynold's number")
    C_D = Variable("C_D", "-", "Drag coefficient")
    C_L = Variable("C_L", "-", "Lift coefficent of wing")
    C_f = Variable("C_f", "-", "skin friction coefficient")

    W_f = Variable("W_f", "N", "fuel weight")
    V_f = Variable("V_f", "m^3", "fuel volume")

    T_flight = Variable("T_{flight}", "hr", "flight time")

    # Free variables (fixed for performance eval.)
    A = Variable("A", "-", "aspect ratio", fix=True)
    S = Variable("S", "m^2", "total wing area", fix=True)
    W_w = Variable("W_w", "N", "wing weight")  # , fix = True)
    W_w_strc = Variable('W_w_strc', 'N', 'wing structural weight', fix=True)
    W_w_surf = Variable('W_w_surf', 'N', 'wing skin weight', fix=True)
    V_f_wing = Variable("V_f_wing", 'm^3', 'fuel volume in the wing', fix=True)
    V_f_fuse = Variable('V_f_fuse', 'm^3', 'fuel volume in the fuselage', fix=True)
    V_f_avail = Variable("V_{f_{avail}}", "m^3", "fuel volume available")  # , fix = True)
    constraints = []
    # Drag model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (np.pi * A * e)
    constraints += [C_D >= C_D_fuse * toz + C_D_wpar / toz + C_D_ind * toz]

    with SignomialsEnabled():
        # Wing weight model
        # NOTE: This is a signomial constraint that has been GPified. Could revert back to signomial?
        constraints += [W_w >= W_w_surf + W_w_strc,
                        # W_w_strc >= W_W_coeff1 * (N_ult * A ** 1.5 * ((W_0+V_f_fuse*g*rho_f) * W * S) ** 0.5) / tau, #[GP]
                        W_w_strc ** 2. >= W_W_coeff1 ** 2. * (
                            N_ult ** 2. * A ** 3. * ((W_0 + V_f_fuse * g * rho_f) * W * S)) / tau ** 2.,
                        W_w_surf >= W_W_coeff2 * S]

        # and the rest of the models
        constraints += [LoD == C_L / C_D,
                        D >= 0.5 * rho * S * C_D * V ** 2,
                        Re <= (rho / mu) * V * (S / A) ** 0.5,
                        C_f >= 0.074 / Re ** 0.2,
                        T_flight >= Range / V,
                        W_0 + W_w + 0.5 * W_f <= 0.5 * rho * S * C_L * V ** 2,
                        W <= 0.5 * rho * S * C_Lmax * V_min ** 2,
                        W >= W_0 + W_w + W_f,
                        V_f == W_f / g / rho_f,
                        V_f_avail <= V_f_wing + V_f_fuse,  # [SP]
                        V_f_wing ** 2 <= 0.0009 * S ** 3 / A * tau ** 2,  # linear with b and tau, quadratic with chord!
                        V_f_fuse <= 10 * units('m') * CDA0,
                        V_f_avail >= V_f,
                        W_f >= TSFC * T_flight * D,
                        D <= ejer]

    # return Model(W_f/LoD, constraints)
    return Model(D, constraints)
    # return Model(W_f, constraints)
    # return Model(W,constraints)
    # return Model(W_f*T_flight,constraints)
    # return Model(W_f + 1*T_flight*units('N/min'),constraints)


def simpleWingTwoDimensionalUncertainty():
    k = Variable("k", 1.17, "-", "form factor", pr=11.111111)
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
    toz = Variable("toz", 1, "-", pr=15)

    # Free Variables
    D = Variable("D", "N", "total drag force")
    A = Variable("A", "-", "aspect ratio")
    S = Variable("S", "m^2", "total wing area")
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
    constraints += [C_D >= C_D_fuse / toz + C_D_wpar / toz + C_D_ind / toz]

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

    return Model(D, constraints)


def test_model():
    x = Variable('x')
    y = Variable('y')

    a = Variable('a', 0.6, pr=10)
    b = Variable('b', 0.5, pr=10)

    constraints = [a * b * x + a * b* y <= 1,
                   b * x / y + b * x * y + a*b**2 * x ** 2 <= 1]
    return Model((x * y) ** -1, constraints)


def example_sp():
    x = Variable('x')
    y = Variable('y')
    a = Variable('a', 1, pr=10)
    b = Variable('b', 1, pr=10)
    constraints = []
    with SignomialsEnabled():
        constraints = constraints + [x >= 1 - a * y, b * y <= 0.1]
    return Model(x, constraints)


def mike_solar_model():
    import gassolar.solar.solar as solar_mike
    model = solar_mike.Mission(latitude=11)
    model.cost = model["W_{total}"]

    uncertain_var_dic = {}  # 'V_{wind-ref}': 10, "m_{fac}": 1.5, "B_{PM}": 4, "\eta": 5,}# "\\eta_{charge}": 6,
                         # "\\eta_{discharge}": 4, "h_{batt}": 7, "\\tau": 4, "(E/\\mathcal{V})": 3,
                         # '\\eta_{prop}': 1, '\\kappa': 3, '\\bar{q}': 2, 'V_{NE}': 2, '1-cos(\\eta)': 2,
                         # '\\bar{M}_{tip}': 2, "W_{pay}": 2, '\\rho_{sl}': 2, '\\bar{S}_{tip}': 16, 'k': 2,
                         # 'p_{wind}': 2, '\\rho_{foam}': 2, '\\theta_{root}': 2, '\\bar{A}_{NACA0008}': 2,
                         # '\\theta_{max}': 2, 'Re**1.00_{low-bound}': 2, 'N_{max}': 2, '\\bar{J/t}': 2,
                         # '\\rho_{CFRP}': 2, 'Re**1.00_{up-bound}': 2, '\\rho_{solar}': 2, '(E/S)_{var}': 4,
                         # '\\rho_{skin}': 2,  '(1-k/2)': 2, 't_{min}': 2, '\\bar{c}_{ave}': 10, '\\rho_{ref}': 2,
                         # 'V**-1.00V_{gust}**1.001-cos(\\eta)**1.00_{low-bound}': 12, '\\lambda_h/(\\lambda_h+1)': 4,
                         # '\\sigma_{CFRP}': 9, '\\tau_{CFRP}': 8, 'C_{m_w}': 13, '\\bar{\\delta}_{root}': 7,
                         # 'w_{lim}': 10, '(E/S)_{irr}': 2, 't_{night}': 2, '\\mu': 8, '\\bar{c}': 8,
                         # '\\lambda_v/(\\lambda_v+1)': 8, '(P/S)_{var}': 2, 'V_{gust}': 3, "C_M": 4, "e": 1, 'E': 3,
                         # "C_{L_{max}}": 3, }  # 'P_{acc}': 0.1,

    keys = uncertain_var_dic.keys()
    # print keys
    for i in xrange(len(uncertain_var_dic)):
        # if len(model.variables_byname(keys[i])) > 1:
            # print keys[i]
        print("--------------------------------")
        print model.variables_byname(keys[i])
        for j in xrange(len(model.variables_byname(keys[i]))):
            # model.variables_byname(keys[i])[j].key.descr['pr'] = uncertain_var_dic.get(keys[i])
            copy_key = VarKey(**model.variables_byname(keys[i])[j].key.descr)
            # print copy_key
            copy_key.key.descr["pr"] = uncertain_var_dic.get(keys[i])
            # print copy_key.key.descr
            model.subinplace({model.variables_byname(keys[i])[j].key: copy_key})

    new_model = SameModel(model)
    new_model.substitutions.update(model.substitutions)
    new_model.unique_varkeys = model.varkeys
    new_model.reset_varkeys()

    # print new_model.variables_byname('V_h')[0].key.descr
    return new_model

def solve_model(model, *args):
    initial_guess = {}
    if args:
        initial_guess = args[0]
    try:
        sol = model.solve(verbosity=0)
    except:
        sol = model.localsolve(verbosity=0, x0=initial_guess)
    print (sol.summary())
    return sol

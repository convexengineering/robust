from gpkit import Variable, Model, SignomialsEnabled, VarKey, units, Vectorize
import numpy as np

from simulations import simulate, read_simulation_data


def simple_wing_sp():

    # Env. constants
    g = Variable("g", 9.81, "m/s^2", "gravitational acceleration")
    mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air", pr=4.)
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
    W_w = Variable("W_w", "N", "wing weight")
    W_w_strc = Variable('W_w_strc', 'N', 'wing structural weight', fix=True)
    W_w_surf = Variable('W_w_surf', 'N', 'wing skin weight', fix=True)
    V_f_wing = Variable("V_f_wing", 'm^3', 'fuel volume in the wing', fix=True)
    V_f_fuse = Variable('V_f_fuse', 'm^3', 'fuel volume in the fuselage', fix=True)
    V_f_avail = Variable("V_{f_{avail}}", "m^3", "fuel volume available")
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
                        W_f >= TSFC * T_flight * D]

    # return Model(W_f/LoD, constraints)
    return Model(D, constraints)
    # return Model(W_f, constraints)
    # return Model(W,constraints)
    # return Model(W_f*T_flight,constraints)
    # return Model(W_f + 1*T_flight*units('N/min'),constraints)


def example_sp():
    x = Variable('x')
    y = Variable('y')
    a = Variable('a', 1, pr=10)
    b = Variable('b', 1, pr=10)
    constraints = []
    with SignomialsEnabled():
        constraints = constraints + [x >= 1 - a * y, b * y <= 0.1]
    return Model(x, constraints)

if __name__ == '__main__':
    model = simple_wing_sp()
    number_of_time_average_solves = 2
    number_of_iterations = 40
    nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
        simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)
    model_name = 'Signomial Simple Wing'
    gammas = [0.3, 0.5]  # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    min_num_of_linear_sections = 3
    max_num_of_linear_sections = 99
    linearization_tolerance = 1e-3
    verbosity = 1

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True}]
    uncertainty_sets = ['box', 'elliptical']

    model = simple_wing_sp()

    variable_gamma_file_name = 'signomial_simple_wing/simulation_data_variable_gamma.txt'
    simulate.generate_variable_gamma_results(model, model_name, gammas, number_of_iterations,
                                             min_num_of_linear_sections,
                                             max_num_of_linear_sections, verbosity, linearization_tolerance,
                                             variable_gamma_file_name, number_of_time_average_solves, methods,
                                             uncertainty_sets, nominal_solution, nominal_solve_time,
                                             nominal_number_of_constraints, directly_uncertain_vars_subs)

    gamma = 1
    numbers_of_linear_sections = [12, 14]  # [12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 36, 44, 52, 60, 70, 80]

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False}]
    uncertainty_sets = ['box', 'elliptical']

    variable_pwl_file_name = 'signomial_simple_wing/simulation_data_variable_pwl.txt'
    simulate.generate_variable_piecewiselinearsections_results(model, model_name, gamma, number_of_iterations,
                                                               numbers_of_linear_sections, linearization_tolerance,
                                                               verbosity, variable_pwl_file_name,
                                                               number_of_time_average_solves, methods, uncertainty_sets,
                                                               nominal_solution, nominal_solve_time,
                                                               nominal_number_of_constraints, directly_uncertain_vars_subs)

    file_path_gamma = 'signomial_simple_wing/simulation_data_variable_gamma.txt'
    file_path_pwl = 'signomial_simple_wing/simulation_data_variable_pwl.txt'
    read_simulation_data.generate_all_plots(file_path_gamma, file_path_pwl)

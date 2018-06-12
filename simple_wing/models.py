from gpkit import Variable, Model, SignomialsEnabled, VarKey, units, Vectorize
import numpy as np

from simulations import simulate, read_simulation_data


def simple_wing():
    # Substitutions Variables
    k = Variable("k", 1.17, "-", "form factor", pr=31.111111, sigma=0.035)
    e = Variable("e", 0.92, "-", "Oswald efficiency factor", pr=7.6086956)
    mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air", pr=4.225352)
    rho = Variable("\\rho", 1.23, "kg/m^3", "density of air", pr=10)
    tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio", pr=33.333333)
    N_ult = Variable("N_{ult}", 3.3, "-", "ultimate load factor", pr=33.333333)
    V_min = Variable("V_{min}", 25, "m/s", "takeoff speed", pr=20)
    C_Lmax = Variable("C_{L,max}", 1.6, "-", "max CL with flaps down", pr=25)
    S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.075, "-", "wetted area ratio", pr=3.6144578)
    W_W_coeff1 = Variable("W_{W_{coeff1}}", 12e-5, "1/m", "Wing Weight Coefficent 1", pr=60)
    W_W_coeff2 = Variable("W_{W_{coeff2}}", 60, "Pa", "Wing Weight Coefficent 2", pr=66)
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


if __name__ == '__main__':
    model = simple_wing()
    number_of_time_average_solves = 100
    number_of_iterations = 1000
    nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
        simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations)
    model_name = 'Simple Wing'
    gammas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    min_num_of_linear_sections = 3
    max_num_of_linear_sections = 99
    linearization_tolerance = 1e-3
    verbosity = 0

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True},
               {'name': 'Two Term', 'twoTerm': False, 'boyd': True, 'simpleModel': False}]
    uncertainty_sets = ['box', 'elliptical']

    model = simple_wing()

    variable_gamma_file_name = \
        '/home/saab/Dropbox (MIT)/MIT/Masters/Code/robust/simple_wing/simulation_data_variable_gamma.txt'
    simulate.generate_variable_gamma_results(model, model_name, gammas, number_of_iterations,
                                             min_num_of_linear_sections,
                                             max_num_of_linear_sections, verbosity, linearization_tolerance,
                                             variable_gamma_file_name, number_of_time_average_solves, methods,
                                             uncertainty_sets, nominal_solution, nominal_solve_time,
                                             nominal_number_of_constraints, directly_uncertain_vars_subs)

    gamma = 1
    numbers_of_linear_sections = [12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 36, 44, 52, 60, 70, 80]

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Two Term', 'twoTerm': False, 'boyd': True, 'simpleModel': False}]
    uncertainty_sets = ['box', 'elliptical']

    variable_pwl_file_name = \
        '/home/saab/Dropbox (MIT)/MIT/Masters/Code/robust/simple_wing/simulation_data_variable_pwl.txt'
    simulate.generate_variable_piecewiselinearsections_results(model, model_name, gamma, number_of_iterations,
                                                               numbers_of_linear_sections, linearization_tolerance,
                                                               verbosity, variable_pwl_file_name,
                                                               number_of_time_average_solves, methods, uncertainty_sets,
                                                               nominal_solution, nominal_solve_time,
                                                               nominal_number_of_constraints, directly_uncertain_vars_subs)

    file_path_gamma = '/home/saab/Dropbox (MIT)/MIT/Masters/Code/robust/simple_wing/simulation_data_variable_gamma.txt'
    file_path_pwl = '/home/saab/Dropbox (MIT)/MIT/Masters/Code/robust/simple_wing/simulation_data_variable_pwl.txt'
    read_simulation_data.generate_all_plots(file_path_gamma, file_path_pwl)

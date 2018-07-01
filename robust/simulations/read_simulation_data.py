import matplotlib.pyplot as plt
import numpy as np


def read_simulation_data(the_file_path_name):
    the_variable_dictionary = {}
    the_simulation_properties = {}
    f = open(the_file_path_name, 'r')

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        the_simulation_properties['title'] = line
        line = f.readline()

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        line_data = line.split(': ')
        the_simulation_properties[line_data[0]] = line_data[1]
        line = f.readline()

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        line_data = line.split(': ')
        the_simulation_properties[line_data[0]] = float(line_data[1])
        line = f.readline()

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        line_data = line.split(': ')
        the_simulation_properties[line_data[0]] = float(line_data[1].split(' ')[0])
        line = f.readline()

    line = f.readline()
    a_variable = None
    an_uncertainty_set = None
    a_method = None
    while line != '':
        if line[0] != '\t':
            line = line[0:-2]
            line_data = line.split(' = ')
            a_variable = float(line_data[1][0:-1])
            the_variable_dictionary[a_variable] = {}
            line = f.readline()
        elif line[1] != '\t':
            a_method = line[1:-2]
            the_variable_dictionary[a_variable].update({a_method: {}})
            line = f.readline()
        elif line[2] != '\t':
            an_uncertainty_set = line[2:-2]
            the_variable_dictionary[a_variable][a_method].update({an_uncertainty_set: {}})
            line = f.readline()
        else:
            line = line[3:-1]
            line_data = line.split(': ')
            the_variable_dictionary[a_variable][a_method][an_uncertainty_set].update(
                {line_data[0]: float(line_data[1].split(' ')[0])})
            line = f.readline()
    return the_variable_dictionary, the_simulation_properties


def objective_proboffailure_vs_gamma(the_gammas, the_objective_values, the_objective_name, the_objective_units,
                                     the_prob_of_failure, the_title):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lines1 = ax1.plot(the_gammas, the_objective_values, 'r--', label=the_objective_name)
    lines2 = ax2.plot(the_gammas, the_prob_of_failure, 'b-', label='Probability of Failure')
    ax1.set_xlabel(r'Uncertainty Set Scaling Factor $\Gamma$', fontsize=18)
    ax1.set_ylabel(the_objective_name + ' (' + the_objective_units + ')', fontsize=18)
    ax2.set_ylabel("Probability of Failure", fontsize=18)
    plt.title(the_title, fontsize=18)
    lines = lines1 + lines2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper center", fontsize=18)
    plt.show()


def generate_comparison_plots(the_relative_objective_values, the_objective_name, the_relative_number_of_constraints,
                              the_relative_setup_times, the_relative_solve_times, the_uncertainty_set, the_methods):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = np.arange(len(the_methods))
    lines1 = ax1.bar(x + [0.2] * len(the_methods),
                     the_relative_objective_values,
                     [0.25] * len(the_methods), color='r', label=the_objective_name)
    lines2 = ax2.bar(x + [0.5] * len(the_methods),
                     the_relative_number_of_constraints,
                     [0.25] * len(the_methods), color='b', label='No. of Constraints')
    ax1.set_ylabel("Scaled Average " + the_objective_name, fontsize=18)
    ax2.set_ylabel("Scaled Number of Constraints", fontsize=18)
    plt.xticks(x + .45, the_methods, fontsize=18)
    plt.title(the_uncertainty_set.capitalize() + ' Uncertainty Set', fontsize=18)
    lines = [lines1, lines2]
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper left")
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = np.arange(len(the_methods))
    lines1 = ax1.bar(x + [0.2] * len(the_methods),
                     the_relative_setup_times,
                     [0.25] * len(the_methods), color='r', label='Setup Time')
    lines2 = ax2.bar(x + [0.5] * len(the_methods),
                     the_relative_solve_times,
                     [0.25] * len(the_methods), color='b', label='Solve Time')
    ax1.set_ylabel("Scaled Setup Time", fontsize=18)
    ax2.set_ylabel("Scaled Solve Time", fontsize=18)
    plt.xticks(x + .45, the_methods, fontsize=18)
    plt.title(the_uncertainty_set.capitalize() + ' Uncertainty Set', fontsize=18)
    lines = [lines1, lines2]
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper left")
    plt.show()


def generate_performance_vs_pwl_plots(the_numbers_of_linear_sections, the_method_performance_dictionary,
                                      the_objective_name, the_objective_units, the_uncertainty_set,
                                      worst_case_or_average):
    plt.figure()
    for method in the_method_performance_dictionary:
        plt.plot(the_numbers_of_linear_sections, the_method_performance_dictionary[method], label=method)
    plt.xlabel("Number of Piecewise-linear Sections", fontsize=18)
    plt.ylabel(the_objective_name + '(' + the_objective_units + ')', fontsize=18)
    plt.title('The ' + worst_case_or_average + ' Performance: ' + the_uncertainty_set.capitalize() + ' Uncertainty Set', fontsize=18)
    plt.legend(loc=0)
    plt.show()


def generate_all_plots(the_variable_gamma_file_path_name, the_variable_pwl_file_path_name):
    dictionary_gamma, properties_gamma = read_simulation_data(the_variable_gamma_file_path_name)

    the_gammas = dictionary_gamma.keys()
    the_gammas.sort()
    the_methods = dictionary_gamma.values()[0].keys()
    the_uncertainty_sets = dictionary_gamma.values()[0].values()[0].keys()
    for uncertainty_set in the_uncertainty_sets:
        for method in the_methods:
            objective_values = [dictionary_gamma[gamma][method][uncertainty_set]['Average performance'] for gamma in
                                the_gammas]
            prob_of_failure = [dictionary_gamma[gamma][method][uncertainty_set]['Probability of failure'] for gamma in
                               the_gammas]
            objective_proboffailure_vs_gamma(the_gammas, objective_values, properties_gamma['Objective'],
                                             properties_gamma['Units'],
                                             prob_of_failure,
                                             method + ' Formulation: ' + uncertainty_set.capitalize() + ' Uncertainty Set')

        rel_objective_values = [
            dictionary_gamma[the_gammas[-1]][method][uncertainty_set]['Relative average performance']
            for method in the_methods]
        rel_num_of_cons = [dictionary_gamma[the_gammas[-1]][method][uncertainty_set]['Relative number of constraints']
                           for method in the_methods]
        rel_setup_times = [dictionary_gamma[the_gammas[-1]][method][uncertainty_set]['Relative setup time']
                           for method in the_methods]
        rel_solve_times = [dictionary_gamma[the_gammas[-1]][method][uncertainty_set]['Relative solve time']
                           for method in the_methods]

        generate_comparison_plots(rel_objective_values, properties_gamma['Objective'], rel_num_of_cons, rel_setup_times,
                                  rel_solve_times, uncertainty_set, the_methods)

    dictionary_pwl, properties_pwl = read_simulation_data(the_variable_pwl_file_path_name)
    the_numbers_of_linear_sections = dictionary_pwl.keys()
    the_numbers_of_linear_sections.sort()
    the_methods = dictionary_pwl.values()[0].keys()
    the_uncertainty_sets = dictionary_pwl.values()[0].values()[0].keys()
    for uncertainty_set in the_uncertainty_sets:
        method_average_objective_dictionary = \
            {method: [dictionary_pwl[number_of_linear_sections][method][uncertainty_set]['Average performance']
                      for number_of_linear_sections in the_numbers_of_linear_sections] for method in the_methods}
        method_worst_objective_dictionary = \
            {method: [dictionary_pwl[number_of_linear_sections][method][uncertainty_set]['Worst-case performance']
                      for number_of_linear_sections in the_numbers_of_linear_sections] for method in the_methods}
        generate_performance_vs_pwl_plots(the_numbers_of_linear_sections, method_average_objective_dictionary,
                                          properties_pwl['Objective'], properties_pwl['Units'], uncertainty_set,
                                          'Average')
        generate_performance_vs_pwl_plots(the_numbers_of_linear_sections, method_worst_objective_dictionary,
                                          properties_pwl['Objective'], properties_pwl['Units'], uncertainty_set,
                                          'Worst-case')

if __name__ == '__main__':
    pass

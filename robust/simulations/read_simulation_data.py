import matplotlib.pyplot as plt
import numpy as np
import itertools


def read_simulation_data(file_path_name):
    variable_dictionary = {}
    simulation_properties = {}
    f = open(file_path_name, 'r')

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        simulation_properties['title'] = line
        line = f.readline()

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        line_data = line.split(': ')
        simulation_properties[line_data[0]] = line_data[1]
        line = f.readline()

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        line_data = line.split(': ')
        simulation_properties[line_data[0]] = float(line_data[1])
        line = f.readline()

    line = f.readline()
    while '-----------------' not in line:
        line = line[0:-1]
        line_data = line.split(': ')
        simulation_properties[line_data[0]] = float(line_data[1].split(' ')[0])
        line = f.readline()

    line = f.readline()
    a_variable = None
    an_uncertainty_set = None
    a_method = None
    while line != '':
        if line[0] != '\t':
            line = line[0:-2]
            line_data = line.split(' = ')
            a_variable = float(line_data[1])
            variable_dictionary[a_variable] = {}
            line = f.readline()
        elif line[1] != '\t':
            a_method = line[1:-2]
            variable_dictionary[a_variable].update({a_method: {}})
            line = f.readline()
        elif line[2] != '\t':
            an_uncertainty_set = line[2:-2]
            variable_dictionary[a_variable][a_method].update({an_uncertainty_set: {}})
            line = f.readline()
        else:
            line = line[3:-1]
            line_data = line.split(': ')
            variable_dictionary[a_variable][a_method][an_uncertainty_set].update(
                {line_data[0]: float(line_data[1].split(' ')[0])})
            line = f.readline()
    return variable_dictionary, simulation_properties


def objective_proboffailure_vs_gamma(gammas, objective_values, objective_name, objective_units, min_obj,
                                     max_obj, prob_of_failure, title, objective_stddev = None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lines1 = ax1.plot(gammas, objective_values, 'r--', label=objective_name)
    if objective_stddev:
        inds = np.nonzero(np.ones(len(gammas)) - prob_of_failure)[0]
        uppers = [objective_values[ind] + objective_stddev[ind] for ind in inds]
        lowers = [objective_values[ind] - objective_stddev[ind] for ind in inds]
        x = [gammas[ind] for ind in inds]
        ax1.fill_between(x, lowers, uppers,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    lines2 = ax2.plot(gammas, prob_of_failure, 'b-', label='Prob. of Fail.')
    ax1.set_xlabel(r'Uncertainty Set Scaling Factor $\Gamma$', fontsize=18)
    ax1.set_ylabel(objective_name + ' (' + objective_units.capitalize() + ')', fontsize=18)
    ax2.set_ylabel("Probability of Failure", fontsize=18)
    ax1.set_ylim([min_obj, max_obj])
    # ax2.set_ylim([0, 1])
    plt.title(title, fontsize=18)
    lines = lines1 + lines2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper center", fontsize=18, numpoints=1)
    plt.show()


def generate_comparison_plots(relative_objective_values, objective_name, relative_number_of_constraints,
                              relative_setup_times, relative_solve_times, uncertainty_set, methods):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = np.arange(len(methods))
    lines1 = ax1.bar(x + [0.2] * len(methods),
                     relative_objective_values,
                     [0.25] * len(methods), color='r', label=objective_name)
    lines2 = ax2.bar(x + [0.5] * len(methods),
                     relative_number_of_constraints,
                     [0.25] * len(methods), color='b', label='No. of Cons.')
    ax1.set_ylabel("Scaled Average Cost", fontsize=18)
    ax1.set_ylim([min(relative_objective_values) - 0.1*min(relative_objective_values),
                 max(relative_objective_values) + 0.1*max(relative_objective_values)])
    ax2.set_ylabel("Scaled Number of Constraints", fontsize=18)
    plt.xticks(x + .45, methods)
    ax1.tick_params(axis='x', which='major', labelsize=17)
    plt.title(uncertainty_set.capitalize() + ' Uncertainty Set', fontsize=18)
    lines = [lines1, lines2]
    labs = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labs, loc="lower right", ncol=1)
    leg.remove()
    ax2.add_artist(leg)
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = np.arange(len(methods))
    lines1 = ax1.bar(x + [0.2] * len(methods),
                     relative_setup_times,
                     [0.25] * len(methods), color='r', label='Setup Time')
    lines2 = ax2.bar(x + [0.5] * len(methods),
                     relative_solve_times,
                     [0.25] * len(methods), color='b', label='Solve Time')
    ax1.set_ylabel("Scaled Setup Time", fontsize=18)
    ax2.set_ylabel("Scaled Solve Time", fontsize=18)
    plt.xticks(x + .45, methods)
    ax1.tick_params(axis='x', which='major', labelsize=17)
    plt.title(uncertainty_set.capitalize() + ' Uncertainty Set', fontsize=18)
    lines = [lines1, lines2]
    labs = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labs, loc="lower right", ncol=1)
    leg.remove()
    ax2.add_artist(leg)
    plt.show()


def generate_performance_vs_pwl_plots(numbers_of_linear_sections, method_performance_dictionary,
                                      objective_name, objective_units, uncertainty_set,
                                      worst_case_or_average):
    plt.figure()
    marker = itertools.cycle(('s', '*', 'o', '.', ','))
    for method in method_performance_dictionary:
        plt.plot(numbers_of_linear_sections, method_performance_dictionary[method], marker=next(marker),
                 linestyle='', label=method)
    plt.xlabel("Number of Piecewise-linear Sections", fontsize=18)
    plt.ylabel(objective_name + '(' + objective_units + ')', fontsize=18)
    plt.title('The ' + worst_case_or_average + ' Performance: ' + uncertainty_set.capitalize() + ' Uncertainty Set',
              fontsize=18)
    plt.legend(loc=0, numpoints=1)
    plt.show()

def generate_variable_gamma_plots(variable_gamma_file_path_name):
    dictionary_gamma, properties_gamma = read_simulation_data(variable_gamma_file_path_name)

    gammas = dictionary_gamma.keys()
    gammas.sort()
    methods = dictionary_gamma.values()[0].keys()
    uncertainty_sets = dictionary_gamma.values()[0].values()[0].keys()
    min_obj = min([dictionary_gamma[gamma][method][uncertainty_set]['Average performance']
                       for gamma in gammas
                       for method in methods
                       for uncertainty_set in uncertainty_sets])

    max_obj = max([dictionary_gamma[gamma][method][uncertainty_set]['Average performance']
                       for gamma in gammas
                       for method in methods
                       for uncertainty_set in uncertainty_sets])
    for uncertainty_set in uncertainty_sets:
        for method in methods:
            objective_values = [dictionary_gamma[gamma][method][uncertainty_set]['Average performance'] for gamma in
                                gammas]
            prob_of_failure = [dictionary_gamma[gamma][method][uncertainty_set]['Probability of failure'] for gamma in
                               gammas]
            objective_proboffailure_vs_gamma(gammas, objective_values, properties_gamma['Objective'],
                                             properties_gamma['Units'], min_obj, max_obj,
                                             prob_of_failure,
                                             method + ' Formulation: ' + uncertainty_set.capitalize() + ' Uncertainty Set')

        rel_objective_values = [
            dictionary_gamma[gammas[-1]][method][uncertainty_set]['Relative average performance']
            for method in methods]
        rel_num_of_cons = [dictionary_gamma[gammas[-1]][method][uncertainty_set]['Relative number of constraints']
                           for method in methods]
        rel_setup_times = [dictionary_gamma[gammas[-1]][method][uncertainty_set]['Relative setup time']
                           for method in methods]
        rel_solve_times = [dictionary_gamma[gammas[-1]][method][uncertainty_set]['Relative solve time']
                           for method in methods]

        generate_comparison_plots(rel_objective_values, properties_gamma['Objective'], rel_num_of_cons, rel_setup_times,
                                  rel_solve_times, uncertainty_set, methods)

def generate_variable_pwl_plots(variable_pwl_file_path_name):
    dictionary_pwl, properties_pwl = read_simulation_data(variable_pwl_file_path_name)
    numbers_of_linear_sections = dictionary_pwl.keys()
    numbers_of_linear_sections.sort()
    methods = dictionary_pwl.values()[0].keys()
    uncertainty_sets = dictionary_pwl.values()[0].values()[0].keys()
    for uncertainty_set in uncertainty_sets:
        method_average_objective_dictionary = \
            {method: [dictionary_pwl[number_of_linear_sections][method][uncertainty_set]['Average performance']
                      for number_of_linear_sections in numbers_of_linear_sections] for method in methods}
        method_worst_objective_dictionary = \
            {method: [dictionary_pwl[number_of_linear_sections][method][uncertainty_set]['Worst-case performance']
                      for number_of_linear_sections in numbers_of_linear_sections] for method in methods}
        generate_performance_vs_pwl_plots(numbers_of_linear_sections, method_average_objective_dictionary,
                                          properties_pwl['Objective'], properties_pwl['Units'], uncertainty_set,
                                          'Average')
        generate_performance_vs_pwl_plots(numbers_of_linear_sections, method_worst_objective_dictionary,
                                          properties_pwl['Objective'], properties_pwl['Units'], uncertainty_set,
                                          'Worst-case')

def generate_all_plots(variable_gamma_file_path_name, variable_pwl_file_path_name):
    generate_variable_gamma_plots(variable_gamma_file_path_name)
    generate_variable_pwl_plots(variable_pwl_file_path_name)

if __name__ == '__main__':
    pass

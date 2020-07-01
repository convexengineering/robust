


from robust.synthetic_model import models

from robust.robust import RobustModel


def robustify_synthetic_model(the_model, is_two_term, is_boyd, is_simple_model, the_uncertainty_set,
                              the_min_number_of_linear_sections=3, the_max_number_of_linear_sections=99,
                              the_verbosity=0,
                              the_linearization_tolerance=1e-3):
    the_robust_model = RobustModel(the_model, the_uncertainty_set, twoTerm=is_two_term, boyd=is_boyd,
                                   simpleModel=is_simple_model)
    the_robust_model_solution = the_robust_model.robustsolve(verbosity=the_verbosity,
                                                             linearizationTolerance=the_linearization_tolerance,
                                                             minNumOfLinearSections=the_min_number_of_linear_sections,
                                                             maxNumOfLinearSections=the_max_number_of_linear_sections)
    return the_robust_model, the_robust_model_solution


def print_robust_results(the_robust_model, the_robust_model_solution, the_nominal_solution, the_method_name):
    print(
        '\n' + the_robust_model.type_of_uncertainty_set + ' uncertainty using ' + the_method_name + ' formulation: \n' +
        ' \t cost : %s \n \t relative cost : %s \n  \t number of constraints : %s \n \t setup time : %s \n \t solve time : %s'
        % (the_robust_model_solution['cost'], the_robust_model_solution['cost'] / the_nominal_solution['cost'],
           len([cs for cs in the_robust_model.get_robust_model().flat()]),
           the_robust_model_solution['setuptime'], the_robust_model_solution['soltime']))


if __name__ == '__main__':
    method_names = {'Best Pairs': {'twoTerm': True, 'boyd': False, 'simpleModel': False},
                    'Linearized Perturbations': {'twoTerm': False, 'boyd': False, 'simpleModel': False},
                    'Simple Conservative': {'twoTerm': False, 'boyd': False, 'simpleModel': True},
                    'Two Term': {'twoTerm': False, 'boyd': True, 'simpleModel': False}}
    uncertainty_sets = ['box', 'elliptical']

    # a_synthetic_model = Models.synthetic_model(15)
    model = models.test_synthetic_model()
    nominal_solution = model.solve(verbosity=0)
    print('nominal cost = %s' % nominal_solution['cost'])

    for method_name in method_names:
        for uncertainty_set in uncertainty_sets:
            robust_model, robust_model_solution = robustify_synthetic_model(model, method_names[method_name]['twoTerm'],
                                                                            method_names[method_name]['boyd'],
                                                                            method_names[method_name]['simpleModel'],
                                                                            uncertainty_set)
            print_robust_results(robust_model, robust_model_solution, nominal_solution, method_name)

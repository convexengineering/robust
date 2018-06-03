import GPModels as Models
from Robust import RobustModel

# model = Models.synthetic_model(15)
model = Models.test_synthetic_model()
nominal_solution = model.solve(verbosity=0)
print('nominal cost = %s' % nominal_solution['cost'])

uncertain_exps_box = RobustModel(model, 'box')
uncertain_exps_box_solution = uncertain_exps_box.robustsolve(verbosity=0, linearizationTolerance=1e-3,
                                                             minNumOfLinearSections=10, maxNumOfLinearSections=99)
print('box uncertain exponents: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
      % (uncertain_exps_box_solution['cost'], uncertain_exps_box_solution['cost']/nominal_solution['cost'],
         len([cs for cs in uncertain_exps_box.get_robust_model().flat(constraintsets=False)]),
         uncertain_exps_box_solution['setuptime'], uncertain_exps_box_solution['soltime']))

uncertain_coeffs_box = RobustModel(model, 'box', twoTerm=False)
uncertain_coeffs_box_solution = uncertain_coeffs_box.robustsolve(verbosity=0, minNumOfLinearSections=10, maxNumOfLinearSections=40)
print('box uncertain coefficients: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
      % (uncertain_coeffs_box_solution['cost'], uncertain_coeffs_box_solution['cost']/nominal_solution['cost'],
         len([cs for cs in uncertain_coeffs_box.get_robust_model().flat(constraintsets=False)]),
         uncertain_coeffs_box_solution['setuptime'], uncertain_coeffs_box_solution['soltime']))

boyd_box = RobustModel(model, 'box', boyd=True)
boyd_box_solution = boyd_box.robustsolve(verbosity=0, minNumOfLinearSections=10, maxNumOfLinearSections=99)
print('box boyd: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
      % (boyd_box_solution['cost'], boyd_box_solution['cost']/nominal_solution['cost'],
         len([cs for cs in boyd_box.get_robust_model().flat(constraintsets=False)]),
         boyd_box_solution['setuptime'], boyd_box_solution['soltime']))

simple_box = RobustModel(model, 'box', simpleModel=True)
simple_box_solution = simple_box.robustsolve(verbosity=0, minNumOfLinearSections=10, maxNumOfLinearSections=99)
print('simple box: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
      % (simple_box_solution['cost'], simple_box_solution['cost']/nominal_solution['cost'],
         len([cs for cs in simple_box.get_robust_model().flat(constraintsets=False)]),
         simple_box_solution['setuptime'], simple_box_solution['soltime']))

uncertain_exps_elliptical = RobustModel(model, 'elliptical')
uncertain_exps_elliptical_solution = uncertain_exps_elliptical.robustsolve(verbosity=0, minNumOfLinearSections=10, maxNumOfLinearSections=99)
print('elliptical uncertain exponents: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
      % (uncertain_exps_elliptical_solution['cost'], uncertain_exps_elliptical_solution['cost']/nominal_solution['cost'],
         len([cs for cs in uncertain_exps_elliptical.get_robust_model().flat(constraintsets=False)]),
         uncertain_exps_elliptical_solution['setuptime'], uncertain_exps_elliptical_solution['soltime']))

# uncertain_coeffs_elliptical = RobustModel(model, 'elliptical', twoTerm=False)
# uncertain_coeffs_elliptical_solution = uncertain_coeffs_elliptical.robustsolve(verbosity=0, minNumOfLinearSections=10, maxNumOfLinearSections=40)
# print('elliptical uncertain coefficients: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
#       % (uncertain_coeffs_elliptical_solution['cost'], uncertain_coeffs_elliptical_solution['cost']/nominal_solution['cost'],
#          len([cs for cs in uncertain_coeffs_elliptical.get_robust_model().flat(constraintsets=False)]),
#          uncertain_coeffs_elliptical_solution['setuptime'], uncertain_coeffs_elliptical_solution['soltime']))

boyd_elliptical = RobustModel(model, 'elliptical', boyd=True)
boyd_elliptical_solution = boyd_elliptical.robustsolve(verbosity=0, minNumOfLinearSections=10, maxNumOfLinearSections=99)
print('elliptical boyd: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
      % (boyd_elliptical_solution['cost'], boyd_elliptical_solution['cost']/nominal_solution['cost'],
         len([cs for cs in boyd_elliptical.get_robust_model().flat(constraintsets=False)]),
         boyd_elliptical_solution['setuptime'], boyd_elliptical_solution['soltime']))

simple_elliptical = RobustModel(model, 'elliptical', simpleModel=True)
simple_elliptical_solution = simple_elliptical.robustsolve(verbosity=0, minNumOfLinearSections=10, maxNumOfLinearSections=99)
print('simple elliptical: cost = %s, relative cost = %s, number of constraints = %s, setup time = %s, solve time = %s'
      % (simple_elliptical_solution['cost'], simple_elliptical_solution['cost']/nominal_solution['cost'],
         len([cs for cs in simple_elliptical.get_robust_model().flat(constraintsets=False)]),
         simple_elliptical_solution['setuptime'], simple_elliptical_solution['soltime']))

import EquivalentModels
from RobustifyLargePosynomial import RobustifyLargePosynomial
from gpkit import Model, Variable, Monomial, SignomialsEnabled

RLP = RobustifyLargePosynomial.RobustifyLargePosynomial()


class UncertainCoefficientsModel(EquivalentModels.EquivalentModel):
    """
    Creates robust Models starting from a model with uncertain coefficients
    """

    def setup(self, gamma, type_of_uncertainty_set, r_min=5, tol=0.001,
              simple_model=False, number_of_regression_points=2,
              linearize_two_term=True, enable_sp=True, boyd=False, *two_term):
        """
        Constructs a robust model starting from a model with uncertain coefficients
        :param gamma: Controls the size of the uncertainty set
        :param type_of_uncertainty_set: box, elliptical, or one norm set
        :param r_min: The minimum number of PWL functions
        :param tol: Determines the accuracy of PWL
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param number_of_regression_points: The number of points per dimension used to replace exponential uncertainty
        function by a linear function
        :param two_term: Solve the problem using two term decoupling rather than linear approximation of exponential
        uncertainties. If the problem is small or the number of uncertain variables per constraint is small,
        switch two_term to True
        :param linearize_two_term: linearize two term functions rather than considering them large posynomials
        :param enable_sp: choose to solve an SP to get a better solution
        :param boyd: choose to apply boyd's two term approximation
        :return: The robust Model, The initial guess if the robust model is an SP, and the number of PWL functions used
        to approximate two term monomials
        """
        r = r_min
        error = 1
        sol = 0

        if len(two_term) == 0:
            if type_of_uncertainty_set == 'box' or type_of_uncertainty_set == 'one norm':
                two_term = True
            elif type_of_uncertainty_set == 'elliptical':
                two_term = False
        else:
            two_term = two_term[0]

        model_upper, model_lower = \
            self.robust_model_fixed_r(gamma, r, type_of_uncertainty_set,
                                      tol, simple_model, number_of_regression_points,
                                      two_term, linearize_two_term, False, boyd)
        while r <= 20 and error > tol:
            flag = 0
            try:
                sol_upper = model_upper.solve(verbosity=0)
                sol = sol_upper
            except:
                flag = 1
            try:
                sol_lower = model_lower.solve(verbosity=0)
            except RuntimeError:
                r = 21
                sol = self.solve(verbosity=0)
                break

            if flag != 1:
                try:
                    error = (sol_upper.get('cost').m -
                             sol_lower.get('cost').m) / (0.0 + sol_lower.get('cost').m)
                except:
                    error = (sol_upper.get('cost') -
                             sol_lower.get('cost')) / (0.0 + sol_lower.get('cost'))
            r += 1
            model_upper, model_lower = \
                self.robust_model_fixed_r(gamma, r, type_of_uncertainty_set, tol,
                                          simple_model, number_of_regression_points,
                                          two_term, linearize_two_term, False, boyd)
        initial_guess = sol.get("variables")

        if enable_sp:
            model_upper, model_lower = \
                self.robust_model_fixed_r(gamma, r - 1, type_of_uncertainty_set,
                                          tol, False, number_of_regression_points,
                                          False, linearize_two_term, True, boyd)
            subs_vars = model_upper.substitutions.keys()

            for i in xrange(len(subs_vars)):
                del initial_guess[subs_vars[i].key]

        return model_upper, initial_guess, r

    def robust_model_fixed_r(self, gamma, r, type_of_uncertainty_set,
                             tol, simple_model, number_of_regression_points, two_term,
                             linearize_two_term, enable_sp, boyd):
        """
        generates a robust model for a fixed number of piece-wise linear functions
        :param gamma: Controls the size of the uncertainty set
        :param r: the number of piece-wise linear functions per two term constraints
        :param type_of_uncertainty_set: box, elliptical, or one norm set
        :param tol: Determines the accuracy of PWL
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param number_of_regression_points: The number of points per dimension used to
        replace exponential uncertainty
        function by a linear function
        :param two_term: Solve the problem using two term decoupling rather than linear
        approximation of exponential uncertainties. If the problem is small or the
        number of uncertain variables per constraint is small, switch two_term to True
        :param linearize_two_term: linearize two term functions rather than considering
        them large posynomials
        :param enable_sp: choose to solve an SP to get a better solution
        :param boyd: choose to apply boyd's two term approximation
        :return: the upper-bound of the robust model and the lower-bound of the robust model
        """

        if type_of_uncertainty_set == 'box':
            dependent_uncertainty_set = False
        else:
            dependent_uncertainty_set = True

        uncertain_vars = self.uncertain_model_variables()

        simplified_model_upper, simplified_model_lower, number_of_no_data_constraints \
            = self.tractableModel(r, tol, uncertain_vars, simple_model,
                                  dependent_uncertainty_set, two_term,
                                  linearize_two_term, boyd)

        no_data_constraints_upper, no_data_constraints_lower = [], []
        data_constraints, data_monomials = [], []

        posynomials_upper = simplified_model_upper.as_posyslt1()
        posynomials_lower = simplified_model_lower.as_posyslt1()

        for i, p in enumerate(posynomials_upper):
            if i < number_of_no_data_constraints:
                no_data_constraints_upper += [p <= 1]
                no_data_constraints_lower += [posynomials_lower[i] <= 1]
            else:
                if len(p.exps) > 1:
                    data_constraints.append(p.RLP.robustify_large_posynomial
                                            (type_of_uncertainty_set, uncertain_vars, i,
                                             enable_sp, number_of_regression_points))
                else:
                    data_monomials.append(p)

        exps_of_uncertain_vars = UncertainCoefficientsModel. \
            uncertain_variables_exponents(data_monomials, uncertain_vars)

        if exps_of_uncertain_vars.size > 0:
            centering_vector, scaling_vector = UncertainCoefficientsModel. \
                normalize_perturbation_vector(uncertain_vars)
            coefficient = UncertainCoefficientsModel. \
                construct_robust_monomial_coefficients(exps_of_uncertain_vars, gamma,
                                                       type_of_uncertainty_set,
                                                       centering_vector, scaling_vector)
            for i in xrange(len(data_monomials)):
                data_constraints += [coefficient[i][0] * data_monomials[i] <= 1]
        output_upper = Model(self.cost,
                             [no_data_constraints_upper, data_constraints])
        output_upper.substitutions.update(self.substitutions)
        output_lower = Model(self.cost,
                             [no_data_constraints_lower, data_constraints])
        output_lower.substitutions.update(self.substitutions)
        return output_upper, output_lower

#
# def solveRobustSPBox(model,Gamma,relTol = 1e-5):
#    initSol = model.localsolve(verbosity=0)
#    try:
#        initCost = initSol['cost'].m
#    except:
#        initCost = initSol['cost']
#    newCost = initCost*(1 + 2*relTol)
#    while (np.abs(initCost - newCost)/initCost) > relTol:
#        apprModel = Model(model.cost,model.as_gpconstr(initSol))
#        robModel = robustModelBoxUncertainty(apprModel,Gamma)[0]
#        sol = robModel.solve(verbosity=0)
#        initSol = sol.get('variables')
#        initCost = newCost
#        try:
#            newCost = sol['cost'].m
#        except:
#            newCost = sol['cost']
#        print(newCost)
#    return initSol

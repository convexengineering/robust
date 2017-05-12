import numpy as np
import EquivalentModels as EM
import EquivalentPosynomial as EP
from gpkit import  Model
from gpkit import Variable, Monomial, SignomialsEnabled
from sklearn import linear_model


class UncertainCoefficientsModel(EquivalentModel):
    """
    Creates robust Models starting from a model with uncertain coefficients
    """
    def setup(self, gamma, type_of_uncertainty_set, r_min = 5, tol=0.001,
              number_of_regression_points = 2, linearize_two_term = True,
              enable_sp = True, *two_term):
        """
        Constructs a robust model starting from a model with uncertain coefficients
        :param gamma: Controls the size of the uncertainty set
        :param type_of_uncertainty_set: box, elliptical, or one norm set
        :param r_min: The minimum number of PWL functions
        :param tol: Determines the accuracy of PWL
        :param number_of_regression_points: The number of points per dimension used to replace exponential uncertainty
        function by a linear function
        :param two_term: Solve the problem using two term decoupling rather than linear approximation of exponential
        uncertainties. If the problem is small or the number of uncertain variables per constraint is small,
        switch two_term to True
        :param linearize_two_term: linearize two term functions rather than considering them large posynomials
        :param enable_sp: choose to solve an SP to get a better solution
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
             two_term =  two_term[0]

        model_upper, model_lower = \
            self.robust_model_fixed_r(gamma, r, type_of_uncertainty_set,
                                              tol, number_of_regression_points,
                                              two_term, linearize_two_term,
                                              False)
        while r <= 20 and error > tol:
            flag = 0
            try:
                sol_upper = model_upper.solve(verbosity = 0)
                sol = sol_upper
            except:
                flag = 1
            try:
                sol_lower = model_lower.solve(verbosity = 0)
            except:
                r=21
                sol = self.solve(verbosity = 0)
                break
            
            if flag != 1:    
                try:
                    error = \
                    (sol_upper.get('cost').m -
                     sol_lower.get('cost').m)/(0.0 + sol_lower.get('cost').m)
                except:
                    error = (sol_upper.get('cost') -
                             sol_lower.get('cost'))/(0.0 + sol_lower.get('cost'))
            r += 1
            model_upper, model_lower = \
                self.robust_model_fixed_r(gamma, r, type_of_uncertainty_set,
                                                  tol, number_of_regression_points,
                                                  two_term, linearize_two_term,
                                                  False)
            
        initial_guess = sol.get("variables")
        
        if enable_sp:
            model_upper, model_lower = \
                self.robust_model_fixed_r(gamma, r-1, type_of_uncertainty_set,
                                                  tol, number_of_regression_points,
                                                  False, linearize_two_term, True)
            subs_vars = model_upper.substitutions.keys()
            
            for i in xrange(len(subs_vars)):
                del initial_guess[subs_vars[i].key]

        return model_upper, initial_guess, r


    @staticmethod
    def __merge_mesh_grid(array,n):
        """
        A method used in perturbation_function method, allows easy computation of the output at the regression points
        :param array: The multidimensional array we need to make simpler (1D)
        :param n: The total number of interesting points
        :return: The simplified array
        """
        if n == 1:
            return [array]
        else:
            output = []
            for i in xrange(len(array)):
                output = output + UncertainCoefficientsModel.merge_mesh_grid(array[i],n/(len(array) + 0.0))
            return output 


    @staticmethod
    def perturbation_function(perturbation_vector,number_of_regression_points):
        """
        A method used to do the linear regression
        :param perturbation_vector: A list representing the perturbation associated with each uncertain parameter
        :param number_of_regression_points: The number of regression points per dimension
        :return: the regression coefficients and intercept
        """
        dim = len(perturbation_vector)
        if dim != 1:
            x = np.meshgrid(*[np.linspace(-1,1,number_of_regression_points)]*dim)
        else:
            x = [np.linspace(-1,1,number_of_regression_points)]

        result, input_list= [], []
        for _ in xrange(number_of_regression_points**dim):
            input_list.append([])

        for i in xrange(dim):
            temp = UncertainCoefficientsModel.merge_mesh_grid(x[i],number_of_regression_points**dim)
            for j in xrange(number_of_regression_points**dim):
                input_list[j].append(temp[j])

        for i in xrange(number_of_regression_points**dim):
            output = 1
            for j in xrange(dim):
                if perturbation_vector[j] != 0:
                    output = output*perturbation_vector[j]**input_list[i][j]
            result.append(output)

        clf = linear_model.LinearRegression()
        clf.fit(input_list,result)
        return clf.coef_, clf.intercept_


    @staticmethod
    def linearize_perturbations(p, p_uncertain_vars, number_of_regression_points):
        """
        A method used to linearize uncertain exponential functions
        :param p: The posynomial containing uncertain parameters
        :param p_uncertain_vars: the uncertain variables in the posynomial
        :param number_of_regression_points: The number of regression points per dimension
        :return: The linear regression of all the exponential functions, and the mean vector
        """
        center, scale = [], []
        mean_vector = []
        coeff, intercept = [],[]

        for i in xrange(len(p_uncertain_vars)):
            pr = p_uncertain_vars[i].key.pr
            center.append(np.sqrt(1 - pr**2/10000.0))
            scale.append(0.5*np.log((1 + pr/100.0)/(1 - pr/100.0)))

        perturbation_matrix = []
        for i in xrange(len(p.exps)):
            perturbation_matrix.append([])
            mon_uncertain_vars = [var for var in p_uncertain_vars if var in p.exps[i]]
            mean = 1
            for j,var in enumerate(p_uncertain_vars):
                if var.key in mon_uncertain_vars:
                    mean = mean*center[j]**(p.exps[i].get(var.key))
                    perturbation_matrix[i].append(np.exp(p.exps[i].get(var.key)*scale[j]))
                else:
                    perturbation_matrix[i].append(0)
                coeff.append([])
                intercept.append([])
                coeff[i],intercept[i] = UncertainCoefficientsModel.perturbation_function(perturbation_matrix[i], number_of_regression_points)
            mean_vector.append(mean)

        return coeff, intercept, mean_vector

    @staticmethod
    def no_coefficient_monomials (p):
        """
        separates the monomials in a posynomial into a list of monomials
        :param p: The posynomial to separate
        :return: The list of monomials
        """
        monomials = []
        for i in xrange(len(p.exps)):
            monomials.append(Monomial(p.exps[i],p.cs[i]))
        return monomials


    @staticmethod
    def generate_robust_constraints(type_of_uncertainty_set,
                                    monomials, perturbation_matrix,
                                    intercept, mean_vector, enable_sp, m):
        """

        :param type_of_uncertainty_set: box, elliptical, or one norm
        :param monomials: the list of monomials
        :param perturbation_matrix: the matrix of perturbations
        :param intercept: the list of intercepts
        :param mean_vector: the list of means
        :param enable_sp: whether or not we prefer sp solutions
        :param m: the index
        :return: the robust set of constraints
        """
        constraints = []
        s_main = Variable("s_%s"%m)
        if type_of_uncertainty_set == 'box' or 'one norm':
            constraints += [sum([a*b for a,b in
                             zip([a*b for a,b in
                                  zip(mean_vector,intercept)],monomials)]) + s_main <= 1]
        elif type_of_uncertainty_set == 'elliptical':
            constraints += [sum([a*b for a,b in
                                 zip([a*b for a,b in
                                      zip(mean_vector,intercept)],monomials)]) + s_main**0.5 <= 1]
        ss = []
        for i in xrange(len(perturbation_matrix[0])):
            positive_pert = []
            negative_pert = []
            positive_monomials = []
            negative_monomials = []
            if type_of_uncertainty_set == 'box' or 'elliptical':
                s = Variable("s^%s_%s"%(i,m))
                ss.append(s)
            else:
                s = s_main
            for j in xrange(len(perturbation_matrix)):
                if perturbation_matrix[j][i] > 0:
                    positive_pert.append(mean_vector[j]*perturbation_matrix[j][i])
                    positive_monomials.append(monomials[j])
                elif perturbation_matrix[j][i] < 0:
                    negative_pert.append(-mean_vector[j]*perturbation_matrix[j][i])
                    negative_monomials.append(monomials[j])
            if enable_sp:
                with SignomialsEnabled():
                    if type_of_uncertainty_set == 'box' or 'one norm':
                        if negative_pert and not positive_pert:
                            constraints += [sum([a*b for a,b in
                                                 zip(negative_pert,negative_monomials)])<= s]
                        elif positive_pert and not negative_pert:
                            constraints += [sum([a*b for a,b in
                                                 zip(positive_pert,positive_monomials)])<= s]
                        else:
                            constraints += [sum([a*b for a,b in
                                                 zip(positive_pert,positive_monomials)]) -
                                            sum([a*b for a,b in
                                                 zip(negative_pert,negative_monomials)])<= s]
                            constraints += [sum([a*b for a,b in
                                                 zip(negative_pert,negative_monomials)]) -
                                            sum([a*b for a,b in
                                                 zip(positive_pert,positive_monomials)])<= s]
                    elif type_of_uncertainty_set == 'elliptical':
                        constraints += [(sum([a*b for a,b in
                                              zip(positive_pert,positive_monomials)])
                                             - sum([a*b for a,b in
                                                    zip(negative_pert,negative_monomials)]))**2]
            else:
                if type_of_uncertainty_set == 'box' or 'one norm':
                    if positive_pert:
                        constraints += [sum([a*b for a,b in
                                             zip(positive_pert,positive_monomials)]) <= s]
                    if negative_pert:
                        constraints += [sum([a*b for a,b in
                                             zip(negative_pert,negative_monomials)]) <= s]
                elif type_of_uncertainty_set == 'elliptical':
                    constraints += [sum([a*b for a,b in
                                         zip(positive_pert,positive_monomials)])**2
                                    + sum([a*b for a,b in
                                           zip(negative_pert,negative_monomials)])**2 <= s]
        if type_of_uncertainty_set == 'box' or 'elliptical':
            constraints.append(sum(ss) <= s_main)
        return constraints




    @staticmethod
    def robustify_large_posynomial(p, type_of_uncertainty_set, uncertain_vars, m,
                                   enable_sp, number_of_regression_points):
        """
        generate a safe approximation for large posynomials with uncertain coefficients
        :param p: The posynomial containing uncertain parameters
        :param type_of_uncertainty_set: 'box', elliptical, or 'one norm'
        :param uncertain_vars: Model's uncertain variables
        :param m: Index
        :param enable_sp: choose whether an sp compatible model is okay
        :param number_of_regression_points: number of regression points per dimension
        :return: set of robust constraints
        """
        p_uncertain_vars = [var for var in p.varkeys if var in uncertain_vars]

        perturbation_matrix, intercept, mean_vector = \
            UncertainCoefficientsModel.\
                linearize_perturbations(p, p_uncertain_vars, number_of_regression_points)

        if not p_uncertain_vars:
            return [p <= 1]

        monomials = UncertainCoefficientsModel.no_coefficient_monomials(p)
        constraints = UncertainCoefficientsModel.generate_robust_constraints(type_of_uncertainty_set, monomials,
                                                  perturbation_matrix, intercept, mean_vector, enable_sp, m)
        return constraints


    def robust_model_fixed_r(self, gamma, r, type_of_uncertainty_set,
                                     tol, number_of_regression_points,
                                     two_term, linearize_two_term, enable_sp):
        
        if type_of_uncertainty_set == 'box':
            dependent_uncertainty_set = False
        else:
            dependent_uncertainty_set = True
        
        simplified_model_upper, simplified_model_lower, number_of_no_data_constraints \
            = self.tractableModel(r, tol, dependent_uncertainty_set,
                                two_term, linearize_two_term)
        
        no_data_constraints_upper, no_data_constraints_lower = [],[]
        data_constraints, data_monomails = [],[]
        
        uncertain_vars = self.uncertain_model_variables()
        posynomials_upper = simplified_model_upper.as_posyslt1()
        posynomials_lower = simplified_model_lower.as_posyslt1()
        
        for i,p in enumerate(posynomials_upper):
            if i < number_of_no_data_constraints:
                no_data_constraints_upper += [p <= 1]
                no_data_constraints_lower += [posynomials_lower[i] <= 1]
            else:
                if len(p.exps) > 1:
                    data_constraints.append(EP.safePosynomialBoxUncertainty
                                           (p, uncertain_vars, i, enable_sp,
                                            number_of_regression_points))
                else:
                    data_monomails.append(p)

        exps_of_uncertain_vars = UncertainCoefficientsModel.uncertain_variables_exponents(data_monomails,
                                                           uncertain_vars)
        
        if exps_of_uncertain_vars.size > 0:
            centering_vector, scaling_vector = \
            UncertainCoefficientsModel.normalize_perturbation_vector(uncertain_vars)
            coefficient = \
            UncertainCoefficientsModel.construct_robust_monomail_coefficients(exps_of_uncertain_vars, gamma,
                                                type_of_uncertainty_set,
                                                centering_vector, scaling_vector)
            for i in xrange(len(data_monomails)):
                data_constraints += [coefficient[i][0]*data_monomails[i] <= 1]
        outputUpper = Model(self.cost,
                            [no_data_constraints_upper,data_constraints])
        outputUpper.substitutions.update(self.substitutions)
        outputLower = Model(self.cost,
                            [no_data_constraints_lower,data_constraints])
        outputLower.substitutions.update(self.substitutions)
        return outputUpper, outputLower
        

    @staticmethod
    def uncertain_variables_exponents(data_monomails, uncertain_vars):
        exps_of_uncertain_vars = \
            np.array([[-p.exps[0].get(var.key, 0) for var in uncertain_vars]
                   for p in data_monomails])
        return  exps_of_uncertain_vars
        

    def normalizePerturbationVector(uncertainVars):
        prs = np.array([var.key.pr for var in uncertainVars])
        etaMax = np.log(1 + prs/100.0)
        etaMin = np.log(1 - prs/100.0)
        centeringVector = (etaMin + etaMax)/2.0
        scalingVector = etaMax - centeringVector
        return centeringVector, scalingVector
        

    def constructRobustMonomailCoefficients(RHS_Coeff_Uncertain, Gamma,
                                            typeOfSet, centeringVector, 
                                            scalingVector):
        b_purt = (RHS_Coeff_Uncertain * scalingVector[np.newaxis])       
        coefficient = []
        for i in xrange(RHS_Coeff_Uncertain.shape[0]):
            norm = 0
            centering = 0
            for j in range(RHS_Coeff_Uncertain.shape[1]):
                if typeOfSet == 'box':
                    norm = norm + np.abs(b_purt[i][j])
                elif typeOfSet == 'elliptical':
                    norm = norm + b_purt[i][j]**2
                elif typeOfSet == 'one norm':
                    norm = max(norm,np.abs(b_purt[i][j]))
                else:
                    raise Exception('This type of set is not supported')
                centering = centering + RHS_Coeff_Uncertain[i][j] * \
                                                            centeringVector[j]
            if typeOfSet == 'elliptical':
                norm = np.sqrt(norm)
            coefficient.append([np.exp(Gamma*norm)/np.exp(centering)])
        return coefficient
        
    
def robustModelBoxUncertaintyUpperLower(model, Gamma,r,tol, 
                                        numberOfRegressionPoints = 4, 
                                        coupled = True, twoTerm = True, 
                                        linearizeTwoTerm = True, 
                                        enableSP = True):
    simplifiedModelUpper, simplifiedModelLower, numberOfNoDataConstraints = \
                EM.tractableModel(model,r,tol,coupled, False, twoTerm, 
                                  linearizeTwoTerm)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    uncertainVars = EM.uncertainModelVariables(model)
    posynomialsUpper = simplifiedModelUpper.as_posyslt1()
    posynomialsLower = simplifiedModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        #print(i)
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + \
                                            [posynomialsLower[i] <= 1]
        else:
            if len(p.exps) > 1:
                dataConstraints.append(EP.safePosynomialBoxUncertainty
                                       (p, uncertainVars, i, enableSP, 
                                        numberOfRegressionPoints))
            else:
                dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, 
                                                       uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = \
                                normalizePerturbationVector(uncertainVars)
        coefficient = constructRobustMonomailCoefficients(expsOfUncertainVars, 
                                                          Gamma, 'box', 
                                                          centeringVector, 
                                                          scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + \
                                    [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def robustModelBoxUncertainty(model, Gamma, tol=0.001, 
                              numberOfRegressionPoints = 4, coupled = True, 
                              twoTerm = True, linearizeTwoTerm = True, 
                              enableSP = True):
    r = 2
    error = 1
    sol = 0
    flag = 0 
    while r <= 20 and error > 0.01:
        flag = 0
        #print(r)
        modelUpper, modelLower = \
            robustModelBoxUncertaintyUpperLower(model, Gamma,r,tol, 
                                                numberOfRegressionPoints, 
                                                coupled, twoTerm, 
                                                linearizeTwoTerm, False)
        try:
            solUpper = modelUpper.solve(verbosity = 0)
            sol = solUpper
        except:
            flag = 1
        try:
            solLower = modelLower.solve(verbosity = 0)
        except:
            print("infeasible")
            r=20
            sol = model.solve(verbosity = 0)
            break
        if flag != 1:    
            try:
                error = \
                    (solUpper.get('cost').m - 
                     solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
            except:
                error = (solUpper.get('cost') - 
                         solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    initialGuess = sol.get("variables")
    if enableSP:
        modelUpper, modelLower = \
            robustModelBoxUncertaintyUpperLower(model,r-1,tol, 
                                                numberOfRegressionPoints, 
                                                coupled, False, 
                                                linearizeTwoTerm, True)
        subsVars = modelUpper.substitutions.keys()
        for i in xrange(len(subsVars)):
            del initialGuess[subsVars[i].key]
    return modelUpper, initialGuess, r
    
def robustModelEllipticalUncertaintyUpperLower(model, Gamma,r,tol, 
                                               numberOfRegressionPoints = 4, 
                                               coupled = True, 
                                               dependentUncertainties = True, 
                                               twoTerm = False, 
                                               linearizeTwoTerm = True, 
                                               enableSP = True):
    simplifiedModelUpper, simplifiedModelLower, numberOfNoDataConstraints = \
        EM.tractableModel(model,r,tol,coupled,dependentUncertainties, 
                          twoTerm, linearizeTwoTerm)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    uncertainVars = EM.uncertainModelVariables(model)
    posynomialsUpper = simplifiedModelUpper.as_posyslt1()
    posynomialsLower = simplifiedModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + \
                                            [posynomialsLower[i] <= 1]
        else:
            if len(p.exps) > 1:
                dataConstraints.append(EP.safePosynomialEllipticalUncertainty
                                       (p, uncertainVars, i, enableSP, 
                                        numberOfRegressionPoints))
            else:
                dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, 
                                                       uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = \
                                    normalizePerturbationVector(uncertainVars)
        coefficient = \
            constructRobustMonomailCoeffiecientsEllipticalUncertainty
            (expsOfUncertainVars, Gamma, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + \
                                    [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def robustModelEllipticalUncertainty(model, Gamma, tol = 0.001, 
                                     numberOfRegressionPoints = 4 , 
                                     coupled = True, 
                                     dependentUncertainties = True, 
                                     twoTerm = False, linearizeTwoTerm = True, 
                                     enableSP = True):
    r = 2
    error = 1
    sol = 0
    flag = 0 
    while r <= 20 and error > 0.00001:
        flag = 0
        #print(r)
        #print(error)
        modelUpper, modelLower = \
            robustModelEllipticalUncertaintyUpperLower(model, Gamma, r, tol, 
                                                       numberOfRegressionPoints, 
                                                       coupled, 
                                                       dependentUncertainties, 
                                                       twoTerm, 
                                                       linearizeTwoTerm, False)
        try:
            solUpper = modelUpper.solve(verbosity = 0)
            sol = solUpper
        except:
            flag = 1
        try:
            solLower = modelLower.solve(verbosity = 0)
        except:
            print("infeasible")
            r=20
            sol = model.solve(verbosity = 0)
            break
        if flag != 1:
            try:
                error = \
                    (solUpper.get('cost').m - 
                     solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
            except:
                error = (solUpper.get('cost') - 
                         solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    initialGuess = sol.get("variables")
    if enableSP:
        modelUpper, modelLower = \
            robustModelEllipticalUncertaintyUpperLower(model, r-1, tol, 
                                                       numberOfRegressionPoints, 
                                                       coupled, 
                                                       dependentUncertainties, 
                                                       False, linearizeTwoTerm, 
                                                       True)
        subsVars = modelUpper.substitutions.keys()
        for i in xrange(len(subsVars)):
            del initialGuess[subsVars[i].key]
    return modelUpper, initialGuess, r
    
def robustModelRhombalUncertaintyUpperLower(model,r,tol, 
                                            numberOfRegressionPoints = 4, 
                                            coupled = True, 
                                            dependentUncertainties = True, 
                                            twoTerm = True, 
                                            linearizeTwoTerm = True, 
                                            enableSP = True):
    simplifiedModelUpper, simplifiedModelLower, numberOfNoDataConstraints = \
        EM.tractableModel(model, r, tol, coupled, dependentUncertainties, 
                          twoTerm, linearizeTwoTerm)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    uncertainVars = EM.uncertainModelVariables(model)
    posynomialsUpper = simplifiedModelUpper.as_posyslt1()
    posynomialsLower = simplifiedModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + \
                                            [posynomialsLower[i] <= 1]
        else:
            if len(p.exps) > 1:
                dataConstraints.append(EP.safePosynomialRhombalUncertainty
                                       (p, uncertainVars, i, enableSP))
            else:
                dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, 
                                                       uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = \
                                    normalizePerturbationVector(uncertainVars)
        coefficient = \
            constructRobustMonomailCoeffiecientsRhombalUncertainty
            (expsOfUncertainVars, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + \
                                    [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def robustModelRhombalUncertainty(model, tol=0.001, 
                                  numberOfRegressionPoints = 4, coupled = True, 
                                  dependentUncertainties = True, 
                                  twoTerm = True, linearizeTwoTerm = True, 
                                  enableSP = True):
    r = 2
    error = 1
    sol = 0
    flag = 0 
    while r <= 20 and error > 0.01:
        flag = 0
        print(r)
        print(error)
        modelUpper, modelLower = \
            robustModelRhombalUncertaintyUpperLower(model, r, tol, 
                                                    numberOfRegressionPoints, 
                                                    coupled, 
                                                    dependentUncertainties, 
                                                    twoTerm, linearizeTwoTerm, 
                                                    False)
        try:
            solUpper = modelUpper.solve(verbosity = 0)
            sol = solUpper
        except:
            flag = 1
        try:
            solLower = modelLower.solve(verbosity = 0)
        except:
            print("infeasible")
            r=20
            sol = model.solve(verbosity = 0)
            break
        if flag != 1:
            try:
                error = \
                    (solUpper.get('cost').m - 
                     solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
            except:
                error = (solUpper.get('cost') - 
                         solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    initialGuess = sol.get("variables")
    if enableSP:
        modelUpper, modelLower = \
            robustModelRhombalUncertaintyUpperLower(model, r-1, tol, 
                                                    numberOfRegressionPoints, 
                                                    coupled, 
                                                    dependentUncertainties, 
                                                    False, linearizeTwoTerm, 
                                                    True)
        subsVars = modelUpper.substitutions.keys()
        for i in xrange(len(subsVars)):
            del initialGuess[subsVars[i].key]
    return modelUpper, initialGuess, r

def boydRobustModelBoxUncertaintyUpperLower(model,r=3,tol=0.001):
    tracModelUpper, tracModelLower, numberOfNoDataConstraints = \
                                        EM.tractableBoydModel(model,r,tol)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    posynomialsUpper = tracModelUpper.as_posyslt1()
    posynomialsLower = tracModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + \
                                            [posynomialsLower[i] <= 1]
        else:
            dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, 
                                                       uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = \
                            normalizePerturbationVector(uncertainVars)
        coefficient = \
            constructRobustMonomailCoeffiecientsBoxUncertainty
            (expsOfUncertainVars, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + \
                                    [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def boydRobustModelBoxUncertainty(model, tol=0.001):
    r = 3
    error = 1
    while r <= 20 and error > 0.01:
        modelUpper, modelLower = boydRobustModelBoxUncertaintyUpperLower(model, 
                                                                         r, 
                                                                         tol)
        solUpper = modelUpper.solve(verbosity = 0)
        solLower = modelLower.solve(verbosity = 0)
        try:
            error = \
                (solUpper.get('cost').m - 
                 solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
        except:
            error = (solUpper.get('cost') - 
                     solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    return modelUpper,r
    
def boydRobustModelEllipticalUncertaintyUpperLower(model,r=3,tol=0.001):
    tracModelUpper, tracModelLower, numberOfNoDataConstraints = \
                                        EM.tractableBoydModel(model,r,tol)
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    dataConstraints = []
    dataMonomails = []
    posynomialsUpper = tracModelUpper.as_posyslt1()
    posynomialsLower = tracModelLower.as_posyslt1()
    for i,p in enumerate(posynomialsUpper):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + \
                                            [posynomialsLower[i] <= 1]
        else:
            dataMonomails.append(p)
    uncertainVars = EM.uncertainModelVariables(model)
    expsOfUncertainVars = uncertainVariablesExponents (dataMonomails, 
                                                       uncertainVars)
    if expsOfUncertainVars.size > 0:
        centeringVector, scalingVector = \
                                normalizePerturbationVector(uncertainVars)
        coefficient = \
            constructRobustMonomailCoeffiecientsEllipticalUncertainty
            (expsOfUncertainVars, centeringVector, scalingVector)
        for i in xrange(len(dataMonomails)):
            dataConstraints = dataConstraints + \
                                    [coefficient[i][0]*dataMonomails[i] <= 1]
    outputUpper = Model(model.cost, [noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions) 
    outputLower = Model(model.cost, [noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions) 
    return outputUpper, outputLower

def boydRobustModelEllipticalUncertainty(model, tol=0.001):
    r = 3
    error = 1
    while r <= 20 and error > 0.01:
        modelUpper, modelLower = \
                boydRobustModelEllipticalUncertaintyUpperLower(model,r,tol)
        solUpper = modelUpper.solve(verbosity = 0)
        solLower = modelLower.solve(verbosity = 0)
        try:
            error = \
                (solUpper.get('cost').m - 
                 solLower.get('cost').m)/(0.0 + solLower.get('cost').m)
        except:
            error = (solUpper.get('cost') - 
                     solLower.get('cost'))/(0.0 + solLower.get('cost'))
        r = r + 1
    return modelUpper,r   
    
def solveRobustSPBox(model,Gamma,relTol = 1e-5):
    initSol = model.localsolve(verbosity=0)
    try:
        initCost = initSol['cost'].m
    except:
        initCost = initSol['cost']
    newCost = initCost*(1 + 2*relTol)
    while (np.abs(initCost - newCost)/initCost) > relTol:
        apprModel = Model(model.cost,model.as_gpconstr(initSol))
        robModel = robustModelBoxUncertainty(apprModel,Gamma)[0]
        sol = robModel.solve(verbosity=0)
        initSol = sol.get('variables')
        initCost = newCost
        try:
            newCost = sol['cost'].m
        except:
            newCost = sol['cost']
        print(newCost)
    return initSol

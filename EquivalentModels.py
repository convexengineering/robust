from gpkit import  Model
import numpy as np
import TwoTermApproximation as TTA
import LinearizeTwoTermPosynomials as LTTP
import EquivalentPosynomial as EP


class EquivalentModel(Model):
    """
    A class that generates models that are equivalent to the original models and ready to be robustified.
    """

    def uncertain_model_variables(self):
        """
        Finds the uncertain variables of a model
        :return: the uncertain variables
        """
        subs_vars = list(self.substitutions.keys())
        uncertain_vars = [var for var in subs_vars if var.key.pr is not None]
        return uncertain_vars

    @staticmethod
    def uncertain_variables_exponents(data_monomails, uncertain_vars):
        """
        gets the exponents of uncertain variables
        :param data_monomails:  the uncertain posynomials
        :param uncertain_vars: the uncertain variables of the model
        :return: the 2 dimensional array of exponents(matrix)
        """
        exps_of_uncertain_vars = \
            np.array([[-p.exps[0].get(var.key, 0) for var in uncertain_vars]
                      for p in data_monomails])
        return exps_of_uncertain_vars

    @staticmethod
    def normalize_perturbation_vector(uncertain_vars):
        """
        normalizes the perturbation elements
        :param uncertain_vars: the uncertain variables of the model
        :return: the centering and scaling vector
        """
        prs = np.array([var.key.pr for var in uncertain_vars])
        eta_max = np.log(1 + prs/100.0)
        eta_min = np.log(1 - prs/100.0)
        centering_vector = (eta_min + eta_max)/2.0
        scaling_vector = eta_max - centering_vector
        return centering_vector, scaling_vector

    @staticmethod
    def construct_robust_monomial_coefficients(exps_of_uncertain_vars, gamma,
                                               type_of_uncertainty_set,
                                               centering_vector, scaling_vector):
        """
        robustify monomials
        :param exps_of_uncertain_vars: the matrix of exponents
        :param gamma: controls the size of the uncertainty set
        :param type_of_uncertainty_set: box, elliptical, or one norm
        :param centering_vector: centers the perturbations around zero
        :param scaling_vector: scales the perturbation to 1
        :return: the coefficient that will multiply the left hand side of a constraint
        """
        b_pert = (exps_of_uncertain_vars * scaling_vector[np.newaxis])
        coefficient = []
        for i in xrange(exps_of_uncertain_vars.shape[0]):
            norm = 0
            centering = 0
            for j in range(exps_of_uncertain_vars.shape[1]):
                if type_of_uncertainty_set == 'box':
                    norm += np.abs(b_pert[i][j])
                elif type_of_uncertainty_set == 'elliptical':
                    norm += b_pert[i][j]**2
                elif type_of_uncertainty_set == 'one norm':
                    norm = max(norm, np.abs(b_pert[i][j]))
                else:
                    raise Exception('This type of set is not supported')
                centering = centering + exps_of_uncertain_vars[i][j] * centering_vector[j]
            if type_of_uncertainty_set == 'elliptical':
                norm = np.sqrt(norm)
            coefficient.append([np.exp(gamma*norm)/np.exp(centering)])
        return coefficient

    def same_model(self):
        """
        replicates a gp model into a new model
        :return: the new model
        """
        constraints = []
        for i, p in enumerate(self.as_posyslt1()):
            constraints.append(p <= 1)
        output = Model(self.cost, constraints)
        output.substitutions.update(self.substitutions)
        return output

    def equivalent_model(self, dependent_uncertainties):
        """
        generates an equivalent model that might not be ready for robustification
        :param dependent_uncertainties: if the uncertainty set is dependent or not
        :return: the equivalent model
        """
        data_constraints, no_data_constraints = [], []
        uncertain_vars = EquivalentModel.uncertain_model_variables(self)

        for i, p in enumerate(self.as_posyslt1()):
            (no_data, data) = EquivalentModel.\
                equivalentPosynomial(p, uncertain_vars, i, dependent_uncertainties)
            data_constraints += data
            no_data_constraints += no_data
        number_of_no_data_constraints = len(no_data_constraints)
        output = Model(self.cost,[no_data_constraints, data_constraints])
        output.substitutions.update(self.substitutions)
        return output, number_of_no_data_constraints

    def two_term_model(self, input_model_simplified, dependent_uncertainties, simple, boyd):
        """
        generates a two term model that is ready to be linearized
        :param input_model_simplified: specifies whether the input model is already simplified or
        can be more simplified
        :param dependent_uncertainties: whether or not the set is dependent
        :param simple: choose to perform simple two term approximation
        :param boyd: choose to apply boyd's two term approximation
        :return: two term model and the number of no data constraints
        """
        equiModel, numberOfNoDataConstraints = equivalentModel(,dependentUncertainties,True)
        dataConstraints = []
        noDataConstraints = []
        uncertainVars = uncertainModelVariables(model)
        for i, p in enumerate(equiModel.as_posyslt1()):
            if i < numberOfNoDataConstraints:
                noDataConstraints =  noDataConstraints + [p <= 1]
            else:
                (noData, data) = TTA.twoTermExpApproximation(p,uncertainVars,i)#data = twoTermExpApproximationBoyd(p,i)#
                dataConstraints = dataConstraints + data
                noDataConstraints = noDataConstraints + noData
        numberOfNoDataConstraints = len(noDataConstraints)
        output = Model(equiModel.cost,[noDataConstraints, dataConstraints])
        output.substitutions.update(model.substitutions)
        return output, numberOfNoDataConstraints

def twoTermBoydModel(model):
    constraints = []
    for i, p in enumerate(model.as_posyslt1()):
        constraints.append(TTA.twoTermExpApproximationBoyd(p,i))
    output = Model(model.cost,constraints)
    output.substitutions.update(model.substitutions)
    return output

def tractableModel(model,r = 3,tol = 0.001, coupled = True, dependentUncertainties = False, twoTerm = True, linearizeTwoTerm = True):
    dataConstraints = []
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    if (dependentUncertainties == False and coupled == True and twoTerm) or twoTerm == True:
        safeModel, numberOfNoDataConstraints = twoTermModel(model,dependentUncertainties)
    else:
        safeModel, numberOfNoDataConstraints = equivalentModel(model,dependentUncertainties,coupled)
    for i, p in enumerate(safeModel.as_posyslt1()):
        if i < numberOfNoDataConstraints:
            noDataConstraintsUpper = noDataConstraintsUpper + [p <= 1]
            noDataConstraintsLower = noDataConstraintsLower + [p <= 1]            
        else:
            if len(p.exps) == 2 and linearizeTwoTerm:
                uncertainSubsVars = uncertainModelVariables(model)
                minVars = len(uncertainSubsVars)
                maxVars = 0
                pUncertainVars = []
                for i in xrange(len(p.exps)):
                    mUncertainVars = [var for var in p.exps[i].keys() \
                                      if var in uncertainSubsVars]
                    minVars = min(minVars,len(mUncertainVars))
                    maxVars = max(maxVars,len(mUncertainVars))
                    for var in mUncertainVars:
                        if var not in pUncertainVars:
                            pUncertainVars.append(var)
                noDataUpper, noDataLower, data = LTTP.linearizeTwoTermExp(p, i, r, tol)
                noDataConstraintsUpper = noDataConstraintsUpper + noDataUpper
                noDataConstraintsLower = noDataConstraintsLower + noDataLower
                dataConstraints = dataConstraints + data
            else:
                dataConstraints = dataConstraints + [p <= 1]
    numberOfNoDataConstraints = len(noDataConstraintsUpper)
    outputUpper = Model(safeModel.cost,[noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions)    
    outputLower = Model(safeModel.cost,[noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions)
    return outputUpper, outputLower, numberOfNoDataConstraints     

def tractableBoydModel(model,r=3,tol=0.001):
    dataConstraints = []
    noDataConstraintsUpper = []
    noDataConstraintsLower = []
    twoTerm = twoTermBoydModel(model)
    for i, p in enumerate(twoTerm.as_posyslt1()):
        if len(p.exps) == 2:
            noDataUpper, noDataLower, data = LTTP.linearizeTwoTermExp(p, i, r, tol)
            noDataConstraintsUpper = noDataConstraintsUpper + noDataUpper
            noDataConstraintsLower = noDataConstraintsLower + noDataLower
            dataConstraints = dataConstraints + data
        else:
            dataConstraints = dataConstraints + [p <= 1]
    numberOfNoDataConstraints = len(noDataConstraintsUpper)
    outputUpper = Model(twoTerm.cost,[noDataConstraintsUpper,dataConstraints])
    outputUpper.substitutions.update(model.substitutions)    
    outputLower = Model(twoTerm.cost,[noDataConstraintsLower,dataConstraints])
    outputLower.substitutions.update(model.substitutions)
    return outputUpper, outputLower, numberOfNoDataConstraints  

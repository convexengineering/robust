from gpkit import Model
import numpy as np


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
    def uncertain_variables_exponents(data_monomials, uncertain_vars):
        """
        gets the exponents of uncertain variables
        :param data_monomials:  the uncertain posynomials
        :param uncertain_vars: the uncertain variables of the model
        :return: the 2 dimensional array of exponents(matrix)
        """
        exps_of_uncertain_vars = \
            np.array([[-p.exps[0].get(var.key, 0) for var in uncertain_vars]
                      for p in data_monomials])
        return exps_of_uncertain_vars

    @staticmethod
    def normalize_perturbation_vector(uncertain_vars):
        """
        normalizes the perturbation elements
        :param uncertain_vars: the uncertain variables of the model
        :return: the centering and scaling vector
        """
        prs = np.array([var.key.pr for var in uncertain_vars])
        eta_max = np.log(1 + prs / 100.0)
        eta_min = np.log(1 - prs / 100.0)
        centering_vector = (eta_min + eta_max) / 2.0
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
                    norm += b_pert[i][j] ** 2
                elif type_of_uncertainty_set == 'one norm':
                    norm = max(norm, np.abs(b_pert[i][j]))
                else:
                    raise Exception('This type of set is not supported')
                centering = centering + exps_of_uncertain_vars[i][j] * centering_vector[j]
            if type_of_uncertainty_set == 'elliptical':
                norm = np.sqrt(norm)
            coefficient.append([np.exp(gamma * norm) / np.exp(centering)])
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

    def equivalent_model(self, simple_model, dependent_uncertainties):
        """
        generates an equivalent model that might not be ready for robustification
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param dependent_uncertainties: if the uncertainty set is dependent or not
        :return: the equivalent model
        """
        data_constraints, no_data_constraints = [], []
        uncertain_vars = EquivalentModel.uncertain_model_variables(self)

        for i, p in enumerate(self.as_posyslt1()):
            (no_data, data) = EquivalentModel. \
                equivalentPosynomial(p, uncertain_vars, i, simple_model, dependent_uncertainties)
            data_constraints += data
            no_data_constraints += no_data
        number_of_no_data_constraints = len(no_data_constraints)
        output = Model(self.cost, [no_data_constraints, data_constraints])
        output.substitutions.update(self.substitutions)
        return output, number_of_no_data_constraints

    def two_term_model(self, dependent_uncertainties, simple, boyd):
        """
        generates a two term model that is ready to be linearized
        :param dependent_uncertainties: whether or not the set is dependent
        :param simple: choose to perform simple two term approximation
        :param boyd: choose to apply boyd's two term approximation
        :return: two term model and the number of no data constraints
        """
        equivalent_model, number_of_no_data_constraints \
            = self.equivalent_model(False, dependent_uncertainties)

        data_constraints, no_data_constraints = [], []
        uncertain_vars = self.uncertain_model_variables()

        for i, p in enumerate(equivalent_model.as_posyslt1()):
            if i < number_of_no_data_constraints:
                no_data_constraints += [p <= 1]
            else:
                (no_data, data) = EquivalentModel. \
                    two_term_equivalent_posynomial(p, uncertain_vars, i, simple, boyd)
                data_constraints += data
                no_data_constraints += no_data
        number_of_no_data_constraints = len(no_data_constraints)
        output = Model(equivalent_model.cost, [no_data_constraints, data_constraints])
        output.substitutions.update(self.substitutions)
        return output, number_of_no_data_constraints

    def tractable_model(self, r, tol, uncertain_vars, simple_model, dependent_uncertainties, two_term,
                        linearize_two_term, boyd):
        """
        generates a tractable model that is ready for robustification, except maybe for large posynomials
        :param r: the number of piece-wise linear functions per two term constraints
        :param tol: Determines the accuracy of PWL
        :param uncertain_vars: the uncertain variables of the model
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param dependent_uncertainties: whether the uncertainty is is dependent or not
        :param two_term: Solve the problem using two term decoupling rather than linear
        approximation of exponential uncertainties.
        :param linearize_two_term: linearize two term functions rather than considering
        them large posynomials
        :param boyd: choose to apply boyd's two term approximation
        :return: the safe model, the relaxed model, and the number of data deprived constraints
        """
        data_constraints, no_data_constraints_upper, no_data_constraints_lower = [], [], []

        if boyd:
            safe_model, number_of_no_data_constraints \
                = self.two_term_model(False, False, True)
        elif two_term:
            safe_model, number_of_no_data_constraints \
                = self.two_term_model(dependent_uncertainties, False, False)
        else:
            safe_model, number_of_no_data_constraints \
                = self.equivalent_model(simple_model, dependent_uncertainties)

        for i, p in enumerate(safe_model.as_posyslt1()):
            if i < number_of_no_data_constraints:
                no_data_constraints_upper += [p <= 1]
                no_data_constraints_lower += [p <= 1]
            else:
                if len(p.exps) == 2 and linearize_two_term:
                    min_vars = len(uncertain_vars)
                    max_vars = 0
                    p_uncertain_vars = []
                    for _ in xrange(len(p.exps)):
                        m_uncertain_vars = [var for var in p.exps[i].keys()
                                            if var in uncertain_vars]
                        min_vars = min(min_vars, len(m_uncertain_vars))
                        max_vars = max(max_vars, len(m_uncertain_vars))
                        for var in m_uncertain_vars:
                            if var not in p_uncertain_vars:
                                p_uncertain_vars.append(var)

                    no_data_upper, no_data_lower, data = EquivalentModel.linearizeTwoTermExp(p, i, r, tol)
                    no_data_constraints_upper += no_data_upper
                    no_data_constraints_lower += no_data_lower
                    data_constraints += data
                else:
                    data_constraints += [p <= 1]

        number_of_no_data_constraints = len(no_data_constraints_upper)
        output_upper = Model(safe_model.cost, [no_data_constraints_upper, data_constraints])
        output_upper.substitutions.update(self.substitutions)
        output_lower = Model(safe_model.cost, [no_data_constraints_lower, data_constraints])
        output_lower.substitutions.update(self.substitutions)

        return output_upper, output_lower, number_of_no_data_constraints

    @staticmethod
    def two_term_equivalent_posynomial(p, uncertain_vars, i, simple, boyd):
        """
        returns a two term posynomial equivalent to the original large posynomial
        :param p: the large posynomial
        :param uncertain_vars: the uncertain variables of the model containing the posynomial
        :param i: the index of the posynomial
        :param simple: whether or not a simple two term approximation is preferred
        :param boyd: whether or not a boyd two term approximation is preferred
        :return: the no data constaints and the data constraints
        """
        return [], []

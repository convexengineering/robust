from gpkit import Model
from EquivalentPosynomials import EquivalentPosynomials
from TwoTermApproximation import TwoTermApproximation
from LinearizeTwoTermPosynomials import LinearizeTwoTermPosynomials


class SameModel(Model):
    """
    copies a model without the substitutions
    """

    def setup(self, model):
        """
        replicates a gp model into a new model
        :param model: the original model
        :return: the new model
        """
        constraints = []
        for i, p in enumerate(model.as_posyslt1()):
            constraints.append(p <= 1)
        # output = Model(model.cost, constraints)
        self.cost = model.cost
        # self.substitutions.update(model.substitutions)
        return constraints

    @staticmethod
    def uncertain_model_variables(model):
        """
        Finds the uncertain variables of a model
        :return: the uncertain variables
        """
        subs_vars = list(model.substitutions.keys())
        uncertain_vars = [var for var in subs_vars if var.key.pr is not None]
        return uncertain_vars


class EquivalentModel(Model):
    """
    A class that generates models that are equivalent to the original models and ready to be
    robustified.
    """
    number_of_no_data_constraints = None

    def setup(self, model, simple_model, dependent_uncertainties):
        """
        generates an equivalent model that might not be ready for robustification
        :param model: the original model
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param dependent_uncertainties: if the uncertainty set is dependent or not
        :return: the equivalent model
        """
        data_constraints, no_data_constraints = [], []
        uncertain_vars = SameModel.uncertain_model_variables(model)

        for i, p in enumerate(model.as_posyslt1()):

            equivalent_p = EquivalentPosynomials(p)
            (no_data, data) = equivalent_p. \
                equivalent_posynomial(uncertain_vars, i, simple_model, dependent_uncertainties)

            data_constraints += data
            no_data_constraints += no_data

        self.number_of_no_data_constraints = len(no_data_constraints)
        self.cost = model.cost
        return [no_data_constraints, data_constraints]

    def get_number_of_no_data_constraints(self):
        return self.number_of_no_data_constraints


class TwoTermModel(Model):
    number_of_no_data_constraints = None

    def setup(self, model, dependent_uncertainties, simple, boyd, maximum_number_of_permutations):
        """
        generates a two term model that is ready to be linearized
        :param model: the original model
        :param dependent_uncertainties: whether or not the set is dependent
        :param simple: choose to perform simple two term approximation
        :param boyd: choose to apply boyd's two term approximation
        :param maximum_number_of_permutations: the maximum allowed number of permutations for two term approximation
        :return: two term model and the number of no data constraints
        """
        equivalent_model = EquivalentModel(model, False, dependent_uncertainties)
        number_of_no_data_constraints = equivalent_model.get_number_of_no_data_constraints()

        data_constraints, no_data_constraints = [], []

        uncertain_vars = SameModel.uncertain_model_variables(model)

        for i, p in enumerate(equivalent_model.as_posyslt1()):

            if i < number_of_no_data_constraints:
                no_data_constraints += [p <= 1]
            else:
                two_term_p = TwoTermApproximation(p)
                (no_data, data) = two_term_p. \
                    two_term_equivalent_posynomial(uncertain_vars, i, simple, boyd, maximum_number_of_permutations)

                data_constraints += data[0]
                no_data_constraints += no_data[0]

        self.number_of_no_data_constraints = len(no_data_constraints)
        self.cost = model.cost

        return [no_data_constraints, data_constraints]

    def get_number_of_no_data_constraints(self):
        return self.number_of_no_data_constraints


class TractableModel:

    upper_model = None
    lower_model = None

    number_of_no_data_constraints = None

    def __init__(self, model, r, tol, uncertain_vars, simple_model, dependent_uncertainties, two_term,
                 simple_two_term, maximum_number_of_permutations, linearize_two_term, boyd):
        """
        generates a tractable model that is ready for robustification, except maybe for large posynomials
        :param model: the original model
        :param r: the number of piece-wise linear functions per two term constraints
        :param tol: Determines the accuracy of PWL
        :param uncertain_vars: the uncertain variables of the model
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param dependent_uncertainties: whether the uncertainty is is dependent or not
        :param two_term: Solve the problem using two term decoupling rather than linear
        approximation of exponential uncertainties.
        :param simple_two_term: if a simple two term approximation is preferred
        :param maximum_number_of_permutations: the maximum allowed number of permutations for two term approximation
        :param linearize_two_term: linearize two term functions rather than considering
        them large posynomials
        :param boyd: choose to apply boyd's two term approximation
        :return: the safe model, the relaxed model, and the number of data deprived constraints
        """
        data_constraints, no_data_constraints_upper, no_data_constraints_lower = [], [], []

        if boyd:
            safe_model = TwoTermModel(model, False, False, True, maximum_number_of_permutations)

        elif two_term:
            safe_model = TwoTermModel(model, dependent_uncertainties, simple_two_term, False,
                                      maximum_number_of_permutations)
        else:
            safe_model = EquivalentModel(model, simple_model, dependent_uncertainties)

        number_of_no_data_constraints = safe_model.get_number_of_no_data_constraints()

        for i, p in enumerate(safe_model.as_posyslt1()):

            if i < number_of_no_data_constraints:
                no_data_constraints_upper += [p <= 1]
                no_data_constraints_lower += [p <= 1]

            else:

                if len(p.exps) == 2 and linearize_two_term:
                    min_vars = len(uncertain_vars)
                    max_vars = 0
                    p_uncertain_vars = []

                    for j in xrange(len(p.exps)):

                        m_uncertain_vars = [var for var in p.exps[j].keys()
                                            if var in uncertain_vars]

                        min_vars = min(min_vars, len(m_uncertain_vars))
                        max_vars = max(max_vars, len(m_uncertain_vars))

                        for var in m_uncertain_vars:
                            if var not in p_uncertain_vars:
                                p_uncertain_vars.append(var)
                    p_linearize = LinearizeTwoTermPosynomials(p)
                    no_data_upper, no_data_lower, data = p_linearize.linearize_two_term_exp(i, r, tol)

                    no_data_constraints_upper += no_data_upper
                    no_data_constraints_lower += no_data_lower

                    data_constraints += data

                else:
                    data_constraints += [p <= 1]

        self.number_of_no_data_constraints = len(no_data_constraints_upper)

        self.upper_model = Model(safe_model.cost, [no_data_constraints_upper, data_constraints])
        self.upper_model.substitutions.update(model.substitutions)
        self.lower_model = Model(safe_model.cost, [no_data_constraints_lower, data_constraints])
        self.lower_model.substitutions.update(model.substitutions)

    def get_number_of_no_data_constraints(self):
        return self.number_of_no_data_constraints

    def get_tractable_models(self):
        return self.upper_model, self.lower_model

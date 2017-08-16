from gpkit import Model
from EquivalentPosynomials import EquivalentPosynomials
from TwoTermApproximation import TwoTermApproximation


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
        self.cost = model.cost
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

    def setup(self, model, uncertain_vars, simple_model, dependent_uncertainties):
        """
        generates an equivalent model that might not be ready for robustification
        :param uncertain_vars: the uncertain variables of the model
        :param model: the original model
        :param simple_model: whether or not a simple conservative robust model is preferred
        :param dependent_uncertainties: if the uncertainty set is dependent or not
        :return: the equivalent model
        """
        data_constraints, no_data_constraints = [], []

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


class TwoTermBoydModel(Model):

    def setup(self, model):
        """
        generates a two term Boyd model that is ready to be linearized
        :param model: the original model
        :return: two term model and the number of no data constraints
        """
        data_constraints = []

        for i, p in enumerate(model.as_posyslt1()):
            _, data = TwoTermApproximation. \
                two_term_equivalent_posynomial(p, i, [], True)

            data_constraints += data

        self.cost = model.cost

        return data_constraints

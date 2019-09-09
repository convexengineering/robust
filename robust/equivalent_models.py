from __future__ import absolute_import
from gpkit import Model
from gpkit.nomials import MonomialEquality, PosynomialInequality
from gpkit.exceptions import InvalidGPConstraint

from .equivalent_posynomials import EquivalentPosynomials
from .twoterm_approximation import TwoTermApproximation


class EquivalentModel(Model):
    """
    A class that generates models that are equivalent to the original models and ready to be
    robustified.
    """
    number_of_no_data_constraints = None

    def setup(self, model, dependent_uncertainties, setting):
        """
        generates an equivalent model that might not be ready for robustification
        :param model: the original model
        :param setting: robustness setting
        :param dependent_uncertainties: if the uncertainty set is dependent or not
        :return: the equivalent model
        """
        data_constraints, no_data_constraints = [], []

        for i, p in enumerate(model.as_posyslt1()):

            equivalent_p = EquivalentPosynomials(p, i,
                                                 setting.get('simpleModel'), dependent_uncertainties)
            no_data, data = equivalent_p.no_data_constraints, equivalent_p.data_constraints

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
        all_constraints = model.flat(constraintsets=False)

        data_constraints = []

        for i, cs in enumerate(all_constraints):
            if isinstance(cs, MonomialEquality):
                data_constraints += [cs]
            elif isinstance(cs, PosynomialInequality):
                _, data = TwoTermApproximation. \
                    equivalent_posynomial(cs.as_posyslt1()[0], i, [], True)
                data_constraints += data
                del data
            else:
                raise InvalidGPConstraint("Two Term Boyd Model supports geometric programs only")
        self.cost = model.cost

        return data_constraints

if __name__ == '__main__':
    pass

from __future__ import absolute_import
from __future__ import division
from builtins import range
from builtins import object
import numpy as np
from gpkit import Variable, Monomial, Posynomial
import random
import math
from copy import copy

from .robust_gp_tools import RobustGPTools


class TwoTermApproximation(object):
    """
    replaces a large posynomial by a data-deprived large posynomial and a set of two term posynomials
    """

    p = Posynomial()
    number_of_monomials = None
    list_of_permutations = []

    def __init__(self, p, setting):
        self.p = p
        self.number_of_monomials = len(self.p.exps)

        self.list_of_permutations = []

        if not setting.get('boyd'):
            if setting.get('smartTwoTermChoose'):
                bad_relations, sizes = self.bad_relations(self.p)
                list_of_couples, new_list_to_permute = TwoTermApproximation. \
                    choose_convenient_couples(bad_relations, sizes, self.number_of_monomials)
            else:
                list_of_couples = []
                new_list_to_permute = list(range(0, self.number_of_monomials))

            first_elements = []
            for couple in list_of_couples:
                first_elements += couple

            length_of_permutation = len(new_list_to_permute)
            max_num_of_perms = \
                TwoTermApproximation.total_number_of_permutations(length_of_permutation)

            counter = 0
            number_of_permutations = min(setting.get('allowedNumOfPerms'), max_num_of_perms)
            while counter < number_of_permutations:
                temp = copy(new_list_to_permute)
                random.shuffle(temp)
                if TwoTermApproximation.check_if_permutation_exists(self.list_of_permutations, first_elements + temp):
                    continue
                else:
                    self.list_of_permutations.append(first_elements + temp)
                    counter += 1

    @staticmethod
    def two_term_equivalent_posynomial(p, m, permutation, boyd):
        """
        returns a two term posynomial equivalent to the original large posynomial
        :param p: the posynomial
        :param m: the index of the posynomial
        :param boyd: whether or not a boyd two term approximation is preferred
        :param permutation: the permutation to be used for two term approximation
        :return: the no data constraints and the data constraints
        """
        number_of_monomials = len(p.exps)
        if number_of_monomials <= 2:
            return [[]], [[p <= 1]]

        data_constraints, no_data_constraints = [], []

        if boyd:
            z_1 = Variable("z^1_(%s)" % m)
            data_constraints += [Monomial(p.exps[0], p.cs[0]) + z_1 <= 1]
            for i in range(number_of_monomials - 3):
                if i > 0:
                    z_1 = Variable("z^%s_(%s)" % (i + 1, m))
                z_2 = Variable("z^%s_(%s)" % (i + 2, m))
                data_constraints += [Monomial(p.exps[i + 1], p.cs[i + 1])/z_1 + z_2 / z_1 <= 1]
            z_2 = Variable("z^%s_(%s)" % (number_of_monomials - 2, m))
            data_constraints += [
                Monomial(p.exps[number_of_monomials - 2], p.cs[number_of_monomials - 2]) / z_2 +
                Monomial(p.exps[number_of_monomials - 1], p.cs[number_of_monomials - 1]) / z_2 <= 1]
            return [], data_constraints

        length_of_permutation = len(permutation)
        number_of_iterations = int(np.floor(length_of_permutation / 2.0))

        zs = []

        for j in range(number_of_iterations):
            z = Variable("z^%s_%s" % (j, m))
            zs.append(z)
            data_constraints += [Monomial(p.exps[permutation[2 * j]], p.cs[permutation[2 * j]]) +
                                 Monomial(p.exps[permutation[2 * j + 1]], p.cs[permutation[2 * j + 1]]) <= z]

        if length_of_permutation % 2 == 1:
            z = Variable("z^%s_%s" % (number_of_iterations, m))
            zs.append(z)
            data_constraints += [Monomial(p.exps[permutation[length_of_permutation - 1]],
                                          p.cs[permutation[length_of_permutation - 1]]) <= z]

        no_data_constraints.append([sum(zs) <= 1])

        return no_data_constraints, data_constraints

    @staticmethod
    def check_if_permutation_exists(permutations, permutation):
        """
        Checks if a permutation already exists in a list of permutations
        :param permutations: the list of permutations
        :param permutation: the permutation to be checked
        :return: True or false
        """
        if permutation in permutations:
            return True
        if len(permutation) == 1:
            return False
        true_or_false = [1] * len(permutations)
        for i in range(int(np.floor(len(permutation)/2))):
            for j in range(len(true_or_false)):
                if true_or_false[j] == 1:
                    ind_one = permutations[j].index(permutation[2 * i])
                    ind_two = permutations[j].index(permutation[2 * i + 1])
                    if np.floor(ind_one / 2) != np.floor(ind_two / 2):
                        true_or_false[j] = 0
        if 1 in true_or_false:
            return True
        else:
            return False

    @staticmethod
    def n_choose_r(n, r):
        """
        Combination formula
        :param n: the number of possibilities
        :param r: the numbers to choose from
        :return: the number of possible combinations
        """
        f = math.factorial
        return f(n) / f(r) / f(n - r)

    @staticmethod
    def total_number_of_permutations(length_of_permutation):
        """
        Finds the total number of possible "different" permutations
        :param length_of_permutation: the number of elements
        :return: the total number of permutations
        """
        if length_of_permutation % 2 == 1:
            length_of_permutation += 1

        n = length_of_permutation
        prod = 1
        while n >= 4:
            prod *= TwoTermApproximation.n_choose_r(n, 2)
            n -= 2
        return prod / math.factorial(length_of_permutation / 2)

    @staticmethod
    def bad_relations(p):
        """
        Investigates the relations between the monomials in a posynomial
        :param p: the posynomial
        :return: the dictionary of relations, and some other assisting dictionary
        """
        number_of_monomials = len(p.exps)
        inverse_relations = {}
        sizes = {}
        for i in range(number_of_monomials):
            direct_vars_only_monomial_ith_exps = RobustGPTools.\
                only_uncertain_vars_monomial(p.exps[i])
            ith_monomial_exps = direct_vars_only_monomial_ith_exps
            m_uncertain_vars = [var for var in list(ith_monomial_exps.keys())
                                if RobustGPTools.is_directly_uncertain(var)]
            for j in range(0, number_of_monomials):
                direct_vars_only_monomial_jth_exps = RobustGPTools.\
                    only_uncertain_vars_monomial(p.exps[j])
                jth_monomial_exps = direct_vars_only_monomial_jth_exps
                for var in m_uncertain_vars:
                    if ith_monomial_exps.get(var.key, 0) * jth_monomial_exps.get(var.key, 0) < 0:
                        if i in inverse_relations:
                            if j in inverse_relations[i]:
                                inverse_relations[i][j] += 1
                            else:
                                inverse_relations[i][j] = 1
                                sizes[i] += 1
                        else:
                            inverse_relations[i] = {j: 1}
                            sizes[i] = 1
        return inverse_relations, sizes

    @staticmethod
    def choose_convenient_couples(relations, sizes, number_of_monomials):
        """
        Chooses which couples goes together for a two term approximation of a posynomial
        :param relations: the dictionary of relations
        :param sizes: some assisting dictionary
        :param number_of_monomials: the total number of monomials
        :return:the list of couples and the remaining monomials that need to be dealt with
        """
        list_of_couples = []
        to_permute = list(range(0, number_of_monomials))
        while len(relations) > 0:
            vals_sizes = list(sizes.values())
            keys_sizes = list(sizes.keys())
            minimum_value_key = keys_sizes[vals_sizes.index(min(vals_sizes))]
            couple = [minimum_value_key]

            del to_permute[to_permute.index(minimum_value_key)]

            vals_relations_of_min = list(relations[minimum_value_key].values())
            keys_relations_of_min = list(relations[minimum_value_key].keys())
            maximum_of_min_value_key = keys_relations_of_min[vals_relations_of_min.index(max(vals_relations_of_min))]
            couple.append(maximum_of_min_value_key)

            del to_permute[to_permute.index(maximum_of_min_value_key)]

            del relations[minimum_value_key]
            del relations[maximum_of_min_value_key]
            del sizes[minimum_value_key]
            del sizes[maximum_of_min_value_key]

            for key in list(relations.keys()):
                if minimum_value_key in relations[key]:
                    del relations[key][minimum_value_key]
                    sizes[key] -= 1

                if maximum_of_min_value_key in relations[key]:
                    del relations[key][maximum_of_min_value_key]
                    sizes[key] -= 1

                if sizes[key] == 0:
                    del sizes[key]
                    del relations[key]

            list_of_couples.append(couple)

        return list_of_couples, to_permute

    def __repr__(self):
        return "TwoTermApproximation(" + self.p.__repr__() + ")"

    def __str__(self):
        return "TwoTermApproximation(" + self.p.__str__() + ")"

if __name__ == '__main__':
    pass

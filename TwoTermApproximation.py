import numpy as np
from gpkit import Variable, Monomial, Posynomial
import random
import math
from copy import copy


class TwoTermApproximation:
    """
    replaces a large posynomial by a data-deprived large posynomial and a set of two term posynomials
    """

    def __init__(self, p):
        self.p = p

    p = Posynomial()
    list_of_possibilities = []

    def two_term_equivalent_posynomial(self, uncertain_vars, m, simple,
                                       boyd, maximum_number_of_permutations):
        """
        returns a two term posynomial equivalent to the original large posynomial
        :param uncertain_vars: the uncertain variables of the model containing the posynomial
        :param m: the index of the posynomial
        :param simple: whether or not a simple two term approximation is preferred
        :param boyd: whether or not a boyd two term approximation is preferred
        :param maximum_number_of_permutations: the maximum number of allowed two term approximations
        per posynomial
        :return: the no data constraints and the data constraints
        """
        number_of_monomials = len(self.p.exps)

        if number_of_monomials <= 2:
            return [[]], [[self.p <= 1]]

        data_constraints, no_data_constraints = [], []

        if boyd:
            z_1 = Variable("z^1_(%s)" % m)
            data_constraints += [Monomial(self.p.exps[0], self.p.cs[0]) + z_1 <= 1]
            for i in xrange(number_of_monomials - 3):
                if i > 0:
                    z_1 = Variable("z^%s_(%s)" % (i + 1, m))
                z_2 = Variable("z^%s_(%s)" % (i + 2, m))
                data_constraints += [Monomial(self.p.exps[i + 1], self.p.cs[i + 1]) + z_2 / z_1 <= 1]
            z_2 = Variable("z^%s_(%s)" % (number_of_monomials - 2, m))
            data_constraints += [
                Monomial(self.p.exps[number_of_monomials - 2], self.p.cs[number_of_monomials - 2]) / z_2 +
                Monomial(self.p.exps[number_of_monomials - 1], self.p.cs[number_of_monomials - 1]) / z_2 <= 1]
            return [[]], [data_constraints]

        if simple:
            maximum_number_of_permutations = 1

        zs = []

        bad_relations, sizes = self.bad_relations(uncertain_vars)
        list_of_couples, new_list_to_permute = TwoTermApproximation. \
            choose_convenient_couples(bad_relations, sizes, number_of_monomials)

        number_of_couples = len(list_of_couples)

        for i, couple in enumerate(list_of_couples):
            z = Variable("z^%s_(%s)" % (i, m))
            zs.append(z)
            data_constraints += [Monomial(self.p.exps[couple[0]], self.p.cs[couple[0]]) +
                                 Monomial(self.p.exps[couple[1]], self.p.cs[couple[1]]) <= z]

        length_of_permutation = len(new_list_to_permute)

        total_number_of_possible_permutations = \
            TwoTermApproximation.total_number_of_permutations(length_of_permutation)

        permutations = []

        counter = 0

        number_of_permutations = min(maximum_number_of_permutations, total_number_of_possible_permutations)

        data_constraints = [data_constraints] * number_of_permutations

        while counter < number_of_permutations:
            temp = copy(new_list_to_permute)
            random.shuffle(temp)

            if TwoTermApproximation.check_if_permutation_exists(permutations, temp):
                continue
            else:
                permutations.append(temp)
                counter += 1

        for i, permutation in enumerate(permutations):
            perm_zs = []
            number_of_iterations = int(np.floor(length_of_permutation / 2.0))

            for j in xrange(number_of_iterations):
                z = Variable("z^%s_(%s,%s)" % (j + number_of_couples, m, i))
                perm_zs.append(z)
                data_constraints[i] += [Monomial(self.p.exps[permutation[2 * j]], self.p.cs[permutation[2 * j]]) +
                                        Monomial(self.p.exps[permutation[2 * j + 1]],
                                                 self.p.cs[permutation[2 * j + 1]]) <= z]

            if length_of_permutation % 2 == 1:
                z = Variable("z^%s_(%s,%s)" % (number_of_iterations + number_of_couples, m, i))
                perm_zs.append(z)
                data_constraints[i] += [Monomial(self.p.exps[permutation[length_of_permutation - 1]],
                                                 self.p.cs[permutation[length_of_permutation - 1]]) <= z]

            no_data_constraints.append([sum(zs) + sum(perm_zs) <= 1])

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
            return False

        true_or_false = [1] * len(permutations)
        for i in xrange(int(len(permutation) / 2)):
            for j in xrange(len(true_or_false)):
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

    def bad_relations(self, uncertain_vars):
        """
        Investigates the relations between the monomials in a posynomial
        :param uncertain_vars: the model's uncertain variables
        :return: the dictionary of relations, and some other assisting dictionary
        """
        number_of_monomials = len(self.p.exps)
        inverse_relations = {}
        sizes = {}
        for i in xrange(number_of_monomials):
            ith_monomial_exps = self.p.exps[i]
            m_uncertain_vars = [var for var in ith_monomial_exps.keys() if var in uncertain_vars]
            for j in range(0, number_of_monomials):
                jth_monomial_exps = self.p.exps[j]
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
        to_permute = range(0, number_of_monomials)
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

            for key in relations.keys():
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

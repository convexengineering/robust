import numpy as np
from gpkit import Variable, Monomial
from copy import copy
from TwoTermApproximation import TwoTermApproximation


def test_check_if_permutation_exists():
    for _ in xrange(10):
        number_of_monomials = int(np.random.rand()*15) + 3
        number_of_permutations = TwoTermApproximation.total_number_of_permutations(number_of_monomials)

        number_of_gp_variables = int(np.random.rand()*20) + 1

        m = [np.random.rand()*10 for _ in xrange(number_of_monomials)]

        for j in xrange(number_of_monomials):
            for i in xrange(number_of_gp_variables):
                x = Variable('x_%s' % i)
                m[j] *= x**(np.random.rand()*10 - 5)

        p = sum(m)

        permutation_list = range(0, number_of_monomials)
        list_of_permutations = []
        list_of_posynomials = []

        counter = 0

        while counter < min(100, int(number_of_permutations/2)):
            temp = copy(permutation_list)
            np.random.shuffle(temp)

            if TwoTermApproximation.check_if_permutation_exists(list_of_permutations, temp):
                continue
            else:
                list_of_permutations.append(temp)
                _, data_constraints = TwoTermApproximation.two_term_equivalent_posynomial(p, 1, temp, False)
                data_posynomial = [constraint.as_posyslt1()[0]*Variable("z^%s_%s" % (i, 1))
                                   for i, constraint in enumerate(data_constraints)]
                list_of_posynomials.append(set(data_posynomial))
                counter += 1

        counter = 0

        while counter < min(100, int(number_of_permutations/2)):
            temp = copy(permutation_list)
            np.random.shuffle(temp)

            flag_one = TwoTermApproximation.check_if_permutation_exists(list_of_permutations, temp)
            _, data_constraints = TwoTermApproximation.two_term_equivalent_posynomial(p, 1, temp, False)
            data_posynomial = [constraint.as_posyslt1()[0]*Variable("z^%s_%s" % (i, 1))
                               for i, constraint in enumerate(data_constraints)]
            flag_two = set(data_posynomial) in list_of_posynomials

            assert (flag_one == flag_two)

            counter += 1
    return


def test_bad_relations():
    for _ in xrange(100):
        number_of_monomials = int(2*np.random.random()) + 3
        number_of_gp_variables = int(np.random.rand()*1) + 1
        number_of_uncertain_variables = int(np.random.rand()*5) + 1
        vector_to_choose_from = [0, 0, 0, 0, 0, 0, 0, 0, 1, -1]

        m = [np.random.rand()*10 for _ in xrange(number_of_monomials)]
        p_uncertain_vars = []
        relations = {}
        sizes = {}
        neg_pos_neutral_powers = []
        # correlations = []
        # dependent_theoretical_partition = []

        for j in xrange(number_of_monomials):
            for i in xrange(number_of_gp_variables):
                x = Variable('x_%s' % i)
                m[j] *= x**(np.random.rand()*10 - 5)

        for i in xrange(number_of_uncertain_variables):
            u = Variable('u_%s' % i, np.random.random(), pr=100*np.random.random())
            p_uncertain_vars.append(u.key)
            neg_pos_neutral_powers.append([vector_to_choose_from[int(10*np.random.random())]
                                           for _ in xrange(number_of_monomials)])
            for j in xrange(number_of_monomials):
                m[j] *= u**(np.random.rand()*5*neg_pos_neutral_powers[i][j])
        for i in xrange(number_of_uncertain_variables):
            for mon_counter_one in xrange(len(neg_pos_neutral_powers[i])):
                for mon_counter_two in xrange(len(neg_pos_neutral_powers[i])):
                    if neg_pos_neutral_powers[i][mon_counter_one]*neg_pos_neutral_powers[i][mon_counter_two] < 0:
                        # print (relations)
                        if m[mon_counter_one] in relations:
                            if m[mon_counter_two] in relations[m[mon_counter_one]]:
                                relations[m[mon_counter_one]][m[mon_counter_two]] += 1
                            else:
                                relations[m[mon_counter_one]][m[mon_counter_two]] = 1
                                sizes[m[mon_counter_one]] += 1
                        else:
                            relations[m[mon_counter_one]] = {m[mon_counter_two]: 1}
                            sizes[m[mon_counter_one]] = 1



        p = sum(m)

        actual_relations, actual_sizes = TwoTermApproximation.bad_relations(p, p_uncertain_vars)

        keys = actual_relations.keys()

        actual_relations_mons = {}
        actual_sizes_mons = {}

        for key in keys:
            internal_map = actual_relations[key]
            map_keys = internal_map.keys()
            map_mons = {}
            for map_key in map_keys:
                map_mons[Monomial(p.exps[map_key],p.cs[map_key])] = internal_map[map_key]
            actual_relations_mons[Monomial(p.exps[key],p.cs[key])] = map_mons

            actual_sizes_mons[Monomial(p.exps[key],p.cs[key])] = actual_sizes[key]

        assert (actual_relations_mons == relations)
        assert (sizes == actual_sizes_mons)

    return



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
        number_of_monomials = int(20*np.random.random()) + 3
        number_of_gp_variables = int(np.random.rand()*10) + 1
        number_of_additional_uncertain_variables = int(np.random.rand()*5) + 1
        vector_to_choose_from_pos_only = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]

        m = [np.random.rand()*10 for _ in xrange(number_of_monomials)]
        p_uncertain_vars = []
        relations = {}
        sizes = {}
        neg_pos_neutral_powers = []

        for j in xrange(number_of_monomials):
            for i in xrange(number_of_gp_variables):
                x = Variable('x_%s' % i)
                m[j] *= x**(np.random.rand()*10 - 5)

        number_of_elements_in_relation = min(number_of_monomials, int(number_of_monomials*np.random.rand()+2))
        all_elements = []

        for dummy_one in xrange(number_of_elements_in_relation):
            element = np.random.choice(range(0, number_of_monomials))
            while element in all_elements:
                element = np.random.choice(range(0, number_of_monomials))
            all_elements.append(element)

            element_map = {}
            number_of_element_map_elements = min(number_of_monomials - 1, int(number_of_monomials*np.random.rand()+2))
            for dummy_two in xrange(number_of_element_map_elements):
                element_map_element = int(np.random.rand()*number_of_monomials)
                while element_map_element in element_map or element_map_element == element:
                    element_map_element = int(np.random.rand()*number_of_monomials)

                size = int(number_of_monomials*np.random.rand())+1
                element_map[element_map_element] = size
                try:
                    relations[element_map_element][element] = size
                except:
                    relations[element_map_element] = {element: size}

            if element in relations:
                relations[element].update(element_map)
            else:
                relations[element] = element_map

        relations_copy = {}
        for key in relations.keys():
            relations_copy[key] = copy(relations[key])
            sizes[key] = len(relations[key])

        counter = 0
        while relations_copy:
            keys = relations_copy.keys()
            for key in keys:
                if not relations_copy[key]:
                    del relations_copy[key]
                    continue

                u = Variable('u_%s' % counter, np.random.random(), pr=100*np.random.random())
                counter += 1
                p_uncertain_vars.append(u.key)

                el_pow = np.random.choice([-1, 1])
                m[key] *= u**(np.random.rand()*5*el_pow)

                element_keys = relations_copy[key].keys()
                for element_key in element_keys:
                    m[element_key] *= u**(-np.random.rand()*5*el_pow)

                    relations_copy[key][element_key] -= 1
                    if relations_copy[key][element_key] == 0:
                        del relations_copy[key][element_key]

                    relations_copy[element_key][key] -= 1
                    if relations_copy[element_key][key] == 0:
                        del relations_copy[element_key][key]

        for i in xrange(number_of_additional_uncertain_variables):
            u = Variable('u_%s' % counter, np.random.random(), pr=100*np.random.random())
            counter += 1
            p_uncertain_vars.append(u.key)
            el_pow = np.random.choice([-1, 1])
            neg_pos_neutral_powers.append([el_pow*vector_to_choose_from_pos_only[int(10*np.random.random())]
                                           for _ in xrange(number_of_monomials)])
            for j in xrange(number_of_monomials):
                m[j] *= u**(np.random.rand()*5*neg_pos_neutral_powers[i][j])

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
                map_mons[Monomial(p.exps[map_key], p.cs[map_key])] = internal_map[map_key]
            actual_relations_mons[Monomial(p.exps[key], p.cs[key])] = map_mons
            actual_sizes_mons[Monomial(p.exps[key], p.cs[key])] = actual_sizes[key]

        keys = relations.keys()
        relations_mons = {}
        sizes_mons = {}
        for key in keys:
            internal_map = relations[key]
            map_keys = internal_map.keys()
            map_mons = {}
            for map_key in map_keys:
                map_mons[m[map_key]] = internal_map[map_key]
            relations_mons[m[key]] = map_mons
            sizes_mons[m[key]] = sizes[key]

        assert (actual_relations_mons == relations_mons)
        assert (sizes_mons == actual_sizes_mons)

    return

from gpkit import Variable, Monomial
import numpy as np
from EquivalentPosynomials import EquivalentPosynomials


def test_merge_intersected_lists():

    for _ in xrange(10000):
        number_of_lists = int(10*np.random.random()) + 1
        list_of_lists = []
        for l in xrange(number_of_lists):
            number_of_elements = int(5*np.random.random()) + 1
            list_of_lists.append([int(20*np.random.random()) for _ in xrange(number_of_elements)])
        partition = EquivalentPosynomials.merge_intersected_lists(list_of_lists)
        for i in xrange(len(partition) - 1):
            assert (all(not (set(partition[i]) & set(partition[j])) for j in range(i+1, len(partition))))
        for from_list in list_of_lists:
            assert(any(set(from_list) <= set(from_partition) for from_partition in partition))
    return


def test_same_sign():
    for _ in xrange(100):
        number_of_elements = int(50*np.random.random()) + 1
        a = [int(20*np.random.random()) for _ in xrange(number_of_elements)]
        assert(EquivalentPosynomials.same_sign(a))
        a = [int(20*np.random.random()) - 19 for _ in xrange(number_of_elements)]
        assert(EquivalentPosynomials.same_sign(a))
        a = [int(20*np.random.random()) - 19 for _ in xrange(number_of_elements)]
        neg_flag = False
        for element in a:
            if element < 0:
                neg_flag = True
                break
        pos_flag = False
        for element in a:
            if element > 0:
                pos_flag = True
                break

        if neg_flag and pos_flag:
            assert(not EquivalentPosynomials.same_sign(a))
        else:
            assert(EquivalentPosynomials.same_sign(a))
    return


def test_correlated_monomials():
    for _ in xrange(100):
        number_of_monomials = int(50*np.random.random())+1
        number_of_gp_variables = int(np.random.rand()*20) + 1
        number_of_uncertain_variables = int(np.random.rand()*5) + 1
        vector_to_choose_from = [0, 0, 0, 0, 0, 0, 0, 0, 1, -1]

        m = number_of_monomials*[1]
        p_uncertain_vars = []
        correlations = []
        dependent_theoretical_partition = []

        for j in xrange(number_of_monomials):
            for i in xrange(number_of_gp_variables):
                x = Variable('x_%s' % i)
                m[j] *= x**(np.random.rand()*10 - 5)

        for i in xrange(number_of_uncertain_variables):
            u = Variable('u_%s' % i, np.random.random(), pr=100*np.random.random())
            p_uncertain_vars.append(u.key)
            neg_pos_neutral_powers = [vector_to_choose_from[int(10*np.random.random())] for _ in xrange(number_of_monomials)]
            same_sign = EquivalentPosynomials.same_sign(neg_pos_neutral_powers)

            if not same_sign:
                related_monomials = [i for i in xrange(number_of_monomials) if neg_pos_neutral_powers[i] != 0]
                correlations.append(related_monomials)

            for j in xrange(number_of_monomials):
                m[j] *= u**(np.random.rand()*5*neg_pos_neutral_powers[j])
                if neg_pos_neutral_powers[j] != 0:
                    dependent_theoretical_partition.append(j)

        theoretical_partition = EquivalentPosynomials.merge_intersected_lists(correlations)

        p = sum(m)

        equivalent_posynomial = EquivalentPosynomials(p, p_uncertain_vars, [], 0, False, False)
        equivalent_posynomial.main_p = p
        actual_partition = equivalent_posynomial.correlated_monomials()

        theoretical_posynomials = []
        actual_posynomials = []

        for theo_list, act_list in zip(theoretical_partition, actual_partition):
            theoretical_posynomials.append(sum([m[i] for i in theo_list]))
            actual_posynomials.append(sum([Monomial(p.exps[j], p.cs[j]) for j in act_list]))

        assert (set(actual_posynomials) == set(theoretical_posynomials))

        # dependent uncertainties
        temp = []
        for part in theoretical_partition:
            temp += part

        theoretical_partition = list(set(dependent_theoretical_partition))
        equivalent_posynomial = EquivalentPosynomials(p, p_uncertain_vars, [], 0, False, True)
        equivalent_posynomial.main_p = p
        actual_partition = equivalent_posynomial.correlated_monomials()

        theoretical_posynomials = []
        actual_posynomials = []

        theoretical_posynomials.append(sum([m[i] for i in theoretical_partition]))
        actual_posynomials.append(sum([Monomial(p.exps[j], p.cs[j]) for j in actual_partition[0]]))
        assert (set(actual_posynomials) == set(theoretical_posynomials))

    return


def test_check_if_in_list_of_lists():
    for _ in xrange(100):
        number_of_lists = int(10*np.random.random()) + 1
        list_of_lists = []
        number_of_elements = []
        for l in xrange(number_of_lists):
            number_of_elements.append(int(5*np.random.random()) + 1)
            list_of_lists.append([int(200*np.random.random()) for _ in xrange(number_of_elements[l])])

        element_list = int(number_of_lists*np.random.random())
        element = list_of_lists[element_list][int(number_of_elements[element_list]*np.random.random())]
        non_element = int(200*np.random.random()) + 200

        assert (EquivalentPosynomials.check_if_in_list_of_lists(element, list_of_lists))
        assert (not EquivalentPosynomials.check_if_in_list_of_lists(non_element, list_of_lists))

    return

def test():
    test_same_sign()
    test_merge_intersected_lists()
    test_correlated_monomials()
    test_check_if_in_list_of_lists()

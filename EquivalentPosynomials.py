from gpkit import Variable, Monomial, Posynomial
import numpy as np


class EquivalentPosynomials:
    """
    replaces a posynomial by an equivalent set of posynomials
    """

    p = Posynomial()

    def __init__(self, p):
        self.p = p

    @staticmethod
    def merge_intersected_lists(coupled_partition):
        """
        merges a list of lists so that the resulting list of lists is a partition
        :param coupled_partition: the list of lists to be merged
        :return: the partition
        """
        l = len(coupled_partition)
        if l <= 1:
            return coupled_partition

        half = int(np.floor(l / 2.0))

        first_half_partitions = EquivalentPosynomials.merge_intersected_lists(
            coupled_partition[0:half])
        second_half_partitions = EquivalentPosynomials.merge_intersected_lists(
            coupled_partition[half:l])

        len_first = len(first_half_partitions)
        len_second = len(second_half_partitions)

        relations = {}
        path = {}

        first_to_delete = set()
        second_to_delete = set()

        i = 0
        while i < len_first:
            j = 0
            while j < len_second:
                if list(set(first_half_partitions[i]) & set(second_half_partitions[j])):
                    second_to_delete.add(j)
                    temp_one = i
                    while temp_one in path:
                        temp_one = path[i]
                    first_half_partitions[temp_one] = \
                        list(set(first_half_partitions[temp_one]) | set(second_half_partitions[j]))
                    if j in relations:
                        first_to_delete.add(temp_one)
                        temp_two = relations[j]
                        while temp_two in path:
                            temp_two = path[i]
                        path[temp_one] = temp_two
                        first_half_partitions[temp_two] = \
                            list(set(first_half_partitions[temp_two]) | set(first_half_partitions[temp_one]))
                    else:
                        relations[j] = temp_one
                j += 1
            i += 1

        first_to_delete = list(first_to_delete)
        first_to_delete.reverse()
        second_to_delete = list(second_to_delete)
        second_to_delete.reverse()

        for k in first_to_delete:
            del first_half_partitions[k]
        for k in second_to_delete:
            del second_half_partitions[k]
        return first_half_partitions + second_half_partitions

    @staticmethod
    def same_sign(a):
        """
        Checks if the elements of a have the same sign
        :param a: the list to be checked
        :return: True or False
        """
        for i in xrange(len(a) - 1):
            if a[0] * a[i + 1] < 0:
                return False
        return True

    def correlated_monomials(self, p_uncertain_vars, dependent_uncertainties):
        """
        Creates partitions of correlated monomials
        :param p_uncertain_vars: the uncertain variables in the posynomial
        :param dependent_uncertainties: whether the uncertainty set is dependent or not
        :return: the list of coupled partitions
        """
        number_of_monomials = len(self.p.exps)
        coupled_partition = []

        if dependent_uncertainties:
            for j in xrange(number_of_monomials):
                for var in p_uncertain_vars:
                    if var.key in self.p.exps[j]:
                        coupled_partition.append(j)
                        break
            return [coupled_partition]

        else:
            for var in p_uncertain_vars:
                partition = []
                check_sign = []

                for j in xrange(number_of_monomials):
                    if var.key in self.p.exps[j]:
                        partition.append(j)
                        check_sign.append(self.p.exps[j].get(var.key))

                if not EquivalentPosynomials.same_sign(check_sign):
                    coupled_partition.append(partition)

            EquivalentPosynomials.merge_intersected_lists(coupled_partition)
            return coupled_partition

    @staticmethod
    def check_if_in_list_of_lists(element, list_of_lists):
        for i in xrange(len(list_of_lists)):
            if element in list_of_lists[i]:
                return True
        return False

    @staticmethod
    def check_if_no_data(p_uncertain_vars, monomial):
        """
        Checks if there is no uncertain data in a monomial
        :param p_uncertain_vars: the posynomial's uncertain variables
        :param monomial: the monomial to be checked for
        :return: True or False
        """
        intersection = [var for var in p_uncertain_vars if var.key in monomial]
        if not intersection:
            return True
        else:
            return False

    def equivalent_posynomial(self, uncertain_vars, m, simple_model, dependent_uncertainties):
        """
        creates a set of posynomials that are equivalent to the input posynomial
        :param uncertain_vars: the model's uncertain variables
        :param m: the index of the posynomial
        :param simple_model: if a simple model is preferred
        :param dependent_uncertainties: if the uncertainty set is dependent or not
        :return: the set of equivalent posynomials
        """

        p_uncertain_vars = [var for var in self.p.varkeys if var in uncertain_vars]
        l_p = len(p_uncertain_vars)

        # Check if there is no uncertain parameters in the posynomial
        if l_p == 0:
            return [self.p <= 1], []

        if len(self.p.exps) == 1:
            # Check if the posynomial is empty !!!!:
            if len(self.p.exps[0]) == 0:
                return [], []
            # Check if the posynomial is a monomial:
            else:
                return [], [self.p <= 1]

        # Check if uncertainties are common between all monomials:
        for i in xrange(len(self.p.exps)):
            m_uncertain_vars = [var for var in self.p.exps[i].keys()
                                if var in uncertain_vars]
            l = len(m_uncertain_vars)
            if l != l_p:
                dependent_uncertainties = False

        # Check if a simple model is preferred:
        if not simple_model:
            coupled_monomial_partitions = self.correlated_monomials(p_uncertain_vars, dependent_uncertainties)
        else:
            coupled_monomial_partitions = []

        # Check if all the monomials are related:
        if len(coupled_monomial_partitions) != 0 and len(coupled_monomial_partitions[0]) == len(self.p.exps):
            return [], [self.p <= 1]

        ts = []
        data_constraints, no_data_constraint = [], []

        elements = list(range(len(self.p.exps)))
        singleton_monomials = [element for element in elements if
                               not EquivalentPosynomials.check_if_in_list_of_lists(element,
                                                                                   coupled_monomial_partitions)]

        super_script = 0
        for i in singleton_monomials:
            if EquivalentPosynomials.check_if_no_data(p_uncertain_vars, self.p.exps[i]):
                ts.append(Monomial(self.p.exps[i], self.p.cs[i]))
            else:
                t = Variable('t_%s^%s' % (m, super_script))
                super_script += 1
                ts.append(t)
                data_constraints += [Monomial(self.p.exps[i], self.p.cs[i]) <= t]

        for i in xrange(len(coupled_monomial_partitions)):
            if coupled_monomial_partitions[i]:
                posynomial = 0
                t = Variable('t_%s^%s' % (m, super_script))
                super_script += 1
                ts.append(t)
                for j in coupled_monomial_partitions[i]:
                    posynomial += Monomial(self.p.exps[j], self.p.cs[j])
                data_constraints += [posynomial <= t]

        no_data_constraint += [sum(ts) <= 1]

        return no_data_constraint, data_constraints

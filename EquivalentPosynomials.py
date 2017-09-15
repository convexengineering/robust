import numpy as np

from gpkit import Variable, Monomial, Posynomial
from RobustGPTools import RobustGPTools


class EquivalentPosynomials:
    """
    replaces a posynomial by an equivalent set of posynomials
    """

    main_p = Posynomial()
    p_uncertain_vars = []
    p_indirect_uncertain_vars = []
    m = None
    simple_model = None
    dependent_uncertainties = None

    no_data_constraints = []
    data_constraints = []

    def __init__(self, p, uncertain_vars, indirect_uncertain_vars, m, simple_model, dependent_uncertainties):
        """
        :param p: the posynomial to be simplified
        :param uncertain_vars: the model's uncertain variables
        :param m: the index of the posynomial
        :param simple_model: if a simple model is preferred
        :param dependent_uncertainties: if the uncertainty set is dependent or not
        """
        self.simple_model = simple_model
        self.m = m
        self.dependent_uncertainties = dependent_uncertainties
        direct_p_uncertain_vars = [var for var in p.varkeys if var in uncertain_vars]
        self.p_indirect_uncertain_vars = [var for var in p.varkeys if var in indirect_uncertain_vars]

        new_direct_uncertain_vars = []
        for var in self.p_indirect_uncertain_vars:
            new_direct_uncertain_vars += RobustGPTools.\
                replace_indirect_uncertain_variable_by_equivalent(var.key.pr).varkeys

        new_direct_uncertain_vars = list(set(new_direct_uncertain_vars) & set(uncertain_vars))

        self.p_uncertain_vars = list(set(direct_p_uncertain_vars) | set(new_direct_uncertain_vars))
        # print self.p_uncertain_vars

        self.no_data_constraints = []
        self.data_constraints = []

        if len(p.exps[0]) == 0:
            self.main_p = p
            return

        number_of_p_uncertain_vars = len(self.p_uncertain_vars)
        if number_of_p_uncertain_vars == 0:
            self.no_data_constraints += [p <= 1]
            self.main_p = p
            return

        uncertain_vars_exps = []
        uncertain_vars_exps_mons = []
        for i in xrange(len(p.exps)):
            m_uncertain_vars_exps = {}

            only_uncertain_vars_monomial = RobustGPTools.\
                only_uncertain_vars_monomial(p.exps[i], p.cs[i], self.p_indirect_uncertain_vars)

            for var in only_uncertain_vars_monomial.exps[0].keys():
                if var in uncertain_vars:

                    m_uncertain_vars_exps[var] = only_uncertain_vars_monomial.exps[0][var]
            if m_uncertain_vars_exps in uncertain_vars_exps:
                index = uncertain_vars_exps.index(m_uncertain_vars_exps)
                uncertain_vars_exps_mons[index].append(i)
            else:
                uncertain_vars_exps.append(m_uncertain_vars_exps)
                uncertain_vars_exps_mons.append([i])

        all_data_mons = []
        for i, mon_list in enumerate(uncertain_vars_exps_mons):
            if len(mon_list) > 1 and uncertain_vars_exps[i]:
                new_no_data_posynomial = 0
                for j in mon_list:
                    new_no_data_posynomial += Monomial(p.exps[j], p.cs[j])/Monomial(uncertain_vars_exps[i])
                com_variable = Variable('com_%s^%s' % (m, i))
                self.no_data_constraints += [new_no_data_posynomial <= com_variable]
                all_data_mons.append(Monomial(uncertain_vars_exps[i])*com_variable)
            else:
                temp = sum([Monomial(p.exps[mon_ind], p.cs[mon_ind]) for mon_ind in mon_list])
                all_data_mons.append(temp)

        self.main_p = sum(all_data_mons)

        if len(self.main_p.exps) == 1:
            self.data_constraints += [self.main_p <= 1]
            return

        # if all(len(i) == 1 for i in uncertain_vars_exps_mons) and all():
        #    self.dependent_uncertainties = False

        if not simple_model:
            coupled_monomial_partitions = self.correlated_monomials()
        else:
            coupled_monomial_partitions = []

        # Check if all the monomials are related:
        if len(coupled_monomial_partitions) != 0 and len(coupled_monomial_partitions[0]) == len(self.main_p.exps):
            self.data_constraints += [self.main_p <= 1]
            return

        ts = []

        elements = list(range(len(self.main_p.exps)))
        singleton_monomials = [element for element in elements if
                               not EquivalentPosynomials.check_if_in_list_of_lists(element,
                                                                                   coupled_monomial_partitions)]

        super_script = 0
        for i in singleton_monomials:

            if RobustGPTools.\
                    check_if_no_data(self.p_uncertain_vars + self.p_indirect_uncertain_vars, self.main_p.exps[i]):
                ts.append(Monomial(self.main_p.exps[i], self.main_p.cs[i]))
            else:
                t = Variable('t_%s^%s' % (m, super_script))
                super_script += 1
                ts.append(t)
                self.data_constraints += [Monomial(self.main_p.exps[i], self.main_p.cs[i]) <= t]

        for i in xrange(len(coupled_monomial_partitions)):
            if coupled_monomial_partitions[i]:
                posynomial = 0
                t = Variable('t_%s^%s' % (m, super_script))
                super_script += 1
                ts.append(t)
                for j in coupled_monomial_partitions[i]:
                    posynomial += Monomial(self.main_p.exps[j], self.main_p.cs[j])
                self.data_constraints += [posynomial <= t]

        self.no_data_constraints += [sum(ts) <= 1]

        return

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

        first_half_partitions = EquivalentPosynomials.merge_intersected_lists(coupled_partition[0:half])
        second_half_partitions = EquivalentPosynomials.merge_intersected_lists(coupled_partition[half:l])

        len_first, len_second = len(first_half_partitions), len(second_half_partitions)

        relations, path = {}, {}

        first_to_delete, second_to_delete = set(), set()

        i = 0
        while i < len_first:
            j = 0
            while j < len_second:
                if list(set(first_half_partitions[i]) & set(second_half_partitions[j])):
                    second_to_delete.add(j)
                    temp_one = i
                    while temp_one in path:
                        temp_one = path[temp_one]  # i]
                    first_half_partitions[temp_one] = \
                        list(set(first_half_partitions[temp_one]) | set(second_half_partitions[j]))
                    if j in relations:
                        temp_two = relations[j]
                        while temp_two in path:
                            temp_two = path[temp_two]  # i]
                        if temp_two != temp_one:
                            first_to_delete.add(temp_one)
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
        j = 0
        while j < len(a):
            if a[j] == 0:
                j += 1
            else:
                break

        if j == len(a):
            return True

        for i in xrange(len(a) - 1):
            if a[j] * a[i + 1] < 0:
                return False
        return True

    def correlated_monomials(self):
        """
        Creates partitions of correlated monomials
        :return: the list of coupled partitions
        """
        number_of_monomials = len(self.main_p.exps)
        coupled_partition = []

        if self.dependent_uncertainties:
            for j in xrange(number_of_monomials):

                only_uncertain_vars_monomial = RobustGPTools.\
                    only_uncertain_vars_monomial(self.main_p.exps[j], self.main_p.cs[j], self.p_indirect_uncertain_vars)

                for var in self.p_uncertain_vars:
                    if var.key in only_uncertain_vars_monomial.exps[0]:
                        coupled_partition.append(j)
                        break
            return [coupled_partition]

        else:
            for var in self.p_uncertain_vars:
                partition = []
                check_sign = []

                for j in xrange(number_of_monomials):
                    # print("ejer")
                    only_uncertain_vars_monomial = RobustGPTools.\
                        only_uncertain_vars_monomial(self.main_p.exps[j], self.main_p.cs[j],
                                                     self.p_indirect_uncertain_vars)
                    # print("ejer")
                    # print only_uncertain_vars_monomial
                    # print var.key
                    # print [i.key for i in only_uncertain_vars_monomial.exps[0].keys()]
                    if var in only_uncertain_vars_monomial.exps[0]:
                        # print "kes"
                        partition.append(j)
                        check_sign.append(only_uncertain_vars_monomial.exps[0].get(var.key))

                if not EquivalentPosynomials.same_sign(check_sign):
                    coupled_partition.append(partition)

            coupled_partition = EquivalentPosynomials.merge_intersected_lists(coupled_partition)
            return coupled_partition

    @staticmethod
    def check_if_in_list_of_lists(element, list_of_lists):
        for i in xrange(len(list_of_lists)):
            if element in list_of_lists[i]:
                return True
        return False

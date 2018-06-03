from LinearizeTwoTermPosynomials import LinearizeTwoTermPosynomials
import numpy as np


def construct_linearization_data(max_num_of_linear_sections):
    file = open("data/linearization_data.txt", "w")
    for i in range(2, max_num_of_linear_sections):
        print i
        a = LinearizeTwoTermPosynomials.compute_two_term_posynomial_linearization_coeff(i, 2*np.finfo(float).eps)
        for j in xrange(4):
            for item in a[j]:
                file.write("%s, " % item)
            file.write(": ")
        file.write("%.16s" % a[4])
        file.write("\n")
    file.close()

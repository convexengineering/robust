from __future__ import print_function
from builtins import range
import numpy as np

from robust.linearize_twoterm_posynomials import LinearizeTwoTermPosynomials
import os


def construct_linearization_data(max_num_of_linear_sections, the_file_path):
    the_file = open(the_file_path, "w")
    for i in range(2, max_num_of_linear_sections):
        print(i)
        a = LinearizeTwoTermPosynomials.compute_two_term_posynomial_linearization_coeff(i, 2*np.finfo(float).eps)
        for j in range(4):
            for item in a[j]:
                the_file.write("%s, " % item)
            the_file.write(": ")
        the_file.write("%.16s" % a[4])
        the_file.write("\n")
    the_file.close()

if __name__ == '__main__':
    construct_linearization_data(100, os.path.dirname(__file__) + "/linearization_data.txt")

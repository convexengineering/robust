from LinearizeTwoTermPosynomials import LinearizeTwoTermPosynomials
import numpy as np


def construct_linearization_data(max_num_of_linear_sections):
    the_file = open("/home/saab/Dropbox (MIT)/MIT/Masters/Code/robust/data/linearization_data.txt", "w")
    for i in range(2, max_num_of_linear_sections):
        print i
        a = LinearizeTwoTermPosynomials.compute_two_term_posynomial_linearization_coeff(i, 2*np.finfo(float).eps)
        for j in xrange(4):
            for item in a[j]:
                the_file.write("%s, " % item)
            the_file.write(": ")
        the_file.write("%.16s" % a[4])
        the_file.write("\n")
    the_file.close()

if __name__ == '__main__':
    construct_linearization_data(100)

"""Script for running all gpkit unit tests"""
from gpkit.tests.run_tests import run

def import_tests():
    """Get a list of all robust unit test TestCases"""
    tests = []

    from robust.testing import t_equivalent_posynomials
    tests += t_equivalent_posynomials.TESTS

    from robust.testing import t_linearization
    tests += t_linearization.TESTS

    from robust.testing import t_primitives
    tests += t_primitives.TESTS

    from robust.testing import t_robust_gp_tools
    tests += t_robust_gp_tools.TESTS

    from robust.testing import t_two_term_approximation
    tests += t_two_term_approximation.TESTS

    return tests


def test():
    run(tests=import_tests())


if __name__ == '__main__':
    test()

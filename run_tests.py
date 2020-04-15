"""Script for running all unit tests"""
import gpkit
from gpkit.tests.run_tests import run
from gpkit.tests.test_repo import git_clone, pip_install

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

    from robust.testing import t_simulation
    tests += t_simulation.TESTS

    from robust.testing import t_legacy
    tests += t_legacy.TESTS

    return tests

def test(xmloutput=True):
    try:
        import gpkitmodels
    except:
        git_clone("gplibrary")
        pip_install("gplibrary", local=True)
    alltests = import_tests()
    TESTS = []
    for testcase in alltests:
        for solver in gpkit.settings["installed_solvers"]:
            if solver:
                test = type(str(testcase.__name__+"_"+solver),
                            (testcase,), {})
                setattr(test, "solver", solver)
                TESTS.append(test)
    run(tests=TESTS, xmloutput=xmloutput)

if __name__ == '__main__':
    test(xmloutput=False)

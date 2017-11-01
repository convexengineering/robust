from Robust import RobustModel
import GPModels as Models
from plot_feasibilities import plot_feasibilities


# for you to see the structure of the robust model, check the variables ending with _robust_model
def test():
    # tm = Models.test_model()
    # tm.solve()
    # tm_box_uncertain_coeff = RobustModel(tm, 'box', two_term=False)
    # # tm_box_uncertain_coeff.setup() creates the robust model and saves it in tm_box_uncertain_coeff.robust_model
    # sol_tm_box_uncertain_coeff = tm_box_uncertain_coeff.robustsolve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # # otherwise it will directly solve the robust_model
    # tm_box_uncertain_coeff_robust_model = tm_box_uncertain_coeff.get_robust_model()
    # plot_feasibilities(tm["a"], tm["b"], tm, tm_box_uncertain_coeff_robust_model, "box")
    #
    # tm = Models.test_model()
    # tm.solve()
    # tm_ell_uncertain_coeff = RobustModel(tm, 'elliptical', two_term=False)
    # sol_tm_ell_uncertain_coeff = tm_ell_uncertain_coeff.robustsolve(verbosity=1)
    # tm_ell_uncertain_coeff_robust_model = tm_ell_uncertain_coeff.get_robust_model()
    # plot_feasibilities(tm["a"], tm["b"], tm, tm_ell_uncertain_coeff_robust_model, "elliptical")
    #
    # tm = Models.test_model()
    # tm.solve()
    # tm_box_uncertain_exponents = RobustModel(tm, 'box')  # this method can handle uncertain exponents,
    # # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    # sol_tm_box_uncertain_exponents = tm_box_uncertain_exponents.robustsolve(verbosity=1)
    # tm_box_uncertain_exponents_robust_model = tm_box_uncertain_exponents.get_robust_model()
    # plot_feasibilities(tm["a"], tm["b"], tm, tm_box_uncertain_exponents_robust_model, "box")
    #
    # tm = Models.test_model()
    # tm.solve()
    # tm_ell_uncertain_exponents = RobustModel(tm, 'elliptical')
    # sol_tm_ell_uncertain_exponents = tm_ell_uncertain_exponents.robustsolve(verbosity=1)
    # tm_ell_uncertain_exponents_robust_model = tm_ell_uncertain_exponents.get_robust_model()
    # plot_feasibilities(tm["a"], tm["b"], tm, tm_ell_uncertain_exponents_robust_model, "elliptical")

    # simp = Models.simpleWing()
    # simp.solve()
    # simp_box_uncertain_coeff = RobustModel(simp, 'box', two_term=False)
    # # simp_box_uncertain_coeff.setup() creates the robust model and saves it in simp_box_uncertain_coeff.robust_model
    # sol_simp_box_uncertain_coeff = simp_box_uncertain_coeff.robustsolve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # # otherwise it will directly solve the robust_model
    # simp_box_uncertain_coeff_robust_model = simp_box_uncertain_coeff.get_robust_model()
    # plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "box")
    #
    import numpy as np
    simp = Models.simpleWing()
    # simp.solve()
    x, y = simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"]
    eta_max_x = np.log(1 + x.key.pr / 100.0)
    eta_min_x = np.log(1 - x.key.pr / 100.0)
    center_x = (eta_min_x + eta_max_x) / 2.0
    eta_max_y = np.log(1 + y.key.pr / 100.0)
    eta_min_y = np.log(1 - y.key.pr / 100.0)
    center_y = (eta_min_y + eta_max_y) / 2.0
    xc = x.key.value/np.exp(center_x)
    yc = y.key.value/np.exp(center_y)
    print simp.substitutions[y], yc
    simp.substitutions[x] = xc
    simp.substitutions[y] = yc
    # print simp.solution(y)
    simp_ell_uncertain_coeff = RobustModel(simp, 'elliptical', two_term=False)
    sol_simp_ell_uncertain_coeff = simp_ell_uncertain_coeff.robustsolve(verbosity=1)
    simp_ell_uncertain_coeff_robust_model = simp_ell_uncertain_coeff.get_robust_model()
    # print simp.solution(y)  # why does this change??
    simp.substitutions[x] = x.key.value
    simp.substitutions[y] = y.key.value
    simp.solve()
    plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
                       simp_ell_uncertain_coeff_robust_model, "elliptical")
    #
    # simp = Models.simpleWing()
    # simp.solve()
    # simp_box_uncertain_exponents = RobustModel(simp, 'box')  # this method can handle uncertain exponents,
    # # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    # sol_simp_box_uncertain_exponents = simp_box_uncertain_exponents.robustsolve(verbosity=1)
    # simp_box_uncertain_exponents_robust_model = simp_box_uncertain_exponents.get_robust_model()
    # plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
    #                    simp_box_uncertain_exponents_robust_model, "box")
    #
    # simp = Models.simpleWing()
    # simp.solve()
    # simp_ell_uncertain_exponents = RobustModel(simp, 'elliptical')
    # sol_simp_ell_uncertain_exponents = simp_ell_uncertain_exponents.robustsolve(verbosity=1)
    # simp_ell_uncertain_exponents_robust_model = simp_ell_uncertain_exponents.get_robust_model()
    # plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
    #                    simp_ell_uncertain_exponents_robust_model, "elliptical")
    #
    # simpSP = Models.simpleWing()
    # simpSP.localsolve()
    # simp_box_uncertain_coeff = RobustModel(simpSP, 'box', two_term=False)
    # # simp_box_uncertain_coeff.setup() creates the robust model and saves it in simp_box_uncertain_coeff.robust_model
    # sol_simp_box_uncertain_coeff = simp_box_uncertain_coeff.robustsolve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # # otherwise it will directly solve the robust_model
    # simp_box_uncertain_coeff_robust_model = simp_box_uncertain_coeff.get_robust_model()
    # plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "box")
    #
    # simpSP = Models.simpleWing()
    # simpSP.localsolve()
    # simp_ell_uncertain_coeff = RobustModel(simpSP, 'elliptical', two_term=False)
    # sol_simp_ell_uncertain_coeff = simp_ell_uncertain_coeff.robustsolve(verbosity=1)
    # simp_ell_uncertain_coeff_robust_model = simp_ell_uncertain_coeff.get_robust_model()
    # plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "box")
    #
    # simpSP = Models.simpleWing()
    # simpSP.localsolve()
    # simp_box_uncertain_exponents = RobustModel(simpSP, 'box')  # this method can handle uncertain exponents,
    # # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    # sol_simp_box_uncertain_exponents = simp_box_uncertain_exponents.robustsolve(verbosity=1)
    # simp_box_uncertain_exponents_robust_model = simp_box_uncertain_exponents.get_robust_model()
    # plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "box")
    #
    # simpSP = Models.simpleWing()
    # simpSP.localsolve()
    # simp_ell_uncertain_exponents = RobustModel(simpSP, 'elliptical')
    # sol_simp_ell_uncertain_exponents = simp_ell_uncertain_exponents.robustsolve(verbosity=1)
    # simp_ell_uncertain_exponents_robust_model = simp_ell_uncertain_exponents.get_robust_model()
    # plot_feasibilities(simp["W_{W_{coeff1}}"], simp["W_{W_{coeff2}}"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "box")

    # simp = Models.simpleWingTwoDimensionalUncertainty()
    # simp.solve()
    # simp_box_uncertain_coeff = RobustModel(simp, 'box', two_term=False)
    # # simp_box_uncertain_coeff.setup() creates the robust model and saves it in simp_box_uncertain_coeff.robust_model
    # sol_simp_box_uncertain_coeff = simp_box_uncertain_coeff.robustsolve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # # otherwise it will directly solve the robust_model
    # simp_box_uncertain_coeff_robust_model = simp_box_uncertain_coeff.get_robust_model()
    # plot_feasibilities(simp["k"], simp["toz"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "box")
    #
    # simp = Models.simpleWingTwoDimensionalUncertainty()
    # simp.solve()
    # simp_ell_uncertain_coeff = RobustModel(simp, 'elliptical', two_term=False)
    # sol_simp_ell_uncertain_coeff = simp_ell_uncertain_coeff.robustsolve(verbosity=1)
    # simp_ell_uncertain_coeff_robust_model = simp_ell_uncertain_coeff.get_robust_model()
    # plot_feasibilities(simp["k"], simp["toz"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "elliptical")
    #
    # simp = Models.simpleWingTwoDimensionalUncertainty()
    # simp.solve()
    # simp_box_uncertain_exponents = RobustModel(simp, 'box')  # this method can handle uncertain exponents,
    # # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    # sol_simp_box_uncertain_exponents = simp_box_uncertain_exponents.robustsolve(verbosity=1)
    # simp_box_uncertain_exponents_robust_model = simp_box_uncertain_exponents.get_robust_model()
    # plot_feasibilities(simp["k"], simp["toz"], simp,
    #                    simp_box_uncertain_coeff_robust_model, "box")
    #
    # simp = Models.simpleWingTwoDimensionalUncertainty()
    # simp.solve()
    # simp_ell_uncertain_exponents = RobustModel(simp, 'elliptical')
    # sol_simp_ell_uncertain_exponents = simp_ell_uncertain_exponents.robustsolve(verbosity=1)
    # simp_ell_uncertain_exponents_robust_model = simp_ell_uncertain_exponents.get_robust_model()
    # plot_feasibilities(simp["k"], simp["toz"], simp,
    #                    simp_ell_uncertain_exponents_robust_model, "elliptical")

if __name__ == "__main__":
    test()

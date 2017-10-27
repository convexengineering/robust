from Robust import RobustModel
import GPModels as Models


# for you to see the structure of the robust model, check the variables ending with _robust_model
def test():
    tm = Models.test_model()

    tm_box_uncertain_coeff = RobustModel(tm, 'box', two_term=False)
    # tm_box_uncertain_coeff.setup() creates the robust model and saves it in tm_box_uncertain_coeff.robust_model
    sol_tm_box_uncertain_coeff = tm_box_uncertain_coeff.robustsolve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # otherwise it will directly solve the robust_model
    tm_box_uncertain_coeff_robust_model = tm_box_uncertain_coeff.get_robust_model()

    tm_ell_uncertain_coeff = RobustModel(tm, 'elliptical', two_term=False)
    sol_tm_ell_uncertain_coeff = tm_ell_uncertain_coeff.robustsolve(verbosity=1)
    tm_ell_uncertain_coeff_robust_model = tm_ell_uncertain_coeff.get_robust_model()

    tm_box_uncertain_exponents = RobustModel(tm, 'box')  # this method can handle uncertain exponents,
    # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    sol_tm_box_uncertain_exponents = tm_box_uncertain_exponents.robustsolve(verbosity=1)
    tm_box_uncertain_exponents_robust_model = tm_box_uncertain_exponents.get_robust_model()

    tm_ell_uncertain_exponents = RobustModel(tm, 'elliptical')
    sol_tm_ell_uncertain_exponents = tm_ell_uncertain_exponents.robustsolve(verbosity=1)
    tm_ell_uncertain_exponents_robust_model = tm_ell_uncertain_exponents.get_robust_model()

    simp = Models.simpleWing()

    simp_box_uncertain_coeff = RobustModel(simp, 'box', two_term=False)
    # simp_box_uncertain_coeff.setup() creates the robust model and saves it in simp_box_uncertain_coeff.robust_model
    sol_simp_box_uncertain_coeff = simp_box_uncertain_coeff.robustsolve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # otherwise it will directly solve the robust_model
    simp_box_uncertain_coeff_robust_model = simp_box_uncertain_coeff.get_robust_model()

    simp_ell_uncertain_coeff = RobustModel(simp, 'elliptical', two_term=False)
    sol_simp_ell_uncertain_coeff = simp_ell_uncertain_coeff.robustsolve(verbosity=1)
    simp_ell_uncertain_coeff_robust_model = simp_ell_uncertain_coeff.get_robust_model()

    simp_box_uncertain_exponents = RobustModel(simp, 'box')  # this method can handle uncertain exponents,
    # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    sol_simp_box_uncertain_exponents = simp_box_uncertain_exponents.robustsolve(verbosity=1)
    simp_box_uncertain_exponents_robust_model = simp_box_uncertain_exponents.get_robust_model()

    simp_ell_uncertain_exponents = RobustModel(simp, 'elliptical')
    sol_simp_ell_uncertain_exponents = simp_ell_uncertain_exponents.robustsolve(verbosity=1)
    simp_ell_uncertain_exponents_robust_model = simp_ell_uncertain_exponents.get_robust_model()

    simpSP = Models.simpleWing()

    simp_box_uncertain_coeff = RobustModel(simpSP, 'box', two_term=False)
    # simp_box_uncertain_coeff.setup() creates the robust model and saves it in simp_box_uncertain_coeff.robust_model
    sol_simp_box_uncertain_coeff = simp_box_uncertain_coeff.robustsolve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # otherwise it will directly solve the robust_model
    simp_box_uncertain_coeff_robust_model = simp_box_uncertain_coeff.get_robust_model()

    simp_ell_uncertain_coeff = RobustModel(simpSP, 'elliptical', two_term=False)
    sol_simp_ell_uncertain_coeff = simp_ell_uncertain_coeff.robustsolve(verbosity=1)
    simp_ell_uncertain_coeff_robust_model = simp_ell_uncertain_coeff.get_robust_model()

    simp_box_uncertain_exponents = RobustModel(simpSP, 'box')  # this method can handle uncertain exponents,
    # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    sol_simp_box_uncertain_exponents = simp_box_uncertain_exponents.robustsolve(verbosity=1)
    simp_box_uncertain_exponents_robust_model = simp_box_uncertain_exponents.get_robust_model()

    simp_ell_uncertain_exponents = RobustModel(simpSP, 'elliptical')
    sol_simp_ell_uncertain_exponents = simp_ell_uncertain_exponents.robustsolve(verbosity=1)
    simp_ell_uncertain_exponents_robust_model = simp_ell_uncertain_exponents.get_robust_model()

if __name__ == "__main__":
    test()

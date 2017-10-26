from RobustGP import RobustGPModel
import GPModels as Models


# for you to see the structure of the robust model, check the variables ending with _robust_model
def test():
    tm = Models.test_model()

    tm_box_uncertain_coeff = RobustGPModel(tm, 1, 'box')
    # tm_box_uncertain_coeff.setup() creates the robust model and saves it in tm_box_uncertain_coeff.robust_model
    sol_tm_box_uncertain_coeff = tm_box_uncertain_coeff.solve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # otherwise it will directly solve the robust_model
    tm_box_uncertain_coeff_robust_model = tm_box_uncertain_coeff.robust_model

    tm_ell_uncertain_coeff = RobustGPModel(tm, 1, 'elliptical')
    sol_tm_ell_uncertain_coeff = tm_ell_uncertain_coeff.solve(verbosity=1)
    tm_ell_uncertain_coeff_robust_model = tm_ell_uncertain_coeff.robust_model

    tm_box_uncertain_exponents = RobustGPModel(tm, 1, 'box', two_term=True)  # this method can handle uncertain exponents,
    # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    sol_tm_box_uncertain_exponents = tm_box_uncertain_exponents.solve(verbosity=1)
    tm_box_uncertain_exponents_robust_model = tm_box_uncertain_exponents.robust_model

    tm_ell_uncertain_exponents = RobustGPModel(tm, 1, 'elliptical', two_term=True)
    sol_tm_ell_uncertain_exponents = tm_ell_uncertain_exponents.solve(verbosity=1)
    tm_ell_uncertain_exponents_robust_model = tm_ell_uncertain_exponents.robust_model

    simp = Models.simpleWing()

    simp_box_uncertain_coeff = RobustGPModel(simp, 1, 'box')
    # simp_box_uncertain_coeff.setup() creates the robust model and saves it in simp_box_uncertain_coeff.robust_model
    sol_simp_box_uncertain_coeff = simp_box_uncertain_coeff.solve(verbosity=1)  # calls the setup() if the robust model is not created yet,
    # otherwise it will directly solve the robust_model
    simp_box_uncertain_coeff_robust_model = simp_box_uncertain_coeff.robust_model

    simp_ell_uncertain_coeff = RobustGPModel(simp, 1, 'elliptical')
    sol_simp_ell_uncertain_coeff = simp_ell_uncertain_coeff.solve(verbosity=1)
    simp_ell_uncertain_coeff_robust_model = simp_ell_uncertain_coeff.robust_model

    simp_box_uncertain_exponents = RobustGPModel(simp, 1, 'box', two_term=True)  # this method can handle uncertain exponents,
    # we do not need signomial constraints, and in most cases it is better than the method for uncertain coefficients.
    sol_simp_box_uncertain_exponents = simp_box_uncertain_exponents.solve(verbosity=1)
    simp_box_uncertain_exponents_robust_model = simp_box_uncertain_exponents.robust_model

    simp_ell_uncertain_exponents = RobustGPModel(simp, 1, 'elliptical', two_term=True)
    sol_simp_ell_uncertain_exponents = simp_ell_uncertain_exponents.solve(verbosity=1)
    simp_ell_uncertain_exponents_robust_model = simp_ell_uncertain_exponents.robust_model

if __name__ == "__main__":
    test()

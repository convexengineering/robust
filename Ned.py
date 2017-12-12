from os import sep
from os.path import abspath, dirname
import numpy as np
import pandas as pd
from gpkit import Model, parse_variables
from gpfit.fit_constraintset import XfoilFit

class ejer(Model):
    """ejer class for test

    scalar Variables
    ----------------
    S                                   [ft^2]  surface area
    AR                                  [-]     aspect ratio
    b               1                   [ft]    span
    tau             0.115               [-]     airfoil thickness ratio
    """
    def setup(self):
        exec parse_variables(ejer.__doc__)
        tau.key.descr['pr'] = 1
        self.cost = 1/S
        return [S <= 2*b**2]

ob = ejer()

Getting started
===============

Once you have installed **robust**, and have a GP or SP model, you are ready to begin.
From here onward, we will use *nominal* to describe models with no uncertainty straight
out of GPkit, and *robust* to describe models that have been robustified using **robust**.

The uncertainties in **robust** are defined by adding attribute *pr* to any variable
in your model. This attribute
describes the :math:`3\sigma` uncertainty for the given parameter, normalized by its mean (otherwise known
as 3 times the coefficient of variation, eg. :math:`pr = 10`
would specify a 10% 3CV). Note that these attributes
are carried by nominal models but only come into effect when **robust** is applied.

.. code-block:: python

    from gpkit import Variable, Model
    x = Variable('x', pr = 10)
    % ...
    % after more variables, constraints
    % ...
    m = Model(objective, constraints, substitutions)

Once you have added uncertainties to parameters, and created a GPkit model,
robustifying said model and solving it is easy. The most straight-forward
inputs for uncertainty_set are 'box' or 'elliptical'. *gamma* defines the size of
the uncertainty set protected against, where *gamma=1* protects against :math:`3\sigma`
uncertainty.

.. code-block:: python

    from robust.robust import RobustModel
    rm = RobustModel(m, uncertainty_set, gamma = float)
    rsol = rm.robustsolve()

You have solved your robust model! To be able to quickly compare the robust solution *rsol* with the nominal solution *sol*,
we recommend you try 'diffing' the two, which is done as follows:

.. code-block:: python

    print rsol.diff(sol)

This will allow you to see the percent differences between the two designs!
Since the robust design protects against uncertainty in the parameters, it will necessarily
have lower performance than the nominal design.
If this has piqued your interest, please continue to explore the documentation.

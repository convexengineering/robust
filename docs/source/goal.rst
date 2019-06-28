Goal programming
****************

Recall the standard RO form below

.. math::

    \begin{split}
        \text{min} &~~f_0(x) \\
    \text{s.t.}     &~~\underset{u}{\text{max}}~f_i(x,u) \leq 0,~i = 1,\ldots,n \\
                    &~~\left\lVert u \right\rVert \leq \Gamma \\
        \end{split}

where we attempt to minimize :math:`f_0` for the worst-case realization of
uncertain parameters in the set. We can flip this on its head, and
solve the following problem

.. math::

    \begin{split}
    \text{max}~~\Gamma \\
    \text{s.t.}~~f_i(x,u) &\leq 0,~i = 1,\ldots,n \\
                    \left\lVert u \right\rVert &\leq \Gamma \\
                    f_0(x) &\leq (1+\delta)f_0^*,~\delta \geq 0
    \end{split}

where :math:`f_0^*` is the optimum of the nominal problem and :math:`\delta`
is a fractional penalty on the objective that we are willing to sacrifice for robustness, which
gives :math:`(1+\delta)f_0^*` as the upper bound on the objective value.

The benefit of this goal programming form is that we can start to use risk as a global
design variable against with all objectives can be weighed.

Implementation
--------------

To use the goal programming functions in **robust**, you can use the ```simulate.variable_goal_results```
function, which has the same inputs as ```simulate.variable_gamma_result``` except for
having the penalty parameter :math:`\delta` instead of the uncertainty set size :math:`\Gamma` as its input.

What occurs to the solved nominal model under the hood is the following:

.. code-block:: python

    Gamma = Variable('\\Gamma', '-', 'Uncertainty bound')
    delta = Variable(value, '1+\\delta', '-', 'Acceptable optimal solution bound', fix = True)
    origcost = model.cost
    mGoal = Model(1 / Gamma, [model, origcost <= Monomial(nominal_solution(origcost)) * delta, Gamma <= 1e30, delta <= 1e30],
                  model.substitutions)
    robust_goal_model = RobustModel(mGoal, uncertainty_set, gamma=Gamma)
    sol = robust_goal_model.robustsolve()

This is an exact formulation of the aforementioned risk minimization problem!


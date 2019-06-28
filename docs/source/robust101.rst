Robust optimization 101
***********************

RO is a tractable method for optimization under uncertainty, and specifically under uncertain
parameters. It optimizes the worst-case objective outcome over uncertainty sets,
unlike general stochastic optimization methods which optimize statistics of the distribution
of the objective over probability distributions of uncertain parameters. As such, RO
sacrifices generality for tractability, probabilistic guarantees and engineering intuition.

Basic mathematical principles
-----------------------------

[*paraphrased from Ozturk and Saab, 2019*]

Given a general optimization problem under parametric uncertainty, we define the set of possible
realizations of uncertain vector of parameters :math:`u` in the uncertainty set :math:`\mathcal{U}`. This
allows us to define the problem under uncertainty below.

.. math::

    \text{min} &~~f_0(x) \\
    \text{s.t.}     &~~f_i(x,u) \leq 0,~\forall u \in \mathcal{U},~i = 1,\ldots,n

This problem is infinite-dimensional, since it is possible to formulate an infinite number of constraints
with the countably infinite number of possible realizations of :math:`u \in \mathcal{U}`. To circumvent this issue,
we can define the following robust formulation of the uncertain problem below.

.. math::

    \text{min} &~~f_0(x) \\
    \text{s.t.}     &~~\underset{u \in \mathcal{U}}{\text{max}}~f_i(x,u) \leq 0,~i = 1,\ldots,n

This formulation hedges against the worst-case realization of the uncertainty in the defined uncertainty
set. The set is often described by a norm, which contains possible uncertain outcomes from distributions with
bounded support

.. math::

    \begin{split}
        \text{min} &~~f_0(x) \\
    \text{s.t.}     &~~\underset{u}{\text{max}}~f_i(x,u) \leq 0,~i = 1,\ldots,n \\
                    &~~\left\lVert u \right\rVert \leq \Gamma \\
        \end{split}

where :math:`\Gamma` is defined by the user as a global uncertainty bound. The larger the :math:`\Gamma`,
the greater the size of the uncertainty set that is protected against!


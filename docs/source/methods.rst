Approximations for tractable robust GPs
***************************************

Within **robust**, there are 3 tractable approximate robust formulations for
GPs, which can then be extended to SPs through heuristics
The methods are detailed at a high level below, in decreasing order of conservativeness.
Please see [Saab, 2018] for further details.

*(The following overview has been paraphrased from [Ozturk, 2019].)*

The robust counterpart of an uncertain geometric program is:

.. math::

    \begin{split}
        \min &~~f_0\left(\mathbf{x}\right)\\
        \text{s.t.} &~~\max_{\mathbf{\zeta} \in \mathcal{Z}} \left\{\textstyle{\sum}_{k=1}^{K_i}e^{\mathbf{a_{ik}}\left(\zeta\right)\mathbf{x} + b_{ik}\left(\zeta\right)}\right\} \leq 1, ~\forall i \in 1,...,m\\
    \end{split}

which is Co-NP hard in its natural posynomial form [Chassein, 2014]. We will present three approximate formulations of a RGP.

Simple Conservative Approximation
---------------------------------

One way to approach the intractability of the above problem is to replace each constraint by a tractable approximation.
Replacing the max-of-sum by the sum-of-max will lead to the following formulation.

.. math::

    \begin{split}
        \min &~~f_0\left(\mathbf{x}\right)\\
        \text{s.t.} &~~\textstyle{\sum}_{k=1}^{K_i} {\displaystyle \max_{\mathbf{\zeta} \in \mathcal{Z}}} \left\{e^{\mathbf{a_{ik}}\left(\zeta\right)\mathbf{x} + b_{ik}\left(\zeta\right)}\right\} \leq 1, ~\forall i \in 1,...,m
    \end{split}

Maximizing a monomial term is equivalent to maximizing an affine function, therefore the Simple Conservative approximation is tractable.

Linearized Perturbations
------------------------

The Linearized Perturbations formulation separates large posynomials
into decoupled posynomials, depending on the dependence of monomial terms.
If the exponents are known and certain, then large posynomial constraints can be approximated as signomial constraints.
The exponential perturbations in each posynomial are linearized using a modified least squares method, and then the
posynomial is robustified using techniques from robust linear programming. The resulting set of constraints is SP-compatible,
therefore, a robust GP can be approximated as a SP.

Best Pairs
----------

If the exponents of a posynomial are uncertain as well as the coefficients,
then large posynomials can't be approximated as a SP, and further simplification is needed.
This formulation allows for uncertain exponents, by maximizing each pair of monomials in each posynomial,
while finding the best combination of monomials that gives the least conservative solution.
[Saab, 2018] provides a descent algorithm to find locally optimal combinations of the monomials,
and shows how the uncertain GP can be approximated as a GP for polyhedral uncertainty,
and a conic optimization problem for elliptical uncertainty with uncertain exponents.

To reiterate, please refer to [Saab, 2018] for further details
on robust GP approximations.

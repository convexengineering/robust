.. _rspapproaches:

Approaches to solving robust SPs
================================

*(borrowed from [Ozturk, 2019])*

This section presents a heuristic algorithm to solve a RSP
based on our previous discussion on robust geometric programming.

General RSP Solver
------------------

A common heuristic algorithm to solve a SP is
by sequentially solving local GP approximations.
Similarly, our approach to solve a RSP is based on solving
a sequence of local RGP approximations. 
In this heuristic, a good initial guess will lead to faster
convergence and possibly a better solution.
The deterministic solution of the uncertain SP is in general a good candidate :math:`x_0`.

|rspSolve|

.. |rspSolve| image:: rspSolve.png
        :width: 80%

For comparisons between methods ahead, we write the algorithm explicitly as follows:

    - Choose an initial guess :math:`x_0`.
    - Repeat:

      - Find the local GP approximation of the SP at :math:`x_i`.
      - Find the RGP formulation of the GP.
      - Solve the RGP to obtain :math:`x_{i+1}`.
      - If :math:`x_{i+1} \approx x_{i}`: break


Any of the previously mentioned methodologies can be used to formulate the local RGP approximation. 
However, depending on the RGP formulation chosen to solve a RSP, the formulation and solution
blocks in the above figure are adjusted.

Best Pairs RSP Solver
---------------------

If the Best Pairs methodology is exploited, then the above algorithm would change so that
each iteration would solve the local RGP approximation and choose the best permutation
for each large posynomial. The modified algorithm would become as follows:

    - Choose an initial guess :math:`x_0`.
    - Repeat:

      - Find the local GP approximation of the SP at :math:`x_i`.
      - For each large posynomial constraint, select the new permutation :math:`\phi` such that :math:`\phi` minimizes the robust large constraint evaluated at :math:`x_i`.
      - Solve the approximate tractable counterparts of the local GP, and let :math:`\mathbf{x}_{i+1}` be the solution.
      - If :math:`x_{i+1} \approx x_{i}`: break.

Linearized Perturbations RSP Solver
-----------------------------------

On the other hand, if the Linearized Perturbations formulation is to be used,
then we can avoid solving a SP at each iteration by first
approximating the original SP constraints locally, and in the same loop approximating
the robustified possibly signomial constraints locally, thus solving a
GP at each iteration instead of a SP. The algorithm would then become as follows:

    - Choose an initial guess :math:`x_0`.
    - Repeat:

      - Find the local GP approximation of the SP at :math:`x_i`.
      - Robustify the constraints of the local GP approximation using the Linearized Perturbations methodology.
      - Find the local GP approximation of the resulting local SP at :math:`x_i`.
      - Solve the local GP approximation in step c to obtain $x_{i+1}$.
      - If :math:`x_{i+1} \approx x_{i}`: break.

Work in progress...

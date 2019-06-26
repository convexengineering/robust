Robustification methods
***********************

Within **robust**, there are 3 tractable approximate robust formulations for
GPs and SPs. The methods are detailed at a high level below, in decreasing order of conservativeness.
Please see [Saab, 2018] for further details. The following descriptions have been
borrowed from [Ozturk, 2019].

Simple Conservative Approximation
---------------------------------

The simple conservative approximation maximizes each monomial term separately.

Linearized Perturbations
------------------------

The Linearized Perturbations formulation separates large posynomials
into decoupled posynomials, depending on the dependence of monomial terms.
It then robustifies these smaller posynomials using robust linear programming techniques.

Best Pairs
----------

The Best Pairs methodology separates large posynomials into decoupled
posynomials, just like Linearized Perturbations. However, it then solves an
inner-loop problem to find the least conservative combination of monomial pairs.


Work in progress...

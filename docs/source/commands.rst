More advanced commands
======================

*[Documentation work in progress...]*

**robust** has a variety of tools beyond the basics
to aid engineering design under uncertainty.

All of these methods have been implemented in the `robustSPpaper`_
code repository, which defines the models used in [Ozturk, 2019].

.. _robustSPpaper: https://github.com/1ozturkbe/robustSPpaper/tree/master/code

Choosing between RGP approximation methods and uncertainty sets
---------------------------------------------------------------

There are many possible ``**options`` to ``RobustModel``, but the primary function
of the options is to be able to choose between the different RGP approximation methods,
which have differing levels of conservativeness.
We prefer to define the three methods in a dict for ease access, and call ``RobustModel``, for example
for Best Pairs, as follows:

.. code-block:: python

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True}]
    method = methods[0] # Best Pairs
    robust_model = RobustModel(m, uncertainty_set, gamma=Gamma, twoTerm=method['twoTerm'],
                                   boyd=method['boyd'], simpleModel=method['simpleModel'])

For ``uncertainty_set``, ``'box'`` and ``'elliptical'`` are currently supported, and
define the :math:`\infty`- and 2-norms respectively.

Coming soon...

Simulating robust designs
-------------------------

How to generate samples of uncertain parameters...

How to perform a Monte Carlo based probability of failure analysis
based on constraint satisfaction, i.e. feasibility solves...

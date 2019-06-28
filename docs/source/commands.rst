More advanced commands
======================

**robust** has a variety of tools beyond the basics
to aid engineering design under uncertainty.

All of these methods have been implemented in the `robustSPpaper`_
code repository, which defines the models used in [Ozturk, 2019].

.. _robustSPpaper: https://github.com/1ozturkbe/robustSPpaper/tree/master/code

Choosing between RGP approximation methods
------------------------------------------

We prefer to define the methods in a dict

.. code-blocK:: python

    methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
               {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
               {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True}]

and choose the appropriate

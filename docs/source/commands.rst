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
of the options is to be able to choose between the different RGP approximation methods
defined in [Saab, 2018],
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

For ``uncertainty_set``, ``'box'`` and ``'ellipsoidal'`` are currently supported, and
define the :math:`\infty`- and 2-norms respectively.

Simulating robust designs
-------------------------

Robust optimization provides guarantees of constraint satisfaction
under the defined uncertainty sets, but it is possible that the real
distributions of uncertainty are better defined by probability distributions. Within ``robust``,
we have a framework to generate Monte Carlo (MC) simulations of the uncertain parameters
to estimate the probability of constraint violation (i.e. probability of failure [pof])
of models, as well as the expectation and standard deviation of the objective cost.
To generate samples of uncertain parameters and simulate a robust model's performance
over these samples, we use the ``simulate`` module of ``robust``.
Good implementations of ``robust.simulate`` are in
`robustSPpaper/SimPleAC_pof_simulate <https://github.com/1ozturkbe/robustSPpaper/blob/master/code/SimPleAC_pof_simulate.py>`_.

There is one additional step before GPkit models can be simulated. Every free variable
that is *fixed* during MC simulation must have a ``fix = True`` attribute, as following:

.. code-block:: python

        A         = Variable("A", "-", "aspect ratio", fix = True)

This allows for the optimizer to know that these variables (eg. aspect ratio in aircraft design),
once the optimization is complete, cannot change under simulation. Other variables, eg. state variables
such as speed and altitude of an aircraft, can change during MC simulation to fulfill constraints.

There are a daunting number of parameters for the functions in this module,
most of which have little or no effect to optimization and simulation outcomes,
but are necessary for the mathematics of RGPs.
This I would recommend overcoming by creating a parameter function such as the following,
which I have filled with some defaults:

.. code-block:: python

    def pof_parameters():
        model = SomeGPModelWithUncertainParams() # A GP model with uncertain parameters
        number_of_time_average_solves = 3 # number of solves for computing time cost, set to 1 if not important

        ## PARAMETERS THAT AFFECT NUMBER OF GP MODELS SAMPLED
        # Each combination of methods, uncertainty_sets and gammas generates a
        # new RGP model designed with these inputs, to be tested against MC samples.
        methods = [{'name': 'Best Pairs', 'twoTerm': True, 'boyd': False, 'simpleModel': False},
                   {'name': 'Linearized Perturbations', 'twoTerm': False, 'boyd': False, 'simpleModel': False},
                   {'name': 'Simple Conservative', 'twoTerm': False, 'boyd': False, 'simpleModel': True}
                   ]
        uncertainty_sets = ['box', 'ellipsoidal']
        nGammas = 11
        gammas = np.linspace(0, 1.0, nGammas)

        # LINEARIZATION PARAMETERS
        # These exist because RGPs/RSPs implement robust linear programs in their backend.
        # Defaults are great for engineering purposes.
        min_num_of_linear_sections = 3
        max_num_of_linear_sections = 99
        linearization_tolerance = 1e-4

        # PARALLELIZATION (will be available in future, please keep as False for now)
        parallel = False

        # SAMPLING PARAMETERS
        number_of_iterations = 100 # number of MC samples
        distribution = 'normal' # or 'uniform'

        # GENERATING SAMPLES (contained in directly_uncertain_vars_subs)
        nominal_solution, nominal_solve_time, nominal_number_of_constraints, directly_uncertain_vars_subs = \
            simulate.generate_model_properties(model, number_of_time_average_solves, number_of_iterations, distribution)

        verbosity = 1 # controls printouts for simulations

        return [model, methods, gammas, number_of_iterations,
        min_num_of_linear_sections, max_num_of_linear_sections, verbosity, linearization_tolerance,
        number_of_time_average_solves, uncertainty_sets, nominal_solution, directly_uncertain_vars_subs, parallel,
                nominal_number_of_constraints, nominal_solve_time]


Simulating a GPkit model is equivalent to optimizing the model over its remaining free variables (without
the 'fix' attribute). For Monte Carlo simulations, it is important to note that **the solution time is proportional
to the product of number of methods, uncertainty sets, gammas and samples**.
As such, one MC simulation usually takes similar to slightly less time to one
solution to the un-robustified model. Furthermore, MC simulations take *longer* for robustified
models vs. unrobust ones, since infeasibility can be detected faster than a feasible solution.
If in a time crunch, it is recommended that one method, set and gamma is chosen for simulation purposes.

Once the parameters are generated, the ``robust.simulate`` module can be used to generate MC data.

.. code-block:: python

    solutions, solve_times, simulation_results, number_of_constraints = simulate.variable_gamma_results(
                                             model, methods, gammas, number_of_iterations,
                                             min_num_of_linear_sections,
                                             max_num_of_linear_sections, verbosity, linearization_tolerance,
                                             number_of_time_average_solves,
                                             uncertainty_sets, nominal_solution, directly_uncertain_vars_subs, parallel=parallel)

It is highly recommended that you save/pickle the results, since MC simulations can have a large time cost.
`robustSPpaper/SimPleAC_save <https://github.com/1ozturkbe/robustSPpaper/blob/master/code/SimPleAC_save.py>`_
and other files in the ``simulate`` module
have simple demonstrations of saving/pickling.



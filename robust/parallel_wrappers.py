"""
This module contains functions and wrappers for running code in parallel.
"""
try:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    comm = MPI.COMM_WORLD
    has_mpi = True
except ImportError:
    print("mpi4py not loaded. Falling back on multiprocessing")
    has_mpi = False
    from multiprocessing import Pool, cpu_count
import copy
import math


def _wrapped_f(argstup):
    """
    Wrapper for the function to run in parallel.

    :param argstup: arguments
    :type argstup: tuple
    :return: function return
    """
    # if has_mpi:
    #     print("Rank %s/%s, node: %s" %(comm.rank, comm.size, MPI.Get_processor_name()))
    func = argstup[0]
    args = argstup[1]
    kwargs = argstup[2]
    return func(*args, **kwargs)


def parfor(func, dynamic_args=None, static_args=None, dynamic_kwargs=None, static_kwargs=None, num_processes=None, chunksize=16, maxtasksperchild=None):
    """
    Function for running a function multiple times with different inputs.

    The function is assumed to take the dynamic arguments before the static ones.

    If MPI is detected on the system, a :class:`mpi4py.futures.MPIPoolExecutor` computing pool will be launched.
    Otherwise, a :class:`multiprocessing.Pool` computing pool will be used.

    :param func: function to run
    :type func: function
    :param dynamic_args: list of argument tuples to be run
    :type dynamic_args: list of tuple
    :param static_args: tuple of arguments common to all runs
    :type static_args: tuple
    :param dynamic_kwargs: list of keyword arguments to be run
    :type dynamic_kwargs: list of dict
    :param static_kwargs: keyword arguments common to all runs
    :type static_kwargs: dict
    :param num_processes: number of processors/threads to launch
    :type num_processes: int
    :param chunksize: total number of runs can be broken into chunks of size chunksize for efficiency. This is only applicable if :class:`multiprocessing.Pool` is used.
    :type chunksize: int
    :param maxtasksperchild: maximum number of tasks to be run per child process. This is a :class:`multiprocessing.Pool` argument
    :type maxtasksperchild: int
    :return: list of results
    :rtype: list
    """
    if static_args is None:
        static_args = tuple()
    if static_kwargs is None:
        static_kwargs = dict()

    num_jobs = 0
    # todo: this logic seems bad fix
    if dynamic_kwargs is not None and dynamic_args is not None:
        assert len(dynamic_kwargs) == len(dynamic_args), "Lengths of dynamic inputs must be the same"
    if dynamic_args is None:
        dynamic_args = [tuple() for i in range(len(dynamic_kwargs))]
        num_jobs = len(dynamic_kwargs)
    if dynamic_kwargs is None:
        dynamic_kwargs = [dict() for i in range(len(dynamic_args))]
        num_jobs = len(dynamic_args)

    if has_mpi:
        if num_processes is None:
            num_processes = num_jobs
        print("Creating MPIPoolExecutor with %s workers" %num_processes)
        pool = MPIPoolExecutor(max_workers=num_processes, unordered=True)
        argslist = [(func, a+static_args, static_kwargs) for a in dynamic_args]
        for j in range(num_jobs):
            argslist[j][2].update(dynamic_kwargs[j])

        tmp = pool.map(_wrapped_f, argslist, chunksize=1)
        jobs = [r for r in tmp]

    else:
        if num_processes is None:
            num_processes = cpu_count()
        print("Initializing multiprocessing.Pool with %s workers, %s tasks/child" %(num_processes, maxtasksperchild))
        pool = Pool(processes=num_processes, maxtasksperchild=maxtasksperchild)

        num_chunks = int(math.ceil(float(num_jobs)/float(chunksize)))
        jobs = list()
        ct = 0
        for i in range(num_chunks):
            num_part = min(chunksize, num_jobs-ct)

            print("Copying arguments for parallel pool, chunk %s of %s, size %s" %(i+1, num_chunks, num_part))
            argslist = [(func, a+static_args, static_kwargs) for a in dynamic_args[ct:ct+num_part]]

            for j in range(num_part):
                argslist[j][2].update(dynamic_kwargs[ct+j])

            print("Running chunk")
            # result = pool.map_async(_wrapped_f, argslist)
            # jobs += result.get()
            jobs += pool.map(_wrapped_f, argslist)

            ct += num_part

    if has_mpi:
        pool.shutdown()
    else:
        pool.close()
        pool.join()

    return jobs

"""
This module contains functions and wrappers for running code in parallel.
"""

from builtins import range
from multiprocessing import Pool, cpu_count
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


def parfor(func, args = None, kwargs = None, num_processes=None, chunksize=16, maxtasksperchild=None):
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
    if args and kwargs:
        num_jobs = len(args)
        if num_jobs != len(kwargs):
            print("Warning: Different numbers of args and kwargs for function. ")
    elif args:
        num_jobs = len(args)
        kwargs = [None for _ in args]
    elif kwargs:
        num_jobs = len(kwargs)
        args = [None for _ in kwargs]

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
        argslist = [(func, args[i], kwargs[i]) for i in range(ct,ct+num_part)]
        print("Running chunk")
        jobs += pool.map(_wrapped_f, argslist)
        ct += num_part
    pool.close()
    pool.join()
    return jobs

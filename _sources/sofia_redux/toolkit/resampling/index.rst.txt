.. currentmodule:: sofia_redux.toolkit.resampling

.. _sofia_redux.toolkit_resampling:

***********************************************
Resampling (`sofia_redux.toolkit.resampling`)
***********************************************

Introduction
============

The :mod:`sofia_redux.toolkit.resampling` module allows the local fitting of polynomials
to N-dimensional data for resampling.

Samples (coordinates with associated value) are passed into the
:class:`Resample` class during initialization, and then resampled onto second
set of coordinates (points) during a call to the :class:`Resample` instance.
The :func:`resample` function acts as wrapper on :class:`Resample` to directly
resample from a set of samples to points.

Numba
=====

The `numba <https://numba.pydata.org/>`_ package is used heavily to achieve C
performance using JIT (just-in-time) compilation.  As a result, the initial
call to :class:`Resample` will introduce additional overhead while any JIT code
compiles.  Compilation results are stored in a cache for future use so that
this overhead burden is only encountered once.  For example::

    >>> import numpy as np
    >>> from sofia_redux.toolkit.resampling import Resample
    >>> import time
    >>> x = np.arange(10)
    >>> y = np.arange(10)
    >>> resampler = Resample(x, y)
    >>> t1 = time.time()
    >>> y2 = resampler(4.5)
    >>> t2 = time.time()
    >>> y2 = resampler(4.5)
    >>> t3 = time.time()
    >>> print("The 1st run-through took %.6f seconds" % (t2 - t1))  #doctest: +SKIP
    The 1st run-through took 0.018941 seconds
    >>> print("The 2nd run-through took %.6f seconds" % (t3 - t2))  #doctest: +SKIP
    The 2nd run-through took 0.001095 seconds


K-dimensional Polynomial Theory
===============================

.. toctree::
    :maxdepth: 2

    polynomial_theory.rst

Examples
========

.. toctree::
    :maxdepth: 2

    examples.rst

Irregular Kernel Resampling
===========================

.. toctree::
    :maxdepth: 2

    kernel_resampler.rst

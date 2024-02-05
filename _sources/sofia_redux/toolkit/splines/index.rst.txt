.. currentmodule:: sofia_redux.toolkit.resampling.splines

.. _sofia_redux.toolkit_splines:

**************************************************
Splines (`sofia_redux.toolkit.resampling.splines`)
**************************************************

Introduction
============

The :mod:`sofia_redux.toolkit.resampling.splines` module contains classes and
functions to represent and evaluate splines.  Algorithms are based on the
Fortran routines `DIERCKX <http://www.netlib.org/dierckx/>`_ by P. Dierckx
reimplemented to N-dimensions, and accelerated to C-like speeds through
use of the `numba <https://numba.pydata.org/>`_ package.

Theory
======
.. toctree::
    :maxdepth: 2

    spline_theory.rst


Examples
========
.. toctree::
    :maxdepth: 2

    spline_examples.rst

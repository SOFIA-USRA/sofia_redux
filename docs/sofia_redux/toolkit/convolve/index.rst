.. currentmodule:: sofia_redux.toolkit.convolve

***********************************************
Convolution  (`sofia_redux.toolkit.convolve`)
***********************************************

Introduction
============
The :mod:`sofia_redux.toolkit.convolve` module contains classes and functions to enable
N-dimensional convolution with various pre or user-defined kernels.  The
majority of work is handled by the :class:`ConvolveBase` class, a child of
:class:`sofia_redux.toolkit.utilities.base.Model`, which is also the parent of the
:class:`sofia_redux.toolkit.fitting.polynomial.Polyfit` class.  As such, many features
are shared across both classes.

Kernel Convolution
==================

.. toctree::
    :maxdepth: 2

    kernel.rst

Convolution Filters
===================

.. toctree::
    :maxdepth: 2

    filter.rst

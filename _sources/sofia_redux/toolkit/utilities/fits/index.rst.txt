.. currentmodule:: sofia_redux.toolkit.utilities.fits

*************
FITS Handling
*************

Introduction
============

The :mod:`sofia_redux.toolkit.utilities.fits` module provides a number of utilities to
read, write, and modify FITS files for use with SOFIA data pipeline products.
The `astropy.io.fits` module is used as the main interface for FITS handling
operations.  Generally, read and write procedures are designed to be as
robust as possible, and attempts will be made to fix corrupt data whenever
encountered.


FITS File Input/Output Functions
================================

.. toctree::
    :maxdepth: 2

    fitsio.rst


Header Operations
=================

.. toctree::
    :maxdepth: 2

    header.rst

****************************************************
sofia_redux.calibration: Flux Calibration Algorithms
****************************************************

The :mod:`sofia_redux.calibration` package contains data reduction algorithms for the
flux calibration of imaging instruments.  It is designed to be used with
SOFIA pipelines, integrated into the `sofia_redux` package.  Currently,
the FORCAST, FLITECAM, and HAWC+ instruments are supported.

Getting Started
===============

Using Calibration Utilities
---------------------------

The primary calibration functions provided by this package are:

   1. Apply a correction for atmospheric opacity
   2. Compute aperture photometry and calibration parameters on
      known flux standards
   3. Apply flux calibration to an image

The easiest access to these functions are provided in the
`sofia_redux.calibration.pipecal_util` submodule, as `apply_tellcor`, `run_photometry`,
and `apply_fluxcal` respectively.  A standalone script, called
`sofia_redux.calibration.pipecal_applyphot` is also available to directly calculate
photometry and store the results in FITS headers.

Submodule
=========

.. toctree::
  :maxdepth: 1

  standard_model/index.rst


Reference/API
=============

.. automodapi:: sofia_redux.calibration.pipecal_applyphot
.. automodapi:: sofia_redux.calibration.pipecal_calfac
.. automodapi:: sofia_redux.calibration.pipecal_config
.. automodapi:: sofia_redux.calibration.pipecal_fitpeak
.. automodapi:: sofia_redux.calibration.pipecal_photometry
.. automodapi:: sofia_redux.calibration.pipecal_rratio
.. automodapi:: sofia_redux.calibration.pipecal_util
.. automodapi:: sofia_redux.calibration.pipecal_error
   :no-inheritance-diagram:


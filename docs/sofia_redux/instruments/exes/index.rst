************************************************************
sofia_redux.instruments.exes: EXES Data Reduction Algorithms
************************************************************

The :mod:`sofia_redux.instruments.exes` package contains data reduction
algorithms for the EXES instrument.  It is designed to be used with the
`sofia_redux` package, so it does not provide its own interfaces or workflows.
See the `sofia_redux.pipeline` documentation for more information on the pipeline
interfaces, or the API documentation below for more information on EXES
algorithms.

Practical Data Reduction for EXES
=================================

.. toctree::
   :maxdepth: 1

   exes_reduction_guide


Flat Procedure
==============

Flat fields for EXES are used to correct for instrumental response,
calibrate to physical units, and identify or tune optical distortion
parameters.

.. toctree::
  :maxdepth: 1

  flat


Reference/API
=============

.. automodapi:: sofia_redux.instruments.exes.calibrate
.. automodapi:: sofia_redux.instruments.exes.cirrus
.. automodapi:: sofia_redux.instruments.exes.clean
.. automodapi:: sofia_redux.instruments.exes.coadd
.. automodapi:: sofia_redux.instruments.exes.correct_row_gains
.. automodapi:: sofia_redux.instruments.exes.debounce
.. automodapi:: sofia_redux.instruments.exes.derasterize
.. automodapi:: sofia_redux.instruments.exes.derive_tort
.. automodapi:: sofia_redux.instruments.exes.despike
.. automodapi:: sofia_redux.instruments.exes.diff_arr
.. automodapi:: sofia_redux.instruments.exes.get_atran
.. automodapi:: sofia_redux.instruments.exes.get_badpix
.. automodapi:: sofia_redux.instruments.exes.get_resolution
.. automodapi:: sofia_redux.instruments.exes.lincor
.. automodapi:: sofia_redux.instruments.exes.make_template
.. automodapi:: sofia_redux.instruments.exes.makeflat
.. automodapi:: sofia_redux.instruments.exes.mergehdr
.. automodapi:: sofia_redux.instruments.exes.readhdr
.. automodapi:: sofia_redux.instruments.exes.readraw
.. automodapi:: sofia_redux.instruments.exes.spatial_shift
.. automodapi:: sofia_redux.instruments.exes.submean
.. automodapi:: sofia_redux.instruments.exes.tort
.. automodapi:: sofia_redux.instruments.exes.tortcoord
.. automodapi:: sofia_redux.instruments.exes.utils
.. automodapi:: sofia_redux.instruments.exes.wavecal

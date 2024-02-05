******************************************************************
sofia_redux.instruments.fifi_ls: FIFI-LS Data Reduction Algorithms
******************************************************************

The :mod:`sofia_redux.instruments.fifi_ls` package contains data reduction
algorithms for the FIFI-LS instrument.  It is designed to be used with the
`sofia_redux` package, so it does not provide its own interfaces or workflows.
Basic usage via the Redux interface is described below.  See the
`sofia_redux.pipeline` documentation for more information on the pipeline
interfaces, or the API documentation below for more information on
FIFI-LS algorithms.


Reference/API
=============
.. automodapi:: sofia_redux.instruments.fifi_ls
   :skip: UnsupportedPythonError
   :skip: test
.. automodapi:: sofia_redux.instruments.fifi_ls.apply_static_flat
.. automodapi:: sofia_redux.instruments.fifi_ls.combine_grating_scans
.. automodapi:: sofia_redux.instruments.fifi_ls.combine_nods
.. automodapi:: sofia_redux.instruments.fifi_ls.correct_wave_shift
.. automodapi:: sofia_redux.instruments.fifi_ls.fit_ramps
.. automodapi:: sofia_redux.instruments.fifi_ls.flux_calibrate
.. automodapi:: sofia_redux.instruments.fifi_ls.get_atran
.. automodapi:: sofia_redux.instruments.fifi_ls.get_badpix
.. automodapi:: sofia_redux.instruments.fifi_ls.get_resolution
.. automodapi:: sofia_redux.instruments.fifi_ls.get_response
.. automodapi:: sofia_redux.instruments.fifi_ls.lambda_calibrate
.. automodapi:: sofia_redux.instruments.fifi_ls.make_header
.. automodapi:: sofia_redux.instruments.fifi_ls.readfits
.. automodapi:: sofia_redux.instruments.fifi_ls.resample
.. automodapi:: sofia_redux.instruments.fifi_ls.spatial_calibrate
.. automodapi:: sofia_redux.instruments.fifi_ls.split_grating_and_chop
.. automodapi:: sofia_redux.instruments.fifi_ls.subtract_chops
.. automodapi:: sofia_redux.instruments.fifi_ls.telluric_correct

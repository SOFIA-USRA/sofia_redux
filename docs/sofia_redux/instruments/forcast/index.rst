******************************************************************
sofia_redux.instruments.forcast: FORCAST Data Reduction Algorithms
******************************************************************

The :mod:`sofia_redux.instruments.forcast` package contains data reduction
algorithms for the FORCAST instrument.  It is designed to be used with the
`sofia_redux` package, so it does not provide its own interfaces or workflows.
See the `sofia_redux.pipeline` documentation for more information on the pipeline
interfaces, or the API documentation below for more information on FORCAST
algorithms.


Reference/API
=============

.. automodapi:: sofia_redux.instruments.forcast.background
.. automodapi:: sofia_redux.instruments.forcast.calcvar
.. automodapi:: sofia_redux.instruments.forcast.check_readout_shift
.. automodapi:: sofia_redux.instruments.forcast.chopnod_properties
.. automodapi:: sofia_redux.instruments.forcast.clean
.. automodapi:: sofia_redux.instruments.forcast.configuration
.. automodapi:: sofia_redux.instruments.forcast.distcorr_model
.. automodapi:: sofia_redux.instruments.forcast.droop
.. automodapi:: sofia_redux.instruments.forcast.getatran
.. automodapi:: sofia_redux.instruments.forcast.getcalpath
.. automodapi:: sofia_redux.instruments.forcast.getdetchan
.. automodapi:: sofia_redux.instruments.forcast.getmodel
.. automodapi:: sofia_redux.instruments.forcast.getpar
.. automodapi:: sofia_redux.instruments.forcast.hdcheck
.. automodapi:: sofia_redux.instruments.forcast.hdmerge
.. automodapi:: sofia_redux.instruments.forcast.hdrequirements
.. automodapi:: sofia_redux.instruments.forcast.imgnonlin
.. automodapi:: sofia_redux.instruments.forcast.imgshift_header
.. automodapi:: sofia_redux.instruments.forcast.jbclean
.. automodapi:: sofia_redux.instruments.forcast.merge
.. automodapi:: sofia_redux.instruments.forcast.merge_centroid
.. automodapi:: sofia_redux.instruments.forcast.merge_correlation
.. automodapi:: sofia_redux.instruments.forcast.merge_shift
.. automodapi:: sofia_redux.instruments.forcast.peakfind
   :no-inheritance-diagram:
.. automodapi:: sofia_redux.instruments.forcast.read_section
.. automodapi:: sofia_redux.instruments.forcast.readfits
.. automodapi:: sofia_redux.instruments.forcast.readmode
.. automodapi:: sofia_redux.instruments.forcast.register
.. automodapi:: sofia_redux.instruments.forcast.register_datasets
.. automodapi:: sofia_redux.instruments.forcast.rotate
.. automodapi:: sofia_redux.instruments.forcast.shift
.. automodapi:: sofia_redux.instruments.forcast.stack
.. automodapi:: sofia_redux.instruments.forcast.undistort

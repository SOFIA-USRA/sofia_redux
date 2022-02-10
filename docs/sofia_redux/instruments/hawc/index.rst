*************************************************************
sofia_redux.instruments.hawc: HAWC+ Data Reduction Algorithms
*************************************************************

The :mod:`sofia_redux.instruments.hawc` package contains data reduction
algorithms for the HAWC+ instrument.  It is designed to be used with the
`sofia_redux` package, so it does not provide its own interfaces or workflows.
See the `sofia_redux.pipeline` documentation for more information on the
pipeline interfaces, or the API documentation below for more information on
HAWC algorithms.


Reference/API
=============

Core Classes
------------

.. automodapi:: sofia_redux.instruments.hawc
   :headings: ~^

Pipeline Steps
--------------

.. automodapi:: sofia_redux.instruments.hawc.steps
   :headings: ~^

Pipeline Utilities
------------------

.. automodapi:: sofia_redux.instruments.hawc.steps.basehawc
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.basemap
   :headings: ~^

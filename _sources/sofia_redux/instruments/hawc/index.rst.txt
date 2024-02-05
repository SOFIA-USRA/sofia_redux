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

.. automodapi:: sofia_redux.instruments.hawc.datafits
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.dataparent
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.datatext
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steploadaux
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.stepmiparent
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.stepmoparent
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.stepparent
   :headings: ~^


Pipeline Steps
--------------

.. automodapi:: sofia_redux.instruments.hawc.steps.stepbgsubtract
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepbinpixels
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepcalibrate
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepcheckhead
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepcombine
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepdemodulate
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepdmdcut
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepdmdplot
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepflat
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepfluxjump
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepfocus
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepimgmap
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepip
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.steplabchop
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.steplabpolplots
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepmerge
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepmkflat
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepnodpolsub
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepnoisefft
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepnoiseplots
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepopacity
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.steppoldip
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.steppolmap
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.steppolvec
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepprepare
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepregion
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.steprotate
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepscanmap
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepscanmapflat
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepscanmapfocus
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepscanmappol
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepscanstokes
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepshift
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepskycal
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepskydip
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepsplit
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepstdphotcal
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepstokes
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepwcs
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.stepzerolevel
   :headings: ~^

Pipeline Utilities
------------------

.. automodapi:: sofia_redux.instruments.hawc.steps.basehawc
   :headings: ~^
.. automodapi:: sofia_redux.instruments.hawc.steps.basemap
   :headings: ~^

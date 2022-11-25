********************************************************
sofia_redux.pipeline: Data Reduction Pipelines for SOFIA
********************************************************

The :mod:`sofia_redux.pipeline` package is a data reduction interface for
astronomy pipelines. It provides interactive and command-line interfaces to
astronomical algorithms.

This package was developed to support data reduction pipelines for the SOFIA
telescope, but is designed to be a general-purpose interface package to any
arbitrary series of data reduction steps.

Getting Started
===============

Running SOFIA pipelines
-----------------------
Out of the box, Redux provides two command-line scripts:

* **redux** (`sofia_redux.pipeline.sofia.redux_app`): An interactive graphical interface (GUI) for the SOFIA pipelines.
* **redux_pipe** (`sofia_redux.pipeline.sofia.redux_pipe`): A command-line interface to the SOFIA pipelines.

In general, Redux works by reading in input data, deciding which reduction to run, then running
a pre-defined set of reduction steps.

The GUI allows interactive parameter editing, and intermediate product display for each step.
To begin a reduction, start the GUI by typing ``redux``, then load in a set of data with
the File->Open New Reduction menu.

The command-line interface allows fully automatic pipeline reductions.  To begin a reduction
with the automatic pipeline, type ``redux_pipe`` and provide the file names of the data
to reduce on the command line.  Non-default parameters can also be provided by specifying
a configuration file in INI format.

Displaying FITS data
--------------------
This package also provides a stand-alone front end to some of its display tools
(`sofia_redux.pipeline.gui.qad`).  The command line script **qad** (`sofia_redux.pipeline.gui.qad.qad_app`) starts a
small GUI that allows interactive display of FITS images and spectra in DS9 and
a native spectral viewer (`sofia_redux.visualization.eye`). To use it, DS9
must be installed (see http://ds9.si.edu/), and the ``ds9`` executable
must be available in the PATH environment variable.

To use the QAD, start up the GUI by typing ``qad`` at the command line,
then double-click on a FITS image to display it.  Display preferences can be
set from the 'Settings' menu

Developing new pipelines
------------------------
Redux is designed to be a development platform, for running any sequence of data
reduction steps.  To use Redux to develop a new pipeline, the following are necessary:

* A reduction class that inherits from the `sofia_redux.pipeline.Reduction` class.
  This class defines all data reduction steps, and the order in which they should run.
* A parameter class that inherits from the `sofia_redux.pipeline.Parameters` class.
  This class should provide default parameters for the steps defined in the reduction
  object.
* A reduction object chooser class that inherits from the `sofia_redux.pipeline.Chooser` class.
  The chooser decides from input data which reduction object to instantiate.
* A configuration class that inherits from `sofia_redux.pipeline.Configuration`.  This
  class provides default parameters for the interface, such as default log file names,
  and specifies the reduction chooser.
* A pipe script that instantiates a `sofia_redux.pipeline.Pipe` object with the appropriate
  configuration.
* An application script that instantiates a `sofia_redux.pipeline.Application` object with the
  appropriate configuration.

Optionally, custom viewers may also be defined for displaying intermediate data
products.  These should inherit from the `sofia_redux.pipeline.Viewer` class.

See the `sofia_redux.pipeline.sofia` module for examples of all these classes and scripts.


Architecture
============

.. include:: redux_architecture.rst

Usage
=====

.. include:: redux_usage.rst

Reference/API
=============

Redux Core
----------
.. automodapi:: sofia_redux.pipeline.application
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.chooser
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.configuration
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.interface
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.pipe
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.viewer
   :headings: ~^

Redux GUI
---------
.. automodapi:: sofia_redux.pipeline.gui.main
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.matplotlib_viewer
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.qad_viewer
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.textview
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.widgets
   :headings: ~^

QAD Viewer
----------
.. automodapi:: sofia_redux.pipeline.gui.qad.qad_app
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.qad.qad_dialogs
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.qad.qad_headview
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.qad.qad_imview
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.gui.qad.qad_main_panel
   :headings: ~^

SOFIA Redux
-----------
.. automodapi:: sofia_redux.pipeline.sofia.sofia_app
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.sofia_chooser
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.sofia_configuration
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.sofia_pipe
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.sofia_utilities
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.exes_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.fifils_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.flitecam_imaging_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.flitecam_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.flitecam_slitcorr_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.flitecam_spatcal_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.flitecam_wavecal_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.forcast_imaging_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.forcast_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.forcast_slitcorr_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.forcast_spatcal_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.forcast_wavecal_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.hawc_reduction
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.exes_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.fifils_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.flitecam_imaging_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.flitecam_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.flitecam_slitcorr_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.flitecam_spatcal_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.flitecam_spectroscopy_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.flitecam_wavecal_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.forcast_imaging_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.forcast_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.forcast_slitcorr_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.forcast_spatcal_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.forcast_spectroscopy_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.forcast_wavecal_parameters
   :headings: ~^
.. automodapi:: sofia_redux.pipeline.sofia.parameters.hawc_parameters
   :headings: ~^

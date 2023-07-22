***********
SOFIA Redux
***********

Introduction
============

SOFIA Redux (`sofia_redux`) contains data processing pipelines and algorithms
for instruments on the Stratospheric Observatory for Infrared Astronomy
(SOFIA).

SOFIA raw and processed data can be accessed from the
`SOFIA archive <https://irsa.ipac.caltech.edu/applications/sofia/>`__.
Archived data may not match the results of data processed
with this pipeline software.  Questions specific to particular data sets
should be directed to the `SOFIA helpdesk <sofia_help@sofia.usra.edu>`__.

SOFIA pipelines are developed internally by the USRA/SOFIA data processing
software team, then are published publicly at the
`SOFIA Redux GitHub project
<https://github.com/SOFIA-USRA/sofia_redux>`__.
Contributions and feedback are welcome via the GitHub project, but
merge requests cannot be directly accepted.  They will be internally reviewed,
and pushed to the public site as needed.

Getting Started
===============

.. include:: ../install.rst


Running SOFIA pipelines
-----------------------
Out of the box, Redux provides two primary command-line scripts:

* **redux** (`sofia_redux.pipeline.sofia.redux_app`): An interactive graphical interface (GUI) for the SOFIA pipelines.
* **redux_pipe** (`sofia_redux.pipeline.sofia.redux_pipe`): A command-line interface to the SOFIA pipelines.

Currently, the EXES, FIFI-LS, FLITECAM, FORCAST, and HAWC+ instruments are supported by
these pipelines.  See the user's and developer's manuals for
each instrument for full descriptions of all supported observing
modes and scientific algorithms.

In general, Redux works by reading in input data, deciding which reduction to run, then running
a pre-defined set of reduction steps.

The GUI allows interactive parameter editing, and intermediate product display for each step.
To begin a reduction, start the GUI by typing ``redux``, then load in a set of data with
the File->Open New Reduction menu.

The command-line interface allows fully automatic pipeline reductions.  To begin a reduction
with the automatic pipeline, type ``redux_pipe`` and provide the file names of the data
to reduce on the command line.  Non-default parameters can also be provided by specifying
a configuration file in INI format.  Type ``redux_pipe -h`` for a brief
help message.

Tutorials
=========

Tutorials are available for data reduction procedures, via the
`SOFIA website <https://www.sofia.usra.edu/science/data/data-pipelines>`__:

- `FIFI-LS <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FIFI-LS_tutorial.pdf>`__
- `FLITECAM Imaging <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FLITECAM_imaging_tutorial.pdf>`__
- `FLITECAM Spectroscopy <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FLITECAM_spectroscopy_tutorial.pdf>`__
- `FORCAST Imaging <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FORCAST_imaging_tutorial.pdf>`__
- `FORCAST Spectroscopy <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FORCAST_spectroscopy_tutorial.pdf>`__
- `HAWC+ Imaging <https://www.sofia.usra.edu/sites/default/files/2023-07/hawc_imaging_tutorial.pdf>`__
- `HAWC+ Polarimetry <https://www.sofia.usra.edu/sites/default/files/2023-07/hawc_polarimetry_tutorial.pdf>`__
- EXES Spectroscopy: TBD

Manuals
=======

User's Manuals:

.. toctree::
   :maxdepth: 1

   ../manuals/exes/users/users
   ../manuals/fifils/users/users
   ../manuals/flitecam/users/users
   ../manuals/forcast/users/users
   ../manuals/hawc/users/users

Developer's Manuals:

.. toctree::
   :maxdepth: 1

   ../manuals/exes/developers/developers
   ../manuals/fifils/developers/developers
   ../manuals/flitecam/developers/developers
   ../manuals/forcast/developers/developers
   ../manuals/hawc/developers/developers


Submodules
==========

.. toctree::
  :maxdepth: 2

  pipeline/index.rst
  calibration/index.rst
  scan/index.rst
  spectroscopy/index.rst
  toolkit/index.rst
  visualization/index.rst
  instruments/index.rst

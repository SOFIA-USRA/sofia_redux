***********
SOFIA Redux
***********

Introduction
============

SOFIA Redux (`sofia_redux`) contains data processing pipelines and algorithms
for instruments on the Stratospheric Observatory for Infrared Astronomy
(SOFIA).

SOFIA raw and processed data can be accessed from the
`SOFIA archive <https://irsa.ipac.caltech.edu/applications/sofia/>`_.
Archived data may not match the results of data processed
with this pipeline software.  Questions specific to particular data sets
should be directed to the `SOFIA helpdesk <sofia_help@sofia.usra.edu>`_.

SOFIA pipelines are developed internally by the USRA/SOFIA data processing
software team, then are published publicly at the
`SOFIA Redux GitHub project
<https://github.com/SOFIA-USRA/sofia_redux>`_.
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

Currently, the FIFI-LS and FORCAST instruments are supported by these pipelines.
The HAWC+ pipeline is still under development and will be added to this
package when it is available.  See the user's and developer's manuals for
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
`SOFIA website <https://www.sofia.usra.edu/science/data/data-pipelines>`_:

- `FIFI-LS <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FIFI-LS_tutorial.pdf>`_
- `FORCAST Imaging <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FORCAST_imaging_tutorial.pdf>`_
- `FORCAST Spectroscopy <https://www.sofia.usra.edu/sites/default/files/USpot_DCS_DPS/Documents/FORCAST_spectroscopy_tutorial.pdf>`_

Manuals
=======

.. toctree::
   :maxdepth: 1

   ../manuals/fifils/users/users
   ../manuals/fifils/developers/developers
   ../manuals/forcast/users/users
   ../manuals/forcast/developers/developers


Submodules
==========

.. toctree::
  :maxdepth: 2

  pipeline/index.rst
  calibration/index.rst
  spectroscopy/index.rst
  toolkit/index.rst
  visualization/index.rst
  instruments/index.rst

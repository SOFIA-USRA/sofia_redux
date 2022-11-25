********************************
FIFI-LS Redux Developer’s Manual
********************************

.. raw:: latex

    \clearpage


Introduction
============

Document Purpose
----------------

This document is intended to provide all the information necessary to
maintain the FIFI-LS Redux pipeline, used to produce
Level 2, 3, and 4 reduced products for FIFI-LS data, in either manual or
automatic mode. Level 2 is defined as data that has been processed to
correct for instrumental effects; Level 3 is defined as data that has
been flux-calibrated; Level 4 is any higher data product. A more general
introduction to the data reduction procedure and the scientific justification
of the algorithms is available in the FIFI-LS Redux User's Manual.

This manual applies to FIFI-LS Redux version 2.7.0.


Redux Revision History
----------------------

FIFI-LS Redux was developed as four separate reduction packages: PyFIFI,
which provides the data processing algorithms; PypeUtils,
which provides general-purpose scientific algorithms; PySpextool, which
provides some supporting libraries for spectroscopic data reduction; and Redux,
which provides the interactive GUI, the automatic pipeline wrapper,
and the supporting structure to call the FIFI-LS algorithms.

PyFIFI is a Python translation of an IDL package developed for the reduction of
FIFI-LS data. Dr. Kaori Nishikida and Dr. Randolf Klein initially
developed a prototype of the algorithms, around 2004. Development in IDL was
picked up in February 2015 by Jennifer Holt and Dr. William Vacca for
USRA/SOFIA, with reference to a separate FIFI-LS pipeline, developed in
LabView by Rainer Hoenle. Integration into the Redux interface
and completion of the reduction algorithms was undertaken by
Melanie Clarke and Dr. William Vacca in September 2015. Version 1.0.0 of
the package was released for use at SOFIA in November 2015. Version 2.0.0 of
the package was entirely reimplemented in Python, primarily by Daniel Perera,
with support and integration by Melanie Clarke.

PypeUtils was developed as a shared code base for SOFIA Python pipelines,
primarily by Daniel Perera for USRA/SOFIA.  It contains any algorithms
or utilities likely to be of general use for data reduction pipelines.
From this package, the FIFI-LS pipeline uses some FITS handling utilities,
multiprocessing tools, and interpolation and resampling functions.

Like PyFIFI, PySpextool is a translation of an earlier SOFIA IDL library,
called FSpextool.  FSpextool was built on top of a pre-release version of
Spextool 4, an IDL-based package developed by Dr. Michael Cushing and Dr.
William Vacca for the reduction of data from the SpeX instrument on the
NASA Infrared Telescope Facility (IRTF). Spextool was originally released in
October 2000, and has undergone a number of major and minor revisions since
then. The last stable public release was v4.1, released January 2016. As
Spextool does not natively support automatic command-line processing,
FSpextool for SOFIA adapted the Spextool library to the SOFIA architecture
and instruments; version 1.0.0 was originally released for use at SOFIA
in July 2013. PySpextool is a Python translation of the core algorithms in
this package, developed by Daniel Perera and Melanie Clarke, and first
released at SOFIA for use in the FIFI-LS pipeline in October 2019.

Redux was originally developed to be a general-purpose interface to IDL data
reduction algorithms. It provided an interactive GUI and an
object-oriented structure for calling data reduction processes, but it
did not provide its own data reduction algorithms. It was developed by
Melanie Clarke, for the SOFIA DPS team, to provide a consistent
front-end to the data reduction pipelines for multiple instruments and
modes, including FIFI-LS.  It was redesigned and reimplemented in Python,
with similar functionality, to support Python pipelines for SOFIA.  The
first release of the IDL version was in December 2013; the Python version
was first released to support the HAWC+ pipeline in September 2018.

In 2020, all SOFIA pipeline packages were unified into a single package,
called `sofia_redux`.  The interface package (Redux) was renamed to
`sofia_redux.pipeline`, PypeUtils was renamed to `sofia_redux.toolkit`,
PySpextool was renamed to `sofia_redux.spectroscopy`, and the PyFIFI
package was renamed to `sofia_redux.instruments.fifi_ls`.  An additional
package, to support data visualization, was added as
`sofia_redux.visualization`.

In 2021 and 2022, additional optional features from the `sofia_redux`
package were incorporated into the FIFI-LS pipeline.  The interface provides
some interactive photometry routines via `sofia_redux.calibration` and
some additional support for on-the-fly mode data via `sofia_redux.scan`.

Overview of Software Structure
==============================

The sofia_redux package has several sub-modules organized by functionality::

    sofia_redux
    ├── calibration
    ├── instruments
    │   ├── exes
    │   ├── fifi_ls
    │   ├── flitecam
    │   ├── forcast
    │   └── hawc
    ├── pipeline
    ├── scan
    ├── spectroscopy
    ├── toolkit
    └── visualization

The modules used in the FIFI-LS pipeline are described below.


sofia_redux.instruments.fifi_ls
-------------------------------
The `sofia_redux.instruments.fifi_ls` package is written in Python using
standard scientific tools and libraries.

The data reduction algorithms used by
the pipeline are straight-forward functions that generally take an
input file name, corresponding to a FIFI-LS FITS file, as an argument
and write an output file to disk as a result.  The return value in this
case is the output file name.  Optionally, the data may be supplied and
returned as an `astropy.io.fits.HDUList` data structure.

Other optional parameters for these functions are provided via keyword
parameters in the function calls.

sofia_redux.spectroscopy
------------------------

The `sofia_redux.spectroscopy` package contains a library of general-purpose
spectroscopic functions.  The FIFI-LS pipeline uses a few of the algorithms
from this library, for spectroscopic smoothing and binning.

sofia_redux.toolkit
-------------------

`sofia_redux.toolkit` is a repository for classes and functions of general usefulness,
intended to support multiple SOFIA pipelines.  It contains several submodules,
for interpolation, multiprocessing support, numerical calculations, and
FITS handling.  Most utilities are simple functions that take input
as arguments and return output values.  Some more complicated functionality
is implemented in several related classes; see the `sofia_redux.toolkit.resampling`
module documentation for more information.

sofia_redux.visualization
-------------------------

The `sofia_redux.visualization` package contains plotting and display
routines, relating to visualizing SOFIA data.  For the FIFI-LS pipeline,
this package currently provides a module that supports generating
quick-look preview images.


sofia_redux.calibration
-----------------------

The `sofia_redux.calibration` module contains flux calibration algorithms
used to perform photometric or flux calibration calculations on
input images and return their results.  For the FIFI-LS pipeline,
this package currently provides support for interactive photometry tools
in the pipeline interface.

sofia_redux.scan
----------------

The scan package (`sofia_redux.scan`) package implements
an iterative map reconstruction algorithm, for reducing continuously
scanned observations.  In the FIFI-LS pipeline, it is used to provide
optional support for removing residual correlated gain and noise in
on-the-fly (OTF) mode observations.

For more information on the design and structure of the scan package, see the
`HAWC+ pipeline developer's manual <https://sofia-usra.github.io/sofia_redux/manuals/hawc/developers/developers.html#scan-map-architecture>`__
and the software documentation for the
`sofia_redux.scan module <https://sofia-usra.github.io/sofia_redux/sofia_redux/scan/index.html>`__.


sofia_redux.pipeline
--------------------

Design
~~~~~~

.. include:: ../../../sofia_redux/pipeline/redux_architecture.rst


FIFI-LS Redux
~~~~~~~~~~~~~

To interface to the FIFI-LS pipeline algorithms, Redux defines
the `FIFILSReduction` and `FIFILSParameters` classes.
See :numref:`fifi_redux_class` for a sketch of
the Redux classes used by the FIFI-LS pipeline.  The FIFILSReduction class
calls the `sofia_redux.instruments.fifi_ls` reduction functions, with support
from the `sofia_redux.toolkit` and `sofia_redux.spectroscopy packages`.
The FIFILSParameters class defines default parameter values for all reduction
steps.

Since the FIFI-LS reduction algorithms are simple functions, the
FIFILSReduction class provides a wrapper method for each step in the pipeline:

    - Check Headers: calls `sofia_redux.instruments.fifi_ls.make_header`
    - Split Grating/Chop: calls `sofia_redux.instruments.fifi_ls.split_grating_and_chop`
    - Fit Ramps: calls `sofia_redux.instruments.fifi_ls.fit_ramps`
    - Subtract Chops: calls `sofia_redux.instruments.fifi_ls.subtract_chops`
    - Combine Nods: calls `sofia_redux.instruments.fifi_ls.combine_nods`
    - Lambda Calibrate: calls `sofia_redux.instruments.fifi_ls.lambda_calibrate`
    - Spatial Calibrate: calls `sofia_redux.instruments.fifi_ls.spatial_calibrate`
    - Apply Flat: calls `sofia_redux.instruments.fifi_ls.apply_static_flat`
    - Combine Scans: calls `sofia_redux.instruments.fifi_ls.combine_grating_scans`
    - Telluric Correct: calls `sofia_redux.instruments.fifi_ls.telluric_correct`
    - Flux Calibrate: calls `sofia_redux.instruments.fifi_ls.flux_calibrate`
    - Correct Wave Shift: calls `sofia_redux.instruments.fifi_ls.correct_wave_shift`
    - Resample: calls `sofia_redux.instruments.fifi_ls.resample`
    - Make Spectral Map: calls `sofia_redux.visualization.quicklook.make_image`

The recipe attribute for the reduction class specifies the above steps,
in that order.  Most FIFI-LS steps provide a wrapper function that
allows processing to proceed in parallel, if desired, when called on a
list of input files.  The FIFILSReduction interface uses these wrapper functions
where appropriate, and the FIFILSParameters class defines the defaults
for the parallel processing capabilities.

If an intermediate file is loaded, its product type is
identified from the PRODTYPE keyword in its header, and the prodtype_map
attribute is used to identify the next step in the recipe.  This
allows reductions to be picked up at any point, from a saved intermediate
file.  For more information on the scientific goals and methods used in
each step, see the FIFI-LS Redux User's Manual.

The FIFILSReduction class also contains several helper functions, that
assist in reading and writing files on disk, and identifying which
data to display in the interactive GUI.  Display is performed via
the `QADViewer` class provided by the Redux package.


.. index::
   single: FIFI-LS Redux classes (diagram)

.. figure:: images/redux_classes.png
   :alt: FIFI-LS Redux Classes
   :name: fifi_redux_class

   Redux classes used in the FIFI-LS pipeline.



Detailed Algorithm Information
==============================

The following sections list detailed information on the functions and
procedures most likely to be of interest to the developer.


sofia_redux.instruments.fifi_ls
-------------------------------

.. automodapi:: sofia_redux.instruments.fifi_ls.apply_static_flat
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.combine_grating_scans
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.combine_nods
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.correct_wave_shift
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.fit_ramps
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.flux_calibrate
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.get_atran
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.get_badpix
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.get_lines
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.get_resolution
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.get_response
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.lambda_calibrate
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.make_header
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.readfits
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.resample
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.spatial_calibrate
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.split_grating_and_chop
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.subtract_chops
   :headings: ~^
.. automodapi:: sofia_redux.instruments.fifi_ls.telluric_correct
   :headings: ~^

sofia_redux.toolkit
-------------------

.. automodapi:: sofia_redux.toolkit.utilities.fits
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.utilities.multiprocessing
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.resampling
   :headings: ~^
   :no-inheritance-diagram:

sofia_redux.spectroscopy
------------------------

.. automodapi:: sofia_redux.spectroscopy.earthvelocity
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.radvel
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.smoothres
   :headings: ~^

sofia_redux.calibration
-----------------------

.. automodapi:: sofia_redux.calibration.pipecal_photometry
   :headings: ~^

sofia_redux.scan
----------------

.. automodapi:: sofia_redux.scan.reduction.reduction
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.fifi_ls.channels.channels
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.channels.channel_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.channels.channel_data.channel_data
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.channels.channel_group.channel_group
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.frames.frames
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.frames.fifi_ls_frame_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.info.info
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.info.astrometry
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.info.detector_array
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.info.instrument
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.info.telescope
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.integration.integration
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.integration.fifi_ls_integration_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.fifi_ls.scan.scan
   :headings: ~^


sofia_redux.visualization
-------------------------

.. automodapi:: sofia_redux.visualization.quicklook
   :headings: ~^

sofia_redux.pipeline
--------------------

The Redux application programming interface (API), including the FIFI-LS
interface classes, are documented in the `sofia_redux.pipeline` package.

.. toctree::

   redux_doc


Appendix A: Pipeline Recipe
===========================

This JSON document is the black-box interface specification for the
FIFI-LS Redux pipeline, as defined in the Pipetools-Pipeline ICD.


.. include:: include/fifi_ls_recipe.json
   :literal:

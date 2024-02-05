*****************************
EXES Redux Developer's Manual
*****************************

.. raw:: latex

    \clearpage


Introduction
============

Document Purpose
----------------

This document is intended to provide all the information necessary to
maintain the EXES Redux pipeline, used to produce
Level 2 and 3 reduced products for EXES data, in
either manual or automatic mode. Level 2 is defined as data that has
been processed to correct for instrumental effects; Level 3 is defined
as flux-calibrated data. A more general introduction to the data
reduction procedure and the scientific justification of the algorithms
is available in the EXES Redux User's Manual.

This manual applies to EXES Redux version 3.0.0.

Redux Revision History
----------------------

The EXES pipeline was originally developed as three separate packages: the
EXES, which provided the image processing algorithms; FSpextool,
which provided the spectral extraction algorithms and some supporting
libraries; and Redux, which provided the interactive GUI, the automatic
pipeline wrapper, and the supporting structure to call EXES and FSpextool
algorithms.

EXES was a software package developed in IDL for the reduction of EXES
data. Dr. John Lacy initially developed the algorithms in FORTRAN, for
the reduction of data from TEXES (EXES's predecessor instrument). The
FORTRAN algorithms were translated into IDL and adapted for EXES by
Melanie Clarke for the SOFIA DPS team in 2013 and 2014. Version 1.0.0 was
released for use at SOFIA in 2015. The EXES IDL package was never
used as a standalone data reduction package, so it did not provide interface
scripts for its algorithms; it was designed specifically for
incorporation into Redux.

FSpextool was built on top of a pre-release version of Spextool 4, an
IDL-based package developed by Dr. Michael Cushing and Dr. William Vacca
for the reduction of data from the SpeX instrument on the NASA Infrared
Telescope Facility (IRTF). Spextool was originally released in
October 2000, and has undergone a number of major and minor revisions since
then. The last stable public release was v4.1, released January 2016.
As Spextool does not natively support automatic command-line processing,
FSpextool for SOFIA adapted the Spextool library to the SOFIA architecture
and instruments; version 1.0.0 was originally released for use at SOFIA
in July 2013.

Redux was developed to be a general-purpose interface to IDL data
reduction algorithms. It provided an interactive GUI and an
object-oriented structure for calling data reduction processes, but it
does not provide its own data reduction algorithms. It was developed by
Melanie Clarke for the SOFIA DPS team, to provide a consistent front-end
to the data reduction pipelines for multiple instruments and modes,
including EXES. It was first released in December 2013.

Between 2015 and 2021, the EXES Redux package was maintained by the EXES
PI team, under Dr. Matthew Richter at UC Davis.  The principal developers
were Dr. Curtis DeWitt and Dr. Edward Montiel.  In 2021, the SOFIA DPS
team rebased the pipeline from the PI team version for release as EXES Redux
v2.0.0, in support of EXES's transition from PI class to facility class
instrument.

In 2022, the EXES pipeline was entirely reimplemented in Python, as a set of
software modules in the SOFIA Redux Python package:

   - `sofia_redux.instruments.exes`: processing algorithms
     specific to the EXES instrument
   - `sofia_redux.spectroscopy`: spectral extraction algorithms
   - `sofia_redux.pipeline`: interactive and batch mode interface tools for
     managing data reduction processes
   - `sofia_redux.toolkit`: numerical algorithms and supporting utilities
   - `sofia_redux.visualization`: data analysis and visualization tools

The `exes` module reimplements the algorithms in the EXES IDL package,
`pipeline` reimplements the Redux tools, and `spectroscopy` reimplements
FSpextool algorithms.

The SOFIA Redux package was developed as a unified Python package to
support data reduction for all facility class instruments for SOFIA,
replacing all legacy pipelines with an integrated, shared code base.
The package was developed by the SOFIA DPS team, starting in 2018.
The principal developers for SOFIA Redux prior to the EXES 3.0.0
release were Daniel Perera, Dr. Rachel Vander Vliet,
and Melanie Clarke, for the SOFIA DPS team, with additional contributions
from Dr. Karishma Bansal.


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

The modules used in the EXES pipeline are described below.


sofia_redux.instruments.exes
----------------------------
The `sofia_redux.instruments.exes` package is written in Python using
standard scientific tools and libraries.

The data reduction algorithms used by the pipeline are straight-forward
functions that generally take a data array, corresponding to a single
image file, as an argument and return the processed image array as a result.
They generally also take as secondary input a variance array to process
alongside the image, a header structure to track metadata, and keyword
parameters to specify non-default settings.

The `exes` module also stores any reference data needed by the EXES
pipeline.  This includes bad pixel masks, nonlinearity coefficients,
dark files, and default header values and distortion correction parameters.
The default files may vary by date; these defaults are managed by the
`readhdr` function in the `exes` module.  New date configurations may be added
to the caldefault.txt file in *exes/data/caldefault.txt*.

sofia_redux.toolkit
-------------------

`sofia_redux.toolkit` is a repository for classes and functions of general
usefulness, intended to support multiple SOFIA pipelines.  It contains several
submodules, for interpolation, image manipulation, multiprocessing support,
numerical calculations, and FITS handling.  The utilities used by EXES are
generally simple functions that take input as arguments and return output
values.

sofia_redux.spectroscopy
------------------------

The `sofia_redux.spectroscopy` package contains a library of general-purpose
spectroscopic functions.  The EXES pipeline uses these algorithms
for spectroscopic image rectification, aperture identification, and
spectral extraction. Most of these algorithms are simple
functions that take spectroscopic data as input and return processed data
as output.  However, the input and output values may be more complex than the
image processing algorithms in the `exes` package.  The Redux interface
in the `pipeline` package manages the input and output requirements for
EXES data and calls each function individually.  See the
`sofia_redux.spectroscopy` API documentation for more information.


sofia_redux.visualization
-------------------------

The `sofia_redux.visualization` package contains plotting and display
routines, relating to visualizing SOFIA data.  For the EXES pipeline,
this package currently provides a module that supports generating
quick-look preview images.

sofia_redux.pipeline
--------------------

Design
~~~~~~

.. include:: ../../../sofia_redux/pipeline/redux_architecture.rst


EXES Redux
~~~~~~~~~~~~~

To interface to the EXES pipeline algorithms, Redux defines the
`EXESReduction` and `EXESParameters` classes. See :numref:`exes_redux_class`
for a sketch of the Redux classes used by the EXES pipeline.  The EXESReduction
class calls the `sofia_redux.instruments.exes` reduction functions, with
support from the `sofia_redux.toolkit` and `sofia_redux.spectroscopy packages`.
The EXESParameters class defines default parameter values for all reduction
steps.

The EXESReduction holds definitions for all reduction steps for the
EXES pipeline:

    - Load Data: calls `sofia_redux.instruments.exes.readhdr`
    - Coadd Readouts: calls `sofia_redux.instruments.exes.readraw`
      and optionally `sofia_redux.instruments.exes.derasterize` or
      `sofia_redux.instruments.exes.correct_row_gains`
    - Make Flat: calls `sofia_redux.instruments.exes.makeflat` and
      `sofia_redux.instruments.exes.wavecal`
    - Despike: calls `sofia_redux.instruments.exes.despike`
    - Debounce: calls `sofia_redux.instruments.exes.debounce`
    - Subtract Nods: calls `sofia_redux.instruments.exes.diff_arr` and
      optionally `sofia_redux.instruments.exes.cirrus`
    - Flat Correct: calls `sofia_redux.instruments.exes.calibrate`
    - Clean Bad Pixels: calls `sofia_redux.instruments.exes.clean`
    - Undistort: calls `sofia_redux.instruments.exes.tort`
    - Correct Calibration: calls `sofia_redux.instruments.exes.makeflat.bnu`
    - Coadd Pairs: calls `sofia_redux.instruments.exes.coadd` and optionally
      `sofia_redux.instruments.exes.spatial_shift` or
      `sofia_redux.instruments.exes.submean`
    - Make Profiles: calls `sofia_redux.spectroscopy.mkspatprof` and
      `sofia_redux.spectroscopy.rectify`
    - Locate Apertures: calls `sofia_redux.spectroscopy.findapertures`
    - Set Apertures: calls `sofia_redux.spectroscopy.getapertures` and
      `sofia_redux.spectroscopy.mkapmask`
    - Subtract Background: calls `sofia_redux.spectroscopy.extspec.col_subbg`
    - Extract Spectra: calls `sofia_redux.spectroscopy.extspec` and
      `sofia_redux.instruments.exes.get_atran`
    - Combine Spectra: calls `sofia_redux.toolkit.image.combine_images`
    - Refine Wavecal: calls `sofia_redux.instruments.exes.wavecal`
    - Merge Orders: calls `sofia_redux.spectroscopy.mergespec` and
      `sofia_redux.toolkit.image.coadd`
    - Make Spectral Map: calls
      `sofia_redux.visualization.quicklook.make_spectral_plot`

The recipe attribute for the reduction class specifies the above steps,
in that order.

If an intermediate file is loaded, its product type is
identified from the PRODTYPE keyword in its header, and the prodtype_map
attribute is used to identify the next step in the recipe.  This
allows reductions to be picked up at any point, from a saved intermediate
file.  For more information on the scientific goals and methods used in
each step, see the EXES Redux User's Manual.

The EXESReduction class also contains several helper functions, that
assist in reading and writing files on disk, and identifying which
data to display in the interactive GUI.  Display is performed via
the `QADViewer` class provided by the Redux package.  Spatial profiles
and aperture locations are additionally displayed by the `MatplotlibViewer`
class.  One-dimensional spectra are displayed by the `EyeViewer` class.

.. figure:: images/redux_classes.png
   :alt: EXES Redux Classes
   :name: exes_redux_class

   EXES Redux class diagram.


Detailed Algorithm Information
==============================
The following sections list detailed information on the functions and
procedures most likely to be of interest to the developer.


sofia_redux.instruments.exes
-------------------------------

.. automodapi:: sofia_redux.instruments.exes.calibrate
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.cirrus
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.clean
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.coadd
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.correct_row_gains
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.debounce
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.derasterize
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.derive_tort
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.despike
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.diff_arr
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.get_atran
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.get_badpix
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.get_resolution
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.lincor
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.make_template
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.makeflat
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.mergehdr
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.readhdr
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.readraw
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.spatial_shift
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.submean
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.tort
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.tortcoord
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.utils
   :headings: ~^
.. automodapi:: sofia_redux.instruments.exes.wavecal
   :headings: ~^

sofia_redux.toolkit
-------------------
.. automodapi:: sofia_redux.toolkit.convolve.base
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.convolve.kernel
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.convolve.filter
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.fitting.fitpeaks1d
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.fitting.polynomial
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.adjust
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.coadd
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.combine
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.fill
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.resize
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.smooth
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.utilities
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.image.warp
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.interpolate.interpolate
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.stats.stats
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.utilities.base
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.utilities.fits
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.utilities.func
   :headings: ~^
.. automodapi:: sofia_redux.toolkit.utilities.multiprocessing
   :headings: ~^

sofia_redux.spectroscopy
------------------------

.. automodapi:: sofia_redux.spectroscopy.binspec
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.earthvelocity
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.extspec
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.findapertures
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.fluxcal
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.getapertures
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.getspecscale
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.mergespec
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.mkapmask
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.mkspatprof
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.radvel
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.readflat
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.readwavecal
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.rectify
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.rectifyorder
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.smoothres
   :headings: ~^
.. automodapi:: sofia_redux.spectroscopy.tracespec
   :headings: ~^

sofia_redux.visualization
-------------------------

.. automodapi:: sofia_redux.visualization.quicklook
   :headings: ~^
.. automodapi:: sofia_redux.visualization.redux_viewer
   :headings: ~^
.. automodapi:: sofia_redux.visualization.controller
   :headings: ~^
.. automodapi:: sofia_redux.visualization.eye
   :headings: ~^

sofia_redux.pipeline
--------------------

The Redux application programming interface (API), including the EXES
interface classes, are documented in the `sofia_redux.pipeline` package.

.. toctree::

   redux_doc

.. raw:: latex

    \clearpage

Appendix A: Pipeline Recipe
===========================

This JSON document is the black-box interface specification for the
EXES Redux pipeline, as defined in the Pipetools-Pipeline ICD.

.. include:: include/exes_recipe.json
   :literal:

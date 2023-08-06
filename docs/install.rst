============
Installation
============

Stable release
--------------

The `sofia_redux` package is available via anaconda or pip::

   conda install -c sofia-usra -c conda-forge sofia_redux

or::

   pip install sofia_redux


From source
-----------

Obtain the source code for this package from the `SOFIA Redux GitHub project
<https://github.com/SOFIA-USRA/sofia_redux>`__, then install via one of the
two methods below.

Via Anaconda
^^^^^^^^^^^^

We recommend Anaconda for managing your Python environment.  A conda
environment specification is included with this package, as
`environment.yml <https://raw.githubusercontent.com/SOFIA-USRA/sofia_redux/main/environment.yml>`__.

To install a ``sofia_redux`` environment with Anaconda from the package directory::


   conda env create -f environment.yml


Activate the environment::

   conda activate sofia_redux


Install an editable copy of the `sofia_redux` package::

   pip install -e .


Deactivate the environment when done::

   conda deactivate


Remove the environment if necessary::

   conda remove --name sofia_redux --all


Via Pip
^^^^^^^

Alternately, prerequisites for the package can be installed with::

  pip install -r requirements.txt

and the package can then be installed as usual::

   pip install -e .

Optional Requirements
---------------------

PyQt5
^^^^^

If sofia_redux is installed via pip, the PyQt5 package, required for
the pipeline GUI interface, is not automatically installed as a dependency.
To use the GUI tools, install PyQt5 via pip::

  pip install PyQt5

or conda::

  conda install pyqt

Please note that there may be some incompatibilities between some versions
of PyQt5, some versions of the package dependencies, and some host OS versions.
If you run into difficulty, we recommend using the Anaconda installation
method.

DS9
^^^

Some optional visualization tools in the SOFIA Redux interface also
use the `pyds9` and `regions` packages to interface with the external
SAOImage DS9 tool. To use these tools, install
`DS9 <https://sites.google.com/cfa.harvard.edu/saoimageds9>`__, then
install pyds9 and regions directly via pip::

  pip install pyds9 regions

or using the provided optional requirements file::

  pip install -r optional_requirements.txt

Please note that pyds9 requires gcc to compile, and is not available
on the Windows platform.  On MacOS, you will need to make a `ds9`
executable available in your PATH environment variable; see the
`DS9 FAQs <http://ds9.si.edu/doc/faq.html#MacOSX>`__ for more information.

Reference data
^^^^^^^^^^^^^^

Some pipeline modes require additional reference data for optimal data
quality.  These files are too large to distribute with the pipeline code,
so they are provided separately.

Atmospheric models
~~~~~~~~~~~~~~~~~~
For optimal telluric correction, FORCAST, FLITECAM, and FIFI-LS spectroscopic
reductions require a library of FITS files, containing model atmospheric
transmission spectra, derived from the
`ATRAN model <https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi>`__.

Two versions of the model libraries are available for each instrument, except
FLITECAM.  ATRAN files parameterized by water vapor are not available for
FLITECAM.

The EXES pipeline does not use ATRAN models for telluric correction, but it
does attach a reference atmospheric model at a matching altitude and zenith
angle to output spectral products, if available. The models used are derived
from the `Planetary Spectrum Generator (PSG) <https://psg.gsfc.nasa.gov/>`__.

- FORCAST:

  - Approximate models, not accounting for water vapor variation

    - Download: `atran_forcast_standard.tgz <https://irsa.ipac.caltech.edu/data/SOFIA/ATRAN_FITS/atran_forcast_standard.tgz>`__
    - Size: 531.3 MB
    - MD5 checksum: 61141843b245eea1fdfd45167a1a750b

  - More accurate models, enabling programmatic optimization of
    the telluric correction

    - Download: `atran_forcast_wv.tgz <https://irsa.ipac.caltech.edu/data/SOFIA/ATRAN_FITS/atran_forcast_wv.tgz>`__
    - Size: 37.5 GB
    - MD5 checksum: 49264dfd6c3288af2553f73f8082ec97

- FIFI-LS:

  - Approximate models, not accounting for water vapor variation

    - Download: `atran_fifi-ls_standard.tgz <https://irsa.ipac.caltech.edu/data/SOFIA/ATRAN_FITS/atran_fifi-ls_standard.tgz>`__
    - Size: 143.9 MB
    - MD5 checksum: 9a6480d5967f4287388a3070e71e40e8

  - More accurate models, enabling use of water vapor values
    recorded in the FITS headers for more accurate telluric correction

    - Download: `atran_fifi-ls_wv.tgz <https://irsa.ipac.caltech.edu/data/SOFIA/ATRAN_FITS/atran_fifi-ls_wv.tgz>`__
    - Size: 2.8 GB
    - MD5 checksum: 486a34fd229b13d8e45768f3664fff64

- FLITECAM:

  - Approximate models, not accounting for water vapor variation

    - Download: `atran_flitecam_standard.tgz <https://irsa.ipac.caltech.edu/data/SOFIA/ATRAN_FITS/atran_flitecam_standard.tgz>`__
    - Size: 875 MB
    - MD5 checksum: 6576883144bcc381eacdfe16688ad4d2

- EXES:

  - Approximate models, not accounting for water vapor variation

    - Download: `psg_exes_standard.tgz <https://irsa.ipac.caltech.edu/data/SOFIA/ATRAN_FITS/psg_exes_standard.tgz>`__
    - Size: 5.4 GB
    - MD5 checksum: 147cf56cf15f2626b75a600e1ede5410


After downloading and unpacking the library, its location can be provided
to the pipeline as an optional parameter in the *Calibrate Flux* step for
FORCAST or FLITECAM, the *Telluric Correct* step for FIFI-LS, or the
*Extract Spectra* step for EXES.

Standard flux models
~~~~~~~~~~~~~~~~~~~~
In addition to the ATRAN models, a library of standard flux models is
required to reduce FORCAST or FLITECAM standard spectra to instrumental
response curves. This should be rarely needed for standard scientific reductions,
since reference response curves are provided for most data.  If needed for
re-deriving spectral flux calibrations, the standard model spectra are
provided in the
`source distribution <https://github.com/SOFIA-USRA/sofia_redux>`__ of
this package, at sofia_redux/instruments/forcast/data/grism/standard_models
or sofia_redux/instruments/flitecam/data/grism/standard_models.

FLITECAM and EXES auxiliary data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default auxiliary calibration and reference data for FLITECAM and
EXES reductions are too large to be included in the software packages
provided via PyPI or Anaconda.

These files are provided in full in the
`source distribution <https://github.com/SOFIA-USRA/sofia_redux>`__ of
this package.  Since they are required for most data reductions for these
instruments, they may also be automatically downloaded as needed for
non-source installations (i.e. via pip or conda).  Downloaded calibration
files are cached for later use in a '.sofia_redux' directory in the user's
home directory. For offline pipeline reductions, the source installation
is recommended.

For FLITECAM, the data provided in this manner includes nonlinearity
correction coefficients, spectroscopic order masks, and wavelength
calibration files.  For EXES, the large data files are bad pixel masks,
reset dark files, and nonlinearity correction coefficients.


Troubleshooting
---------------

Please note that direct support for this project will end in September 2023.

Prior to this time, please submit a ticket on the GitHub issues page for
installation assistance.

After this time, the source distribution of this package will remain available,
but it will not be maintained for the latest versions of all dependencies. It
is recommended that users fork their own copies of this package for continued
maintenance and development.

The last working set of installed versions of all dependencies is recorded in the
`freeze_requirements.txt <https://raw.githubusercontent.com/SOFIA-USRA/sofia_redux/main/freeze_requirements.txt>`__
file in this package. If errors are encountered in the other listed installation
methods, it may be useful to install the frozen versions directly. For example, to install
from source using conda to create a new Python environment from the sofia_redux package
directory::

   conda create --name sofia_redux python=3.10
   pip install -r freeze_requirements.txt
   pip install -e .



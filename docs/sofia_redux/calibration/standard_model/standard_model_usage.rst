Usage
=====
If the pypecal package has been installed, the standard model program
can be accessed with the command::

    hawc_calibration

The program requires one argument, and has one optional argument. The required
argument is the name of a file that lists the object and timestamps to
generate models for. An example is::

    2016-12-01 05:00:00    Ceres
    2016-12-16 05:00:00    Ceres
    2017-10-17 11:00:00    Ceres
    2017-11-16 07:00:00    Ceres

The first column is the UTC date of the observation. The second column
is the UTC time of the observation. The third column is the name of
the calibration object. The script will generate a model for each
time point in the file.

The script accepts a second, optional argument, which is the name
of the desired ATRAN file to use. The script will look for the
file in /dps/calibrations/ATRAN/fits. If it cannot find it there, it will
try to find it in the limited collection of ATRAN files in this package.
If this argument is not provided, the default ATRAN model will be used::

    atran_41K_45deg_40-300mum.fits


Output
======

The resulting model is written to a file with the naming pattern::

    HAWC_<target>_<date>_<altitude>_<zenith angle>.out

The contents of this file are:

+----------+-------------------+---------------+---------------------------------------------------+
| Column   | Title             | Units         | Description                                       |
+==========+===================+===============+===================================================+
| 0        | lambda\_ref       | microns       | Reference wavelength                              |
+----------+-------------------+---------------+---------------------------------------------------+
| 1        | lambda\_mean      | microns       | Mean wavelength                                   |
+----------+-------------------+---------------+---------------------------------------------------+
| 2        | lambda\_1         | microns       | Mean wavelength weighted by photon distribution   |
+----------+-------------------+---------------+---------------------------------------------------+
| 3        | lambda\_pivot     | microns       | Conversion between F(lambda) and F(nu)            |
+----------+-------------------+---------------+---------------------------------------------------+
| 4        | lambda\_eff       | microns       | Effective wavelength                              |
+----------+-------------------+---------------+---------------------------------------------------+
| 5        | lambda\_eff\_jv   | microns       |                                                   |
+----------+-------------------+---------------+---------------------------------------------------+
| 6        | lambda\_iso       | microns       | Isophotal wavelength                              |
+----------+-------------------+---------------+---------------------------------------------------+
| 7        | lambda\_rms       | microns       | RMS wavelength                                    |
+----------+-------------------+---------------+---------------------------------------------------+
| 8        | width             | microns       |                                                   |
+----------+-------------------+---------------+---------------------------------------------------+
| 9        | Response          | W/mJy         |                                                   |
+----------+-------------------+---------------+---------------------------------------------------+
| 10       | F\_mean           | W/m^2/mum     | Mean flux                                         |
+----------+-------------------+---------------+---------------------------------------------------+
| 11       | Fnu\_mean         | Jy            | Mean flux in Jy                                   |
+----------+-------------------+---------------+---------------------------------------------------+
| 12       | ColorTerm         |               | k0                                                |
+----------+-------------------+---------------+---------------------------------------------------+
| 13       | ColorTerm         |               | k1                                                |
+----------+-------------------+---------------+---------------------------------------------------+
| 14       | Source Rate       | Watts         | Power of source                                   |
+----------+-------------------+---------------+---------------------------------------------------+
| 15       | Source Size       | pixels        | Size of source                                    |
+----------+-------------------+---------------+---------------------------------------------------+
| 16       | Sorce FWHM        | arcsec        | FWHM of source                                    |
+----------+-------------------+---------------+---------------------------------------------------+
| 17       | Bkgd Power        | Watts         | Background Power                                  |
+----------+-------------------+---------------+---------------------------------------------------+
| 18       | NEP               | W/sqrt(Hz)    | Noise Equivalent Power                            |
+----------+-------------------+---------------+---------------------------------------------------+
| 19       | NEFD              | Jy/sqrt(Hz)   | Noise Equivalent Flux Density                     |
+----------+-------------------+---------------+---------------------------------------------------+
| 20       | MDCF              | mJy           |                                                   |
+----------+-------------------+---------------+---------------------------------------------------+
| 21       | Npix              |               | Number of pixels in source                        |
+----------+-------------------+---------------+---------------------------------------------------+
| 22       | --                |               | Lambda prime                                      |
+----------+-------------------+---------------+---------------------------------------------------+
| 23       | --                |               | Wavelength correlation                            |
+----------+-------------------+---------------+---------------------------------------------------+
| 24       | Filter            |               | Name of the filer                                 |
+----------+-------------------+---------------+---------------------------------------------------+

Process overview
================

The overall process is fairly simple. Pass a model of the source flux
through the transmission of the system (such as atmosphere, filter,
instrument, etc) and calculate the total observed flux in each filter.

Source model
------------

The model of the source's flux is either pull from a Herschel model or
generated. Major body source (Neptune, Uranus, Callisto, and Ganymede)
are modeled by Herschel. The actual flux the source would generate at
the time of the observation is the model flux scaled by the square of
the distance ratio of the model and the actual observed distance. The
scaling of the model is handled by the modconvert.py.

Asteroid do not have Herschel models, so the model flux needs to be
generated. The model is a simple blackbody scaled by the location of the
asteroid in the solar system. The generation of the asteroid model is
handled by genastmodel2.py.

To handle either of these methods. the location of the source in the
solar system at the time of the observation is needed. This is handled
by horizons.py. This connects to JPL's Horizons service using the
astroquery package. This determines the ephemeris of the object at the
time of obseration and returns the needed data for each model type.

Calibration
-----------

The source flux is mitigated by the transmission of the light through the
atmosphere, the telescope assembly, and the filter. The background light
from the sky, telescope, window, etc. are also included. Taking this all
into account results in the mean flux from the source in each filter.

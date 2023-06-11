
Significant changes
-------------------
Below are listed the most significant changes for the EXES pipeline
over its history, highlighting impacts to science data products.
See the data handbooks or user manuals associated with each release
for more information.

All pipeline versions prior to v3.0.0 were implemented in IDL;
v3.0.0 and later were implemented in Python.  For previously processed data,
check the PIPEVERS keyword in the FITS header to determine the pipeline
version used.

EXES Redux v3.0.0 (2022-12-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. E*

- Full reimplementation of the IDL pipeline into Python 3.
- All intermediate and final data products were revised to separate
  flux, error, and calibration information into separate extensions in
  the output FITS files.
- Added a calibration correction step to account for wavenumber dependence
  in the blackbody calibration.  Additionally, modified blackbody calibration
  to account for an extra contribution from flat mirror reflection.
- Added conversion to flux units (Jy/pixel for 2D images; Jy for 1D spectra)
  as a separate step, following nod-pair coaddition.
- Allowed despike comparison to be performed across all input files for
  better statistics in truncated observations.
- Allowed NaN propagation for bad pixel identification, rather than
  requiring immediate interpolation.
- Cross-dispersed data are rotated to align spectra with image rows
  immediately after distortion correction, prior to coaddition and extraction.
- Added a reference response spectrum, extracted from flat data at the
  same location as the science and attached to 1D spectral products as a
  5th row.
- Added handling for resolution value configuration tables for
  cross-dispersed modes.
- Replaced ATRAN reference atmospheric spectra with Planetary Spectrum
  Generator (PSG) model files.
- Time keywords EXPTIME, INTTIME, and TOTTIME were revised to reflect
  nominal on-source time, integration time, and total elapsed time,
  respectively.


EXES Redux v2.2.0 (2022-04-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Added HRR and DETROT to distortion correction parameters in configuration
  files, so that the defaults can be overridden by date.
- Fixed a bug in integration tossing in the initial read and coadd, in the
  case where a single valid integration remains after tossing bad ones.


EXES Redux v2.1.0 (2021-11-22)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. C*

- Designated separate product names and types for sky spectra to avoid
  overwriting science products.
- Added additional support for intermediate reductions, including sky
  spectrum products and undistorted files.
- Added capability to compose flat fields from separate subarrays (rasters).
- Modified assumed plate scale by mode to more accurately reflect anamorphic
  magnification effects for cross-dispersed modes.

EXES Redux v2.0.0 (2021-10-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

- Adopted EXES PI team version for SOFIA DPS support as a facility class
  instrument.
- Modified NAIF ID handling to not write default -9999 value for sidereal
  targets.
- Added option to coadd frames across all files with outlier rejection.
- Added ASSC_MSN keyword to track all input mission IDs and ASSC_OBS
  to track all input OBS-IDs.
- Removed a scaling factor in the coadd step, originally intended for
  unit conversion and flux conservation, that was historically inconsistently
  applied to flux and variance planes.

EXES Redux v1.6.0 (2019-04-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

- Added handling for darks to allow for black-dark flat generation method
- Added option to reject 1 or 2 initial readout patterns before readout coadd
- Added option to use only destructive frames for readout coadds
- Modified options to allow manual selection of bad pixel masks
- Allowed manual overrides for the edges of order masks
- Added option to debounce in the spectral direction, as well as the spatial
- Added manual options to set asymmetric apertures with start and end values
- Added options to subtract a dark frame instead of nods, for sky emission
  spectra

EXES Redux v1.0.2 (2015-07-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

- Modified nonlinearity correction to allow for a lower-limit plane in the
  coefficient reference file.

EXES Redux v1.0.1 (2015-05-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

- Added ASSC_AOR keyword to track all input AOR-IDs.
- Attached reference atmospheric transmission data to spectra.
- Improved wavelength calibration.

EXES Redux v1.0.0 (2015-03-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

- Initial release.

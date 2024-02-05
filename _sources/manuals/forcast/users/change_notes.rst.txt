
Significant changes
-------------------
Below are listed the most significant changes for the FORCAST pipeline
over its history, highlighting impacts to science data products.
See the data handbooks or user manuals associated with each release
for more information.

All pipeline versions prior to v2.0.0 were implemented in IDL;
v2.0.0 and later were implemented in Python.  An early predecessor to the
FORCAST Redux pipeline, called DRIP/FG, was also released for FORCAST
reductions in 2013, but no data in the SOFIA archive remains that was
processed with this pipeline.

For previously processed data, check the PIPEVERS keyword in the
FITS header to determine the pipeline version used.

FORCAST Redux v2.7.0 (2022-12-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. M*

Imaging
^^^^^^^
- Add option to use measured water vapor values from the header (WVZ_OBS)
  for calibration reference.

Spectroscopy
^^^^^^^^^^^^
- Add option to use measured water vapor values from the header (WVZ_OBS)
  for selecting the atmospheric model used for telluric correction.


FORCAST Redux v2.6.0 (2022-09-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. L*

Imaging
^^^^^^^
- Add options to allow diagnostic reductions in the detector coordinate frame,
  skipping distortion correction and rotation by the sky angle.

Spectroscopy
^^^^^^^^^^^^
- Fix a bug in water vapor optimization, causing the pipeline to return
  too-low PWV values for observations with strong sky lines and significant
  wavelength shifts.


FORCAST Redux v2.5.0 (2022-05-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. K*

Spectroscopy
^^^^^^^^^^^^
- Accommodate slit scan data with asymmetric nods (SKYMODE=SLITSCAN_NXCAC).

FORCAST Redux v2.4.0 (2022-04-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. K*

Imaging
^^^^^^^
- Replace the scikit-image dependency with local implementations of warping
  and image interpolation algorithms.

Spectroscopy
^^^^^^^^^^^^
- Add a line list overplot feature to the spectral viewer for interactive
  pipeline reductions.


FORCAST Redux v2.3.0 (2021-06-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. J*

Spectroscopy
^^^^^^^^^^^^
- Improve the automatic wavelength shifting algorithm in the flux
  calibration step to be more reliable across a larger range of
  wavelength shifts.
- Additional features for the spectral viewer: reference overplots
  and enhanced feature fitting options.


FORCAST Redux v2.2.1 (2021-04-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. H*

Imaging
^^^^^^^
- Fix NaN handling in peak-finding algorithm for centroid registration.
- Fix expected units for TGTRA keyword, used for non-sidereal target
  registration.


FORCAST Redux v2.2.0 (2021-03-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. H*

All modes
^^^^^^^^^
- Add preview images (\*.png files) for all final data products.

Spectroscopy
^^^^^^^^^^^^
- In GUI mode, replace static spectral plots with an interactive viewer.


FORCAST Redux v2.1.0 (2020-09-01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. G*

Imaging
^^^^^^^
- Add BUNIT keys for all extensions.
- Fix for NaN handling with some image combination methods.

Spectroscopy
^^^^^^^^^^^^
- Fix WAVEPOS extension in spectral cube (SCB) to match wavelengths
  of the cube slices.
- Fix for wavelength shift optimization occasionally reporting spurious
  shifts.
- Add support for wavelength/spatial calibration file generation
  to the pipeline.  The output product is a WCL file (PRODTYPE=wavecal);
  it may be used in the Make Profiles step in the pipeline to update or
  customize the wavelength calibration.
- Add support for combining and smoothing response files generated
  from standards (RSP files).
- Add support for generating slit correction images to the pipeline.
  The output product is a SCR file (PRODTYPE=slit_correction). It
  may be used in the Make Profiles step to correct for slit response.
- Add SPECSYS=TOPOCENT keyword to FITS headers to indicate that wavelengths
  have not been corrected for barycentric velocity shifts.

FORCAST Redux v2.0.0 (2020-05-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. F*

All modes
^^^^^^^^^
- Full reimplementation of the IDL pipeline into Python 3.
- Images and spectral cubes now have the option of registering to
  a non-sidereal target position, rather than to the sidereal
  WCS.

Imaging
^^^^^^^
- Data formats change significantly.  Imaging products now separate
  flux, error, and exposure map into separate FITS image extensions,
  rather than storing them as a 3D cube in the primary extension.
  Note that the error (standard deviation) is now stored instead of
  variance.

Spectroscopy
^^^^^^^^^^^^
- Data formats change significantly.  Images and spectra are stored
  in the same FITS file, under separate extensions.  Final 1D spectra
  (CMB files, PRODTYPE=combined_spectrum) are still stored in the
  same format as before; the spectrum corresponds to the SPECTRAL_FLUX
  extension in the COA (PRODTYPE=coadded_spectrum) file.

FORCAST Redux v1.5.0 (2019-07-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. E*

Imaging
^^^^^^^
- Incorporate new pinhole masks for distortion correction. Allow
  different masks by date.

FORCAST Redux v1.4.0 (2019-02-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. E*

Spectroscopy
^^^^^^^^^^^^
- Introduce support for slit-scan observations.  The output product
  is a spatial-spectral cube (file code SCB, PRODTYPE=speccube,
  PROCSTAT=LEVEL_4).

FORCAST Redux v1.3.2 (2018-09-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

All modes
^^^^^^^^^
- Fix input manifest handling to not expect the number of files at
  the top of the list.

FORCAST Redux v1.3.1 (2018-03-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

All modes
^^^^^^^^^
- Added ASSC-MSN key to track all input MISSN-ID values, for mosaic
  support.  Also added ASSC-OBS keys to track all input OBS_ID values.

Imaging
^^^^^^^
- Fix for registration error in mosaics with non-empty COADX/Y0 keys.


FORCAST Redux v1.3.0 (2017-04-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

Imaging
^^^^^^^
- Exposure map is now stored in units of seconds, instead of
  number of exposures.
- Support for multi-field mosaics is introduced. The Level 4 product
  type is a MOS file (PRODTYPE=mosaic).
- Extra NaN borders are stripped from images after the merge step.
- Default registration method is now WCS comparison, rather than
  header shifts from dither keywords.

Spectroscopy
^^^^^^^^^^^^
- Incorporated process for generating instrumental response curves
  into the pipeline.  The output product is a response file (RSP)
  for each telluric standard observation.  RSP files can be combined
  together with a separate tool to generate a master response spectrum.

FORCAST Redux v1.2.0 (2017-01-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. C*

Imaging
^^^^^^^
- Flux calibration procedure revised to separate telluric correction
  from flux calibration.  Telluric correction is now performed on a
  file-by-file basis, for better accuracy, after registration.  The
  REG file is no longer saved by default; it is replaced by a TEL file
  which is telluric-corrected but not flux calibration.  The final
  calibration factor is still applied at the end of the pipeline, making
  a single CAL file.  The CALFCTR stored in the header is now the
  calibration factor at the reference altitude and zenith angle; it no
  longer includes the telluric correction factor.  The latter value is
  stored in the new keyword TELCORR.

Spectroscopy
^^^^^^^^^^^^
- Introduced telluric correction optimization, using a library of
  ATRAN files at various water vapor values, and using the one that
  best corrects the data. Derived WV values are stored in the FITPWV
  keyword.

FORCAST Redux v1.1.3 (2016-09-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

Imaging
^^^^^^^
- Rotation in the merge step is now performed around the CRPIX
  (boresight center) rather than the image center.  This fixed small
  misalignments among images of fields taken at multiple rotation values.

FORCAST Redux v1.1.2 (2016-07-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

Imaging
^^^^^^^
- Fix for flux calibration procedure to distinguish between
  Barr2 and Barr3 dichroics.

FORCAST Redux v1.1.1 (2016-06-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

Imaging
^^^^^^^
- Fix for bad NaN handling, leaving small artifacts in merged image.

FORCAST Redux v1.1.0 (2016-01-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

Imaging
^^^^^^^
- Flux calibration factors are now applied to data arrays to
  convert them to physical units (Jy).  The calibrated data product
  has file code CAL (PRODTYPE=calibrated).  COA files are no longer
  designated Level 3, even if their headers contain calibration
  factors.
- Border-padding around valid imaging data now has NaN value instead
  of 0.

FORCAST Redux v1.0.8 (2015-10-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

Spectroscopy
^^^^^^^^^^^^
- Bug fix for plot generation in headless mode.

FORCAST Redux v1.0.7 (2015-09-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

All modes
^^^^^^^^^
- Handle DETCHAN keyword set to SW/LW instead of 0/1.

Imaging
^^^^^^^
- Apply average calibration factors to standards, instead of derived
  value from photometry

FORCAST Redux v1.0.6 (2015-06-26)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

Imaging
^^^^^^^
- Fix for negative values in variance plane.
- Stop re-doing photometry for standards when applyin calibration factors.

FORCAST Redux v1.0.5 (2015-05-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

All modes
^^^^^^^^^
- Introduced the TOTINT keyword, to track the total integration time,
  as it would be requested in SITE, for more direct comparison with
  proposals.

FORCAST Redux v1.0.4 (2015-05-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

All modes
^^^^^^^^^
- Total nominal on-source exposure time now tracked in the EXPTIME keyword.
- Introduced the ASSC_AOR key to track all input AOR-IDs for each reduction.

Imaging
^^^^^^^
- Flux calibration is now integrated into the pipeline, rather than applied
  after the fact by a separate package.  Flux calibration factors are
  stored in keywords in the Level 3 data files; they are not directly
  applied to the data.
- Photometry is automatically performed on flux standard observations,
  with values stored in FITS keywords.

Spectroscopy
^^^^^^^^^^^^
- Introduced spatial correction maps for improved rectified images.
- Introduced slit response functions for detector response correction
  in the spatial direction.

FORCAST Redux v1.0.3 (2015-01-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

All modes
^^^^^^^^^
- Nonlinearity correction modified for High/Low capacitance distinction.
- Output filename convention updated to include flight number.
- Introduced date-handling for calibration parameters.

Imaging
^^^^^^^
- Source positions for standards recorded and propagated in SRCPOSX/Y
  keywords.

Spectroscopy
^^^^^^^^^^^^
- Modifications to default spectral extraction parameters to support
  extended sources.
- Scale spectra before merging to account for slit loss.
- Introduced option to turn off subtraction of median level from spatial
  profiles, to support extended sources and short slits.
- Introduced telluric correction and flux calibration.
- ITOT and NEXP keywords introduced to track total integration time.

FORCAST Redux v1.0.2 (2014-07-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

Spectroscopy
^^^^^^^^^^^^
- G2xG1 wavelength calibration update.

FORCAST Redux v1.0.1 (2014-06-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

Imaging
^^^^^^^
- Flux calibration package (pipecal) integration and improvements.

Spectroscopy
^^^^^^^^^^^^
- Wavelength calibration updates.

FORCAST Redux v1.0.0 (2013-12-30)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

All modes
^^^^^^^^^
- Integrated FORCAST imaging algorithms (DRIP) with Spextool spectral
  extraction algorithms, in a standard pipeline interface (Redux).

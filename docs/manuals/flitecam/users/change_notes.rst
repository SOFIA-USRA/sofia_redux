
Significant changes
-------------------
Below are listed the most significant changes for the FLITECAM pipeline
over its history, highlighting impacts to science data products.
See the data handbooks or user manuals associated with each release
for more information.

All pipeline versions prior to v2.0.0 were implemented in IDL;
v2.0.0 and later were implemented in Python.  An early predecessor to the
FLITECAM Redux pipeline, called FDRP/FSpextool, was also released for
FLITECAM reductions in 2013, but no data in the SOFIA archive remains
that was processed with this pipeline.

For previously processed data, check the PIPEVERS keyword in the
FITS header to determine the pipeline version used.


FLITECAM Redux v2.0.0 (2021-09-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

All modes
^^^^^^^^^
- Full reimplementation of the IDL pipeline into Python 3.

Imaging
^^^^^^^
- Data formats change significantly.  Imaging products now separate
  flux, error, and exposure map into separate FITS image extensions,
  rather than storing them as a 3D cube in the primary extension.

Spectroscopy
^^^^^^^^^^^^
- Data formats change significantly.  Images and spectra are stored
  in the same FITS file, under separate extensions.  Final 1D spectra
  (CMB files, PRODTYPE=combined_spectrum) are still stored in the
  same format as before; the spectrum corresponds to the SPECTRAL_FLUX
  extension in the COA (PRODTYPE=coadded_spectrum) file.


FLITECAM Redux v1.2.0 (2017-12-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

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
- Image registration default set to use the WCS for most image shifts,
  instead of centroid or cross-correlation algorithms.

FLITECAM Redux v1.1.0 (2016-09-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

Imaging
^^^^^^^
- Flux calibration factors are now applied to data arrays to
  convert them to physical units (Jy).  The calibrated data product
  has file code CAL (PRODTYPE=calibrated).  COA files are no longer
  designated Level 3, even if their headers contain calibration
  factors.

Spectroscopy
^^^^^^^^^^^^
- Grism calibration incorporated into the pipeline, using stored
  instrumental response files, similar to the FORCAST grism calibration
  process.

FLITECAM Redux v1.0.3 (2015-10-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

All modes
^^^^^^^^^
- Minor bug fixes for filename handling and batch processing.

FLITECAM Redux v1.0.2 (2015-09-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

Imaging
^^^^^^^
- Improvements to the flat generation procedures.

FLITECAM Redux v1.0.1 (2015-05-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

All modes
^^^^^^^^^
- EXPTIME keyword updated to track total nominal on-source integration time.
- ASSC_AOR keyword added to track all input AOR-IDs.

Imaging
^^^^^^^
- Separate flat and sky files accommodated.
- Flux calibration incorporated into pipeline, rather than applied as a
  separate step.

FLITECAM Redux v1.0.0 (2015-01-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

All modes
^^^^^^^^^
- Integrated FLITECAM imaging algorithms (FDRP) with Spextool spectral
  extraction algorithms, in a standard pipeline interface (Redux).

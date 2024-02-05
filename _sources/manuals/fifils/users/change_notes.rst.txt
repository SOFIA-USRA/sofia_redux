
Significant changes
-------------------
Below are listed the most significant changes for the FIFI-LS pipeline
over its history, highlighting impacts to science data products.
See the data handbooks or user manuals associated with each release
for more information.

All pipeline versions prior to v2.0.0 were implemented in IDL;
v2.0.0 and later were implemented in Python. For previously processed
data, check the PIPEVERS keyword in the FITS header to determine the
pipeline version used.



FIFI-LS Redux v2.8.0
~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. L*

- Update spatial flat handling to expect different files by dichroic.
- Reject very small spectral flat values.

FIFI-LS Redux v2.7.1 (2022-12-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. L*

- Fix a bug in the option to produce a final map aligned with the
  detector orientation.

FIFI-LS Redux v2.7.0 (2022-09-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. L*

- Project spatial data into a common WCS grid for correct astrometry in the
  final spectral cube.
- Add optional scan reduction support to the resample step for OTF data.

FIFI-LS Redux v2.6.1 (2022-05-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. K*

- Fix performance issues for very large maps.

FIFI-LS Redux v2.6.0 (2021-12-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. J*

- Reformat reference wavelength calibration data for easier maintenance.

FIFI-LS Redux v2.5.1 (2021-04-26)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. J*

- Allow separate spatial flats for Blue Order 1 and Order 2 to better
  correct for pixel vignetting in the detector.

FIFI-LS Redux v2.5.0 (2021-04-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. J*

- Add preview images (\*.png files) for all final data products.

FIFI-LS Redux v2.4.0 (2020-12-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. H*

- Correction for local standard of rest (LSR) removed from barycentric
  shift correction. It is stored in the LSRSHFT FITS keyword instead.
- Introduce sample filtering for grating instability, when grating
  position data is available in the raw data tables..

FIFI-LS Redux v2.3.0 (2020-08-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. G*

- Output data formats for intermediate products change from binary
  tables to FITS image extensions for all data arrays.
- Introduce support for OTF mode scans.
- Add bias subtraction to the Fit Ramps step, reducing systematic
  flux variations.
- Improvements to error estimates, edge pixel handling, and adaptive
  smoothing in the resampling algorithm.

FIFI-LS Redux v2.2.0 (2020-07-01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. F*

- Python code refactored into common namespace, for compatibility
  with other SOFIA pipelines.
- Fix for improperly cached ATRAN data.
- Decoupled output pixel size from spatial FWHM, so that resampling
  is accurate and output pixel sizes are consistent (1.5"/pixel for the
  blue channel and 3.0"/pixel for the red).

FIFI-LS Redux v2.1.0 (2019-11-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. E*

- Introduce adaptive smoothing kernel for resampling algorithm.
- Reference spatial FWHM revised by wavelength to allow more accurate
  smoothing kernels in resampling.  This allowed output pixel
  sizes to vary, so that they are no longer consistent for each channel.

FIFI-LS Redux v2.0.0 (2019-10-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. E*

- Full reimplementation of the IDL pipeline into Python 3.
- Accommodate flat correction by filter position as well as order,
  to allow 1st order filter with 2nd order blue data at some
  wavelengths.

FIFI-LS Redux v1.7.0 (2019-06-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Reference flat fields revised to use FITS data arrays, rather
  than text files.  Spatial flats now vary by date; spectral flats
  are static, with one provided for each possible combination of channel
  (RED, BLUE), order (1, 2), and dichroic (D105, D130).
- Added support for telluric correction at specific water vapor values,
  recorded in FITS header keys (WVZ_STA, WVZ_END). Requires a library
  of ATRAN files generated at regular altitude, ZA, and WV.

FIFI-LS Redux v1.6.0 (2019-02-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Update flux propagation to ensure that flux densities are propagated
  throughout the pipeline, removing dependence on spectral bin size.
  Flux calibrations for reductions prior to this version may contain
  errors, depending on the relative wavelength bin size between the standards
  used to generate the response curves and those used in the science
  reduction.

FIFI-LS Redux v1.5.1 (2018-11-01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. C*

- Fix input manifest handling to not expect the number of files at
  the top of the list.

FIFI-LS Redux v1.5.0 (2018-03-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. C*

- Modify ramp fitting procedure to remove the first two data points from
  each ramp to allow longer chop transitions and the first two ramps from
  all data sets to allow longer grating transition times.
- Accommodate a new filter set introduced in 2017.

FIFI-LS Redux v1.4.0 (2017-07-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. C*

- Fix for uncorrected flux cube exposure map not matching the data
  cube.
- Attach an additional unsmoothed ATRAN spectrum to the final data
  product, for reference.
- Improve spatial calibration by accounting for offsets between
  the primary array and boresight.

FIFI-LS Redux v1.3.3 (2017-01-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

- Fix for wavelength calibration bug due to accidental integer division.

FIFI-LS Redux v1.3.2 (2016-10-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

- Separated systematic error due to flux calibration from statistical
  error propagated in the ERROR data array.  Mean calibration error
  is instead recorded in the CALERR FITS keyword.

FIFI-LS Redux v1.3.1 (2016-07-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

- Added blocking for known bad pixels to the Fit Ramps step of the
  pipeline.
- Added an additional flux and error cube to the output products,
  uncorrected for atmospheric transmission.

FIFI-LS Redux v1.3.0 (2016-06-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

- Introduced parallel processing for embarrassingly parallel loops
  in pipeline steps.
- Introduced telluric correction, using ATRAN models at matching
  altitude and zenith angle.
- Introduced flux calibration, using response spectra generated from
  standard sources with known models to calibrate spectra to Jy/pixel.

FIFI-LS Redux v1.2.0 (2016-03-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

- Added support for maps generated from multiple base positions.
- Added support for total power (no chop) mode.
- Distance weighting function modified to a Gaussian function,
  improving resampling artifacts.
- Modified wavelength calibration to directly read spreadsheet
  provided by the instrument team.
- Attached model atmospheric transmission data to output product,
  for reference.
- Set default spatial sampling such that output products are 1"/pixel
  for blue channel data and 2"/pixel for red.
- Added edge-blocking to eliminate noisy extrapolated data at the
  edges of maps.

FIFI-LS Redux v1.1.1 (2016-02-16)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

- Update spatial calibration to account for the offset between
  the primary and secondary array.

FIFI-LS Redux v1.1.0 (2016-01-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

- Add full spatial/spectral WCS for final data cube.

FIFI-LS Redux v1.0.0 (2015-11-19)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

- Initial release.

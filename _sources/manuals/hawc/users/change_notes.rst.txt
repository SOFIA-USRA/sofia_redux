
Significant changes
-------------------
Below are listed the most significant changes for the HAWC+ pipeline
over its history, highlighting impacts to science data products.
See the data handbooks or user manuals associated with each release
for more information.

For previously processed data, check the PIPEVERS keyword in the FITS
header to determine the pipeline version used.

HAWC DRP v3.2.0 (2022-12-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. L*

- Improved the 'fixjumps' algorithm for correcting discrepant artifacts
  in scan maps caused by detector flux jumps.

HAWC DRP v3.1.0 (2022-09-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. K*

- Added a 'grid' parameter for the scan map steps to allow easy spatial
  regridding without impacting flux conservation.

HAWC DRP v3.0.0 (2022-02-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. J*

- Replaced the Java sub-pipeline for reconstructing scanned maps with
  a Python implementation (sofia_redux.scan).
- Added optional step to correct the zero level in total intensity
  scan maps.

HAWC DRP v2.7.0 (2021-08-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. H*

- Added support for generating noise plots from lab data.
- Fixed a time accounting bug in the EXPTIME keyword for scan-pol data.
  Prior to this version, EXPTIME in the reduced data products counted
  only the exposure time from a single HWP angle.
- Added new visualization tools to the pipeline interface and QAD tool.

HAWC DRP v2.6.0 (2021-04-26)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. G*

- Improvements to error estimates, edge pixel handling, and adaptive
  smoothing in the resampling algorithm.
- Introduce a pipeline mode to be used to generate new skycal files,
  scan-mode flats, and bad pixel lists from scans of bright sources.
- Add preview images (\*.png files) for all final data products.
- Improvement for parallel processing across disparate architectures.
- Add an optional pixel-binning step for the chop-nod pipeline,
  to allow improved S/N at the cost of decreases resolution.
- Introduce a zero-level correction algorithm for scanning polarimetry
  maps of large, diffuse sources.

HAWC DRP v2.5.0 (2020-06-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. F*

- Python code refactored into common namespace, for compatibility
  with other SOFIA pipelines.
- Improve error estimates for photometry profile fits for flux
  standards.

HAWC DRP v2.4.0 (2020-01-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. E*

- Add SIBS offset value calculation in FITS headers (SIBS_DXE, SIBS_DE),
  for computing pointing corrections for the scan-mode pipeline.
- Internal C library replaced with Python algorithms.

HAWC DRP v2.3.2 (2019-09-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Scan mode data frames at the beginning and end of observations
  are now trimmed, by default, to account for the pause between
  data readouts begin/end and telescope movement begin/end.
- Add option to allow manual override for Stokes combination,
  when HWP angle is inaccurately recorded.

HAWC DRP v2.3.1 (2019-08-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Fix for occasional WCS offset error in scan-pol mode.

HAWC DRP v2.3.0 (2019-07-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Scanning polarimetry now groups data in sets of 4 HWP angles and
  combines the data after computing Stokes parameters, rather than
  running common HWP angles through the CRUSH sub-pipeline together.
  This allows better correction for sky rotation angle (VPA).

HAWC DRP v2.2.0 (2019-05-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Fix for parameter resets between files in a single reduction run.
- Revise Python packaging structure to avoid manual C library compilation.

HAWC DRP v2.1.0 (2019-02-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. D*

- Introduce support for scanning polarimetry.
- Flux calibration improvements: add automated photometry routines
  for flux standards, move scan-mode calibration out of CRUSH sub-pipeline
  and into the same Python step used for chop-nod mode. Default saved
  products are changed.
- Introduced option for sigma-clipping on telescope velocity in the
  scan modes.

HAWC DRP v2.0.0 (2018-09-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. C*

- Refactored all Python 2 code into Python 3.
- Integrated pipeline algorithms into Redux interface for consistency
  with other SOFIA pipelines.
- Fixes for BUNIT keywords in extension headers.

HAWC DRP v1.3.0 (2018-05-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. B*

- Introduce instrumental polarization maps to correct IP for each
  detector pixel.
- Modify background subtraction to apply to Stokes Q and U as well
  as Stokes I images.
- Remove unused, empty pixel covariance planes from output data products.
- Demodulation step separated into two parts in order to separate pixel
  flagging from filtering, to allow inspection of the flagged data.
- Outlier rejection improvements for the time-series combination step.
- Add diagnostic plots (\*DPL\*.png) of demodulated data.
- Error propagation improvements: calculating initial errors from raw
  samples (before demodulation and R-T subtraction), and propagating
  covariance between Stokes parameters.

HAWC DRP v1.2.0 (2017-11-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

- Track all input MISSN-IDs in the ASSC_MSN FITS keyword.

HAWC DRP v1.1.1 (2017-05-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

- Fix sign error for WCS in SI reference frame.

HAWC DRP v1.1.0 (2017-05-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. A*

- Introduce flats for chop-nod mode derived from internal calibrator files
  bracketing science observations.
- Update scan mode opacity corrections to match chop-nod mode method
  (from ATRAN model).

HAWC DRP v1.0.1 (2017-01-30)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

- Fix for bad pixel mask handling for T array.

HAWC DRP v1.0.0 (2017-01-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*User manual: Rev. -*

- Initial release.

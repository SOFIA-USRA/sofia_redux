Instrumental Response Curve
---------------------------

As described above, instrumental response
curves are automatically produced for each spectrum with
OBSTYPE = STANDARD_TELLURIC.  For use in calibrating science spectra,
response curves from multiple observations must be combined together.

For appropriate combination, input response curves must share the same
grism, slit, and detector bias setting.

Matching response curves may be scaled, to account for variations in slit
loss or model accuracy, then are generally combined together with a robust
weighted mean statistic.  The combined curve is smoothed with a Gaussian
of width 2 pixels, to reduce artificial artifacts.  Averaged response curves
for each grism and slit combination are usually produced for each
flight series, and stored for pipeline use in the standard location for
the instrument package (*data/grism/response*).

The scaling, combination, and smoothing of instrument response curves is
implemented as a final step in the pipeline for grism standards.
After individual *response_spectrum* files (\*RSP\*.fits) are grouped
appropriately, the final step in the pipeline can be run on each group to
produce the average *instrument_response* file (\*IRS\*.fits).

Useful Parameters
~~~~~~~~~~~~~~~~~

Below are some useful parameters for combining response spectra.

- **Combine Response**

   - Scaling Parameters

      - *Scaling method*: If 'median', all spectra are scaled to the median
        of the flux stack.  If 'highest', all spectra are scaled to the
        spectrum with the highest median value.  If 'lowest', all spectra
        are scaled to the spectrum with the lowest median value.  If
        'index', all spectra are scaled to the spectrum indicated in the
        *Index* parameter, below.  If 'none', no scaling is applied before
        combination.

      - *Index of spectrum to scale to*: If *Scaling method* is 'index', set
        this value to the index of the spectrum to scale.  Indices start
        at zero and refer to the position in the input file list.

   - Combination Parameters

      - *Combine apertures*: For multi-aperture data, it may be useful
        to produce a separate response curve for each aperture.  Select
        this option to combine them into a single reponse curve instead.

      - *Combination method*: Mean is default; median may also be useful
        for some input data.

      - *Weight by errors*: If set, the average of the data will be
        weighted by the variance. Ignored for method=median.

      - *Robust combination*: If set, data will be sigma-clipped before
        combination for mean or median methods.

      - *Outlier rejection threshold (sigma)*: The sigma-clipping threshold
        for robust combination methods, in units of sigma (standard deviation).

   - Smoothing Parameters

     - *Smoothing Gaussian FWHM*: Full-width-half-max for the Gaussian
       kernel used for smoothing the final response spectrum, specified
       in pixels.

Wavelength Calibration Map
--------------------------

Calibration Principles
~~~~~~~~~~~~~~~~~~~~~~

Grism wavelength and spatial calibrations are stored together in a
single image extension in a FITS file, where the first plane is the wavelength
calibration and the second is the spatial calibration.  The images should match
the dimensions of the raw data arrays, assigning a wavelength value in um
and a slit position in arcsec to every raw pixel.

These calibration files are generally derived from specialized calibration
data.  Wavelength calibration is best derived from images for which strong
emission or absorption lines fill the whole image, from top to bottom, and
evenly spaced from left to right.  Sky data may be used for this purpose
for some of the grism passbands; lab data may be more appropriate for others.
Raw data should be cleaned and averaged or summed to produce an image with as
high a signal-to-noise ratio in the spectroscopic lines as possible.

After preprocessing, the spectroscopic lines must be identified with specific
wavelengths from a priori knowledge, then they must be re-identified with a
centroiding algorithm at as many places across the array as possible.  The
identified positions can then be fit with a smooth 2D surface, which provides
the wavelength value in microns at any pixel, accounting for any optical
distortions as needed.

In principle, the spatial calibration proceeds similarly.  Spatial
calibrations are best derived from identifiable spectral continuua that
fill the whole array from left to right, evenly spaced from top to bottom.
Most commonly, special observations of a spectroscopic standard are taken,
placing the source at multiple locations in the slit.  These spectroscopic
traces are identified then re-fit across the array.  The identified positions
are again fit with a smooth 2D surface to provide the spatial position in
arcseconds up the slit at any pixel.  This calibration can then be used to
correct for spatial distortions, in the same way that the wavelength
calibration is used to rectify distortions along the wavelength axis.

Pipeline Interface
~~~~~~~~~~~~~~~~~~

The input data for calibration tasks is generally raw FITS
files, containing spectroscopic data.  In order to perform calibration steps
instead of the standard spectroscopic pipeline, the pipeline interface
requires a user-provided flag, either in an input configuration file, or
on the command line, as for example::

    redux_pipe -c wavecal=True /path/to/fits/files

for a wavelength calibration reduction or::

    redux_pipe -c spatcal=True /path/to/fits/files

for a spatial calibration reduction.

The first steps in either reduction mode are the same pre-processing steps
used in the standard pipeline reduction.
The stacking steps have optional parameters that allow for
the input data to be summed instead of subtracted (for calibration from
sky lines), or to be summed instead of averaged (for combining multiple
spectral traces into a single image).

Thereafter, the *wavecal* reduction performs the following steps.  Each step
has a number of tunable parameters; see below for parameter descriptions.

    - **Make Profiles**: a spatial profile is generated from the
      unrectified input image.

    - **Extract First Spectrum**: an initial spectrum is extracted from
      a single aperture, via a simple sum over a specified number of rows.

    - **Identify Lines**: spectrosopic lines specified in an input list are
      identified in the extracted spectrum, via Gaussian fits near guess
      positions derived from user input or previous wavelength calibrations.

    - **Reidentify Lines**: new spectra are extracted from the image at
      locations across the array, and lines successfully identified in the
      initial spectrum are attempted to be re-identified in each new spectrum.

    - **Fit Lines**: all input line positions and their assumed wavelength
      values are fit with a low-order polynomial surface.  The fit surface
      is saved to disk as the wavelength calibration file.

    - **Verify Rectification**: the derived wavelength calibration is applied
      to the input image, to verify that correctly rectifies the spectral
      image.

After preprocessing, the *spatcal* reduction performs similar steps:

    - **Make Profiles**: a spatial profile is generated from the
      unrectified input image.

    - **Locate Apertures**: spectral apertures are identified from the spatial
      profile, either manually or automatically.

    - **Trace Continuum**: spectrosopic continuuum positions are fit in
      steps across the array, for each identified aperture.

    - **Fit Traces**: all aperture trace positions are fit with a low-order
      polynomial surface.  The fit surface is saved to disk as the spatial
      calibration file.

    - **Verify Rectification**: the derived spatial calibration is applied
      to the input image, to verify that correctly rectifies the spectral
      image.

Intermediate data can also be saved after any of these steps, and can be
later loaded and used as a starting point for subsequent steps, just as in
the standard spectroscopic pipeline.  Parameter settings can also be saved
in a configuration file, for later re-use or batch processing.

Wavelength and spatial calibrations generally require different pre-processing
steps, or different input data altogether, so they cannot be generated at the
same time.  The pipeline interface will allow a previously generated wavelength
or spatial calibration file to be combined together with the new one in the
final input.  Optional previous spatial calibration input is provided to the
*wavecal* process in the **Fit Lines** step; optional previous wavelength
calibration input is provided to the  *spatcal* process in the **Fit Traces**
step.  If a previously generated file is not provided, the output file will
contain simulated data in the spatial or wavelength plane, as appropriate.

Reference Data
~~~~~~~~~~~~~~

Line lists for wavelength calibration are stored in the standard reference
data directory for the instrument package
(*data/grism/line_lists*).  In these lists,
commented lines (beginning with '#') are used for display only; uncommented
lines are attempted to be fit.  Initial guesses for the pixel position of
the line may be taken from a previous wavelength calibration, or from a
low-order fit to wavelength/position pairs input by the user.  Default
wavelength calibration files and line lists may be set by date, in the usual
way (see *data/grism/caldefault.txt*).

Spatial calibration uses only the assumed slit height in pixels and arcsec
as input data, as stored in the reference files in
*data/grism/order_mask*.  These values are not expected to change over time.

Display Tools
~~~~~~~~~~~~~

The pipeline incorporates several display tools for diagnostic purposes.
In addition to the DS9 display of the input and intermediate FITS files,
spatial profiles and extracted spectra are displayed in separate windows,
as in the standard spectroscopic pipeline. Identified lines for *wavecal*
are marked in the spectral display window (|ref_wavecal_plots|);
identified apertures for *spatcal* are marked in the spatial profile window
(|ref_spatcal_plots|).  Fit positions
and lines of constant wavelength or spatial position are displayed as
DS9 regions.  These region files are
also saved to disk, for later analysis.  Finally, after the line or trace
positions have been fit, a plot of the residuals, against X and Y position
is displayed in a separate window (|ref_wavecal_residuals|
and |ref_spatcal_residuals|). This plot is also saved to disk,
as a PNG file.


Useful Parameters
~~~~~~~~~~~~~~~~~

Some key parameters used specifically for the calibration modes are listed
below.  See above for descriptions of parameters for the steps shared with
the standard pipeline.

Wavecal Mode
^^^^^^^^^^^^

- **Stack Dithers**

   - *Ignore dither information from header*: This option allows all
     input dithers to be combined together, regardless of the dither
     information in the header.  This option may be useful in generating
     a high signal-to-noise image for wavelength identification.

- **Extract First Spectrum**

   - *Save extracted 1D spectra*: If set, a 1D spectrum is saved to disk
     in Spextool format.  This may be useful for identifying line locations
     in external interactive tools like xvspec (in the IDL Spextool package).

   - *Aperture location method*: If 'auto', the most significant peak
     in the spatial profile is selected as the initial spectrum region,
     and the aperture radius is determined from the FWHM of the peak.
     If 'fix to center', the center pixel of the slit is used as the
     aperture location.  If 'fix to input', the value specified as the
     aperture position is used as the aperture location.

   - *Polynomial order for spectrum detrend*: If set to an integer 0
     or higher, the extracted spectrum will be fit with a low order
     polynomial, and this fit will be subtracted from the spectrum.  This
     option may be useful to flatten a spectrum with a a strong trend,
     which can otherwise interfere with line fits.

- **Identify Lines**

   - *Wave/space calibration file*: A previously generated wavelength
     calibration file, to use for generating initial guesses of line
     positions.  If a significant shift is expected from the last wavelength
     calibration, the 'Guess' parameters below should be used instead.

   - *Line list*: List of wavelengths to fit in the extracted spectrum.
     Wavelengths should be listed, one per line, in microns.  If commented
     out with a '#', the line will be displayed in the spectrum as a dotted
     line, but a fit to it will not be attempted.

   - *Line type*: If 'absorption', only concave lines will be expected.  If
     'emission', only convex lines are expected.  If 'either', concave and
     convex lines may be fit.  Fit results for faint lines are generally
     better if either 'absorption' or 'emission' can be specified.

   - *Fit window*: Window (in pixels) around the guess position used as the
     fitting data.  Smaller windows may result in more robust fits for faint
     lines, if the guess positions are sufficiently accurate.

   - *Expected line width (pixel)*: FWHM expected for the fit lines.

   - *Guess wavelengths*: Comma-separated list of wavelengths for known
     lines in the extracted spectrum.  If specified, must match the list
     provided for *Guess wavelength position*, and the *Wave/space calibration
     file* will be ignored.  If two values are provided, they will be fit with
     a first-order polynomial to provide wavelength position guesses for
     fitting. Three or more values will be fit with a second-order polynomial.

   - *Guess wavelength position*: Comma-separated list of pixel positions for
     known lines in the image.  Must match the provided *Guess wavelengths*.

- **Reidentify Lines**

   - *Save extracted 1D spectra*: If set, all extracted spectra are saved to
     disk in Spextool format, for more detailed inspection and analysis.

   - *Aperture location method*: If 'step up slit', apertures will be placed
     at regular intervals up the slit, with step size specified in *Step size*
     and radius specified in *Aperture radius*.  If 'fix to input', then
     apertures will be at the locations specified by *Aperture position*
     and radius specified in *Aperture radius*.  If 'auto', apertures will
     be automatically determined from the spatial profile.

   - *Number of auto apertures*: If *Aperture location method* is 'auto',
     this many apertures will be automatically located.

   - *Aperture position*: Comma-separated list of aperture positions in pixels.
     Apertures in multiple input files may also be specified, using
     semi-colons to separate file input.
     If *Aperture location method* is 'auto', these will be used as starting
     points.  If 'fix to input', they will be used directly.

   - *Aperture radius*: Width of the extracted aperture, in pixels.  The
     radius may be larger than the step, allowing for overlapping spectra.
     This may help get higher S/N for extracted spectra in sky frames.

   - *Polynomial order for spectrum detrend*: As for the Extract First Spectrum
     step, setting this parameter to an integer 0 or higher will detrend
     it.  If detrending is used for the earlier step, it is recommended
     for this one as well.

   - *Fit window*: Window (in pixels) around the guess position used as the
     fitting data.  The guess position used is the position in the initial
     spectrum, so this window must be wide enough to allow for any curvature
     in the line.

   - *Signal-to-noise requirement*: Spectral S/N value in sigma, below
     which a fit will not be attempted at that line position in that
     extracted spectrum.

- **Fit Lines**

   - *Fit order for X*: Polynomial surface fit order in the X direction.
     Orders 2-4 are recommended.

   - *Fit order for Y*: Polynomial surface fit order in the Y direction.
     Orders 2-4 are recommended.

   - *Weight by line height*: If set, the surface fit will be weighted
     by the height of the line at the fit position.  This can be useful
     if there is a good mix of strong and weak lines across the array.
     If there is an imbalance of strong and weak lines across the array,
     this option may throw the fit off at the edges.

   - *Spatial calibration file*: If provided, the spatial calibration plane
     in the specified file will be combined with the wavelength fit to
     produce the output calibration file (\*WCL\*.fits).  The default is the
     wavelength calibration file from the previous series.  If not provided,
     a simulated flat spatial calibration will be produced and attached to
     the output calibration file.

Spatcal Mode
^^^^^^^^^^^^

Aperture location and continuum tracing follow the standard spectroscopic
method, with the exception that units are all in pixels rather than
arcseconds.  See above for descriptions of the parameters for the
Locate Apertures and Trace Continuum steps.

See the *wavecal* mode descriptions, above, for useful parameters for
the Stack and Stack Dithers steps.

- **Fit Trace Positions**

   - *Fit order for X*: Polynomial surface fit order in the X direction.
     Orders 2-4 are recommended.

   - *Fit order for Y*: Polynomial surface fit order in the Y direction.
     Orders 2-4 are recommended.

   - *Weight by profile height*: If set, the surface fit will be weighted
     by the height of the aperture in the spatial map at the fit position.

   - *Wavelength calibration file*: If provided, the wavelength calibration
     plane in the specified file will be combined with the spatial fit to
     produce the output calibration file (\*SCL\*.fits).  The default is the
     wavelength calibration file from the previous series.  If not provided,
     pixel positions will be stored in the wavelength calibration plane in
     the output file.

Slit Correction Image
---------------------

The response spectra used to flux-calibrate spectroscopic
data encode variations in instrument response in the spectral dimension,
but do not account for variations in response in the spatial dimension. For
compact sources, spatial response variations have minimal impact on the
extracted 1D spectrum, but for extended targets or SLITSCAN observations, they
should be corrected for.

To do so, the pipeline divides out a flat field, called a slit correction
image, that contains normalized variations in response in the spatial
dimension only.

These slit correction images can be derived from wavelength-rectified
sky frames, as follows:

    1. Median spectra are extracted at regular positions across the frame.
    #. All spectra are divided by the spectrum nearest the center of
       the slit.
    #. The normalized spectra are fit with a low-order polynomial to derive
       smooth average response variations across the full array.

The fit surface is the slit correction image.  It is stored as a single
extension FITS image, and can be provided to the standard spectroscopic
pipeline at the Make Profiles step.  These images should be regenerated
whenever the wavelength and spatial calibrations are updated, since the slit
correction image matches the rectified dimensions of the spectral data,
not the raw dimensions.

Pipeline Interface
~~~~~~~~~~~~~~~~~~

Similar to the *wavecal* and *spatcal* modes described above, the pipeline
provides a *slitcorr* mode to produce slit correction images starting from
raw FITS files.  This mode can be invoked with a configuration flag::

    redux_pipe -c slitcorr=True /path/to/fits/files


The pre-processing steps in *slitcorr* reduction mode are the same as in the
standard pipeline reduction, except that the default for the stacking steps
is to add all chop/nod frames and average all input files, to produce a
high-quality sky frame.  Rectification and spatial profile generation also
proceeds as usual, using the latest available wavelength calibration file.

Thereafter, the *slitcorr* reduction performs the following steps.  Each step
has a number of tunable parameters; see below for parameter descriptions.

    - **Locate Apertures**: a number of apertures are spaced evenly
      across the slit.

    - **Extract Median Spectra**: flux data is median-combined at each
      wavelength position for each aperture.

    - **Normalize Response**: median spectra are divided by the spectrum
      nearest the center of the slit.  The 2D flux image is similarly
      normalized, for reference.

    - **Make Slit Correction**: the normalized spectra are fit with a
      low-order polynomial to produce a smooth slit correction surface
      that matches the rectified data dimensions.

Intermediate data can also be saved after any of these steps, and can be
later loaded and used as a starting point for subsequent steps, just as in
the standard spectroscopic pipeline.  Parameter settings can also be saved
in a configuration file, for later re-use or batch processing.

Useful Parameters
~~~~~~~~~~~~~~~~~

Some key parameters used specifically for the *slitcorr* mode are listed
below.  See above for descriptions of parameters for the steps shared with
the standard pipeline.

-  **Locate Apertures**

   -  *Number of apertures*: For this mode, apertures are evenly spaced
      across the array.  Specify the desired number of apertures. The
      radius for each aperture is automatically assigned to not overlap
      with its neighbors.

- **Extract Median Spectra**

   - *Save extracted 1D spectra*: If set, all extracted spectra are saved to
     disk in a FITS file in Spextool format, for inspection.

- **Normalize Response**

   - *Save extracted 1D spectra*: Save normalized spectra to disk in
     Spextool format.

- **Make Slit Correction**

   - General Parameters

      - *Fit method*: If '2D', a single surface is fit to all the normalized
        spectral data, producing a smooth low-order polynomial surface.  If
        '1D', polynomial fits are performed in the y-direction only, at
        each wavelength position, then are smoothed in the x-direction
        with a uniform (boxcar) filter.  The 1D option may preserve
        higher-order response variations in the x-direction; the 2D option
        will produce a smoother surface.

      - *Weight by spectral error*: If set, the polynomial fits will be
        weighted by the error propagated for the normalized median spectra.

   - Parameters for 2D fit

      - *Fit order for X*: Polynomial surface fit order in the X direction.
        Orders 2-4 are recommended.

      - *Fit order for Y*: Polynomial surface fit order in the Y direction.
        Orders 2-4 are recommended.

   - Parameters for 1D fit

      - *Fit order for Y*: Polynomial fit order in the Y direction.
        Orders 2-4 are recommended.

      - *Smoothing window for X*: Boxcar width for smoothing in X direction,
        in pixels.

Spectral extraction with Redux is slightly more complicated than
image processing. The GUI breaks down the spectral extraction algorithms
into six separate reduction steps to give more control over the extraction
process. These steps are:

-  Make Profiles: Generate a smoothed model of the relative distribution
   of the flux across the slit (the spatial profile). After this step is
   run, a separate display window showing a plot of the spatial profile
   appears.

-  Locate Apertures: Use the spatial profile to identify spectra to extract.
   By default, Redux attempts to automatically identify sources, but
   they can also be manually identified by entering a guess position to
   fit near, or a fixed position, in the parameters. Aperture locations
   are plotted in the profile window.

-  Trace Continuum: Identify the location of the spectrum across the
   array, by either fitting the continuum or fixing the location to the
   aperture center.  The aperture trace is displayed as a region
   overlay in DS9.

-  Set Apertures: Identify the data to extract from the spatial profile.
   This is done automatically by default, but all aperture
   parameters can be overridden manually in the parameters for this
   step.  Aperture radii and background regions are plotted in the
   profile window (see |ref_profile|).

-  Subtract Background: Residual background is fit and removed for
   each column in the 2D image, using background regions specified
   in the Set Apertures step.

-  Extract Spectra: Extract one-dimensional spectra from the
   identified apertures. By default, Redux will perform standard
   extraction for observations that are marked as extended sources
   (SRCTYPE=EXTENDED\_SOURCE) and will attempt optimal extraction for
   any other value. The method can be overridden in the parameters for
   this step.

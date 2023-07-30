Usage
-----

The scan package can perform reductions from the Python command line
interface using the
:class:`Reduction <sofia_redux.scan.reduction.reduction.Reduction>` object.
Each object must be initialized with the name of the observing instrument.
The reduction must then be run on one or more input files using the
:meth:`Reduction.run() <sofia_redux.scan.reduction.reduction.Reduction.run>`
method.

The following example using the :mod:`example` instrument to create a
simulated FITS file and perform a simple reduction.

.. plot::
    :include-source:

    from sofia_redux.scan.reduction.reduction import Reduction

    from astropy import units
    from astropy.io import fits
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    from astropy.visualization import astropy_mpl_style
    import os
    import tempfile

    plt.style.use(astropy_mpl_style)

    work_path = tempfile.mkdtemp(suffix='scan_example')
    filename = os.path.join(work_path, 'simulated_data.fits')

    # Initialize the reduction for the example instrument
    reduction = Reduction('example')

    # Create a simulated data file
    reduction.info.write_simulated_hdul(
        filename, fwhm=10 * units.arcsec, scan_type='daisy',
        n_oscillations=22, radial_period=12 * units.second,
        ra='17h45m39.60213s', dec='-29d00m22.0000s',
        source_type='single_gaussian', constant_speed=True,
        s2n=30.0)

    # Perform the reduction
    hdul = reduction.run(filename, outpath=work_path,
                         blacklist='correlated.bias')

    # Display the results
    # The contents of the output FITS file and the hdul above are identical.
    output_file = os.path.join(work_path, 'Simulation.Simulation.1.fits')
    image_data = fits.getdata(output_file, ext=0)
    wcs = WCS(fits.getheader(output_file, ext=0))
    plt.subplot(projection=wcs)
    plt.imshow(image_data, origin='lower')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')


The exact nature of the reduction depends heavily on the
:ref:`configuration <scan_configuration>` which may be read from default
files, or passed in as optional keyword arguments.  The keys and values
should correspond to those listed in the :ref:`glossary <scan_glossary>`.
A nested dictionary may be directly passed in, and/or a flattened dot
separated set of options may also be supplied.  For example, all of the
reductions below will run using the same configuration:

.. code-block:: python

    reduction.run(files, correlated={'sky': {'gainrange': '0.1:1'}}, deep=True)

    reduction.run(files, correlated={'sky.gainrange': '0.1:1'}, deep=True)

    reduction.run(files, options={'correlated.sky.gainrange': '0.1:1',
                                  'deep':True})

Note that configuration options passed in this way will remain locked by
default for the remainder of the reduction.  For more flexible options, please
consider creating a separate configuration file.

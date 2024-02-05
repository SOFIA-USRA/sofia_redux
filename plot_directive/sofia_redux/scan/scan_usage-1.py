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
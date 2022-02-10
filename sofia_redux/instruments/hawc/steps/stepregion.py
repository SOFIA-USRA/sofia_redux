# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Polarization data quality cut pipeline step."""

from astropy import log
from astropy.io import fits
import numpy as np
from numpy import nanmax, where, pi, sin, cos, ones

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepRegion']


class StepRegion(StepParent):
    """
    Apply data quality cuts to polarization vectors.

    This step examines polarization vectors present in the POL DATA
    table and removes poor quality vectors, according to the parameters
    defined for the step.  The quality cuts defined include signal-to-noise
    thresholds and range cutoffs.  This step also writes a DS9 formatted
    region file containing vectors that survive the quality cuts.

    Input for this step is a DataFits with STOKES and ERROR frames for
    I, Q and U each and a table containing polarization vectors (POL DATA).
    This step should be run after
    `sofia_redux.instruments.hawc.steps.StepPolVec`.

    Output for this step is the same as the input, with the
    addition of a FINAL POL DATA table, containing the vectors after
    quality cuts. As a side effect, a text file containing the selected
    polarization vectors will be saved to disk, to the same directory and
    base name as the input, with the product type indicator replaced with
    REG.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'region', and are named with
        the step abbreviation 'REG'.

        Parameters defined for this step are:

        skip : int
            If greater than 1, then only every 'skip' vectors
            will be saved.
        offset : list of int
            Offset in pixels, given as [x, y], to start keeping
            vectors, if 'skip' is greater than 1.  For example,
            offset = [1, 1] and skip = 2 will keep pixel (1,1),
            (1, 3), (3, 1), (3, 3), etc.
        mini : float
            Do not save vectors with flux less than this fraction
            of peak flux in Stokes I.
        minp : float
            Do not save vectors with flux less than this value.
        sigma : float
            Do not save vectors with polarization percentage
            signal-to-noise less than this value.
        minisigi : float
            Do not save vectors with Stokes I signal-to-noise less
            than this value.
        maxp : float
            Do not save vectors with polarization percentage greater
            than this value.
        scale : bool
            If not set, all vectors in the output DS9 region
            file will be the same length.
        length : float
            Scale factor for length of polarization vectors in DS9
            region file.
        rotate : bool
            If set, vectors shown in the output DS9 region file will
            be rotated (B-field) vectors.
        debias : bool
            If set, debiased polarization vectors will be used to
            make the DS9 region file.
        """
        # Name of the pipeline reduction step
        self.name = 'region'
        self.description = 'Apply Quality Cuts'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'reg'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['skip', 4,
                               'Only plot every ith pixel'])
        self.paramlist.append(['offset', [0, 0],
                               'Offset in pixels in x, y (controls which '
                               'pixels are extracted)'])
        self.paramlist.append(['mini', 0.005,
                               'Do not plot vectors with flux < this '
                               'fraction of peak flux'])
        self.paramlist.append(['minp', 0.3,
                               'Require percentage polarizations '
                               'to be >= this value'])
        self.paramlist.append(['sigma', 3.0,
                               'p/sigma_p must be >= this value'])
        self.paramlist.append(['minisigi', 10.0,
                               'StokesI/ErrorI must be above this value'])
        self.paramlist.append(['maxp', 50.0,
                               'Pol. Degree must be below this value'])
        self.paramlist.append(['scale', True,
                               'Set to False to make all vectors '
                               'the same length in DS9'])
        self.paramlist.append(['length', 2.5,
                               'Scale factor for length of polarization '
                               'vectors in pixels in DS9'])
        self.paramlist.append(['rotate', False,
                               'Use rotated (B-Field) vectors in DS9'])
        self.paramlist.append(['debias', True,
                               'Use debiased polarizations in DS9'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Apply quality cuts determined by parameters.
        2. Store the data in a FINAL POL DATA table.
        3. Save a DS9 region file to disk.
        """
        self.auxout = []
        self.dataout = self.datain.copy()
        nhwp = self.dataout.getheadval('nhwp')

        if nhwp == 1:
            log.info('Only 1 HWP, so skipping step %s' % self.name)
        else:
            poldata = self.dataout.tableget('POL DATA')
            skip = self.getarg('skip')
            rotflag = self.getarg('rotate')
            scaleflag = self.getarg('scale')
            debiasflag = self.getarg('debias')
            mini = self.getarg('mini')
            minp = self.getarg('minp')
            offset = self.getarg('offset')
            length = self.getarg('length')
            sigma = self.getarg('sigma')
            minisigi = self.getarg('minisigi')
            maxp = self.getarg('maxp')
            stokes_i = self.dataout.imageget('STOKES I')
            estokes_i = self.dataout.imageget('ERROR I')

            # Save data selection cuts in dataout header
            self.dataout.setheadval('CUTMINI', mini,
                                    'Select pol data with '
                                    'flux/peak(flux) > this value')
            self.dataout.setheadval('CUTMINP', minp,
                                    'Select pol pct >= this value')
            self.dataout.setheadval('CUTPSIGP', sigma,
                                    'Select p/sigmap >= this value')
            self.dataout.setheadval('CUTISIGI', minisigi,
                                    'Select I/eI >= this value')
            self.dataout.setheadval('CUTMAXP', maxp,
                                    'Select pct pol <= this value')

            # clip data to correspond to offset and skip parameters
            # subtract 1 because pixels start counting at 1, not zero
            mask = where((poldata['PIXEL X'] - 1) % skip == offset[0])
            poldata = poldata[mask]
            mask = where((poldata['PIXEL Y'] - 1) % skip == offset[1])
            poldata = poldata[mask]

            # clip based on I, ensuring it is well-detected.
            xpix = poldata['PIXEL X'] - 1
            ypix = poldata['PIXEL Y'] - 1
            if mini > 0:
                # maximum flux in map
                maxi = nanmax(stokes_i)
                flux = stokes_i[ypix, xpix].flatten()
                mask = where(flux >= mini * maxi)
                poldata = poldata[mask]
                xpix = xpix[mask]
                ypix = ypix[mask]

            if debiasflag:
                pol = poldata['debiased percent pol']
            else:
                pol = poldata['percent pol']
            dpol = poldata['err. percent pol']

            if rotflag:
                angle = poldata['rotated theta']
            else:
                angle = poldata['theta']

            # clip ensuring p >= minp
            if minp > 0:
                with np.errstate(invalid='ignore'):
                    mask = where(pol >= minp)
                pol = pol[mask]
                angle = angle[mask]
                xpix = xpix[mask]
                ypix = ypix[mask]
                poldata = poldata[mask]

            # clip on sigma
            if sigma > 0:
                with np.errstate(invalid='ignore'):
                    mask = where(pol / dpol >= sigma)
                pol = pol[mask]
                angle = angle[mask]
                xpix = xpix[mask]
                ypix = ypix[mask]
                poldata = poldata[mask]

            # clip on maxp
            if maxp > 0:
                with np.errstate(invalid='ignore'):
                    mask = where(pol <= maxp)
                pol = pol[mask]
                angle = angle[mask]
                xpix = xpix[mask]
                ypix = ypix[mask]
                poldata = poldata[mask]

            # clip on minisigi
            if minisigi > 0:
                flux = stokes_i[ypix, xpix].flatten()
                eflux = estokes_i[ypix, xpix].flatten()
                with np.errstate(invalid='ignore'):
                    mask = where(flux / eflux >= minisigi)
                pol = pol[mask]
                angle = angle[mask]
                xpix = xpix[mask]
                ypix = ypix[mask]
                poldata = poldata[mask]

            # scale vectors
            if scaleflag is False:
                # set all vectors to same length
                pol = ones(angle.shape[0])
            else:
                # change % pol to fractional pol
                length = length / 100.
            pol = length * pol

            self.write_ds9(xpix, ypix, pol, angle)

            # Create final pol. table with data filtered by stepregion
            if len(poldata) > 0:
                cols = [fits.Column(name="Right Ascension", format='D',
                                    array=poldata['Right Ascension'],
                                    unit='deg'),
                        fits.Column(name="Declination", format='D',
                                    array=poldata['Declination'],
                                    unit='deg'),
                        fits.Column(name="Percent Pol", format='D',
                                    array=poldata['Percent Pol']),
                        fits.Column(name="Debiased Percent Pol", format='D',
                                    array=poldata['Debiased Percent Pol']),
                        fits.Column(name="Err. Percent Pol", format='D',
                                    array=poldata['Err. Percent Pol']),
                        fits.Column(name="Theta", format='D', unit='deg',
                                    array=poldata['Theta']),
                        fits.Column(name="Rotated Theta",
                                    format='D', unit='deg',
                                    array=poldata['Rotated Theta']),
                        fits.Column(name="Err. Theta", format='D', unit='deg',
                                    array=poldata['Err. Theta'])]
                c = fits.ColDefs(cols)
                tbhdu = fits.BinTableHDU.from_columns(c)
                self.dataout.tableset(tbhdu.data, "FINAL POL DATA",
                                      tbhdu.header)

            # Save total number of vectors found after cuts
            self.dataout.setheadval('NVECCUT', len(poldata),
                                    'Total number of vectors after cuts')

    def write_ds9(self, xpix, ypix, pol, angle):
        """
        Write output file with polarization data in DS9 region format.

        Parameters
        ----------
        xpix : array-like
            X pixel values.
        ypix : array-like
            Y pixel values.
        pol : array-like
            Polarization percentage values.
        angle : array-like
            Polarization angle values.
        """

        # determine output file name
        regionfile = self.datain.filenamebegin + self.procname.upper() + \
            self.datain.filenameend
        regionfile = regionfile.replace(".fits", ".reg")

        # Add +90 because DS9 references angles from +x axis
        angle = angle + 90.0

        log.info("Number of polarization vectors: %d" % angle.shape[0])
        # DS9 draws polarization vectors starting from the tail, so we have
        # to do some math to move coordinates from the centers to the tails
        # add +1 because DS9 counts pixels starting from 1.
        xpix = xpix - 0.5 * pol * cos(pi * angle / 180.) + 1
        ypix = ypix - 0.5 * pol * sin(pi * angle / 180.) + 1

        fp = open(regionfile, 'w')
        # Write out header
        fp.write("# Region file format: DS9 version 4.1\n")
        fp.write("global color=green dashlist=8 3 width=1 ")
        fp.write('font="helvetica 10 normal roman" select=1 highlite=1 ')
        fp.write("dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        fp.write("image\n")

        for i in range(angle.shape[0]):
            fp.write("# vector(%.2f,%.2f,%.2f,%.2f) vector=0\n" %
                     (xpix[i], ypix[i], pol[i], angle[i]))
        fp.close()
        log.info('Saved result %s' % regionfile)
        self.auxout.append(regionfile)

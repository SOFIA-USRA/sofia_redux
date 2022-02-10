# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""WCS registration pipeline step."""

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepWcs']


class StepWcs(StepParent):
    """
    Add world coordinate system definitions.

    This pipeline step will add WCS keywords to the header, accounting for
    vertical position angle (VPA) and position offsets. The reference
    position is taken to be the boresight pixel position (SIBS_X, SIBS_Y) with
    RA and Dec coordinates taken from TELRA and TELDEC.

    The following header keywords are set by this step:

        - TEL_AZ : telescope azimuth
        - TEL_ELEV : telescope elevation
        - VPOS_ANG : array VPA
        - CROTA2 : image rotation, relative to North (set from -1 * VPA)
        - CRVAL1 : boresight RA (set from TELRA)
        - CRVAL2 : boresight Dec (set from TELDEC)
        - CRPIX1 : boresight x position (set from SIBS_X)
        - CRPIX2 : boresight y position (set from SIBS_Y)
        - CDELT1 : pixel scale (set from PIXSCAL)
        - CDELT2 : pixel scale (set from PIXSCAL)
        - CTYPE1 : WCS projection (set to 'RA---TAN')
        - CTYPE2 : WCS projection (set to 'DEC--TAN')

    Input for this step is a DataFits containing Stokes images and
    associated errors, as produced by
    `sofia_redux.instruments.hawc.steps.StepStokes`.

    Output is the same as the input file, with the above header keywords
    set in the primary header.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'wcs', and are named with
        the step abbreviation 'WCS'.

        Parameters defined for this step are:

        add180vpa : bool
            If set, 180 degrees is added to the VPA derived from
            the 'Array VPA' data column, before setting VPOS_ANG.
        offsibs_x : list of float
            Offset in pixels to add to SIBS_X to correct for
            actual target position. One value for each HAWC waveband.
        offsibs_y : list of float
            Offset in pixels to add to SIBS_Y to correct for
            actual target position. One value for each HAWC waveband.
        labmode : bool
            If set, telescope data will be ignored and placeholder
            astrometry values will be added to the header.
        """
        # Name of the pipeline reduction step
        self.name = 'wcs'
        self.description = 'Add WCS'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'wcs'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['add180vpa', True,
                               'Add 180 degrees to the SIBS_VPA'])
        self.paramlist.append(['offsibs_x', [0.0, 0.0, 0.0, 0.0, 0.0],
                               'Small offset (in pixels along X) between '
                               'SIBS_X and actual target position'])
        self.paramlist.append(['offsibs_y', [0.0, 0.0, 0.0, 0.0, 0.0],
                               'Small offset (in pixels along Y) between '
                               'SIBS_Y and actual target position'])
        self.paramlist.append(['labmode', False,
                               'If labmode = True, will ignore keywords '
                               'and input parameters and create fake '
                               'astrometry'])

    def read_sibs(self):
        """
        Read an offset value from the parameters.

        The parameters are expected to be defined as a list, with
        one entry for each HAWC band. The correct value for the
        input data is selected from the list.

        Returns
        -------
        float
            The SIBS offset value.
        """
        offsibs_x = self.getarg('offsibs_x')
        offsibs_y = self.getarg('offsibs_y')

        waveband = self.datain.getheadval('spectel1')
        bands = ['A', 'B', 'C', 'D', 'E']
        try:
            idx = bands.index(waveband[-1])
        except (ValueError, IndexError):
            # waveband not in list
            msg = 'Cannot parse waveband: %s' % waveband
            log.error(msg)
            raise ValueError(msg)
        try:
            offsibs_x = offsibs_x[idx]
            offsibs_y = offsibs_y[idx]
        except IndexError:
            msg = 'Need offsibs_x/y values for all wavebands'
            log.error(msg)
            raise IndexError(msg)

        return offsibs_x, offsibs_y

    def compute_wcs(self):
        """
        Compute WCS parameters from telescope data.

        Corrects for chop offsets between SIBS and source
        position.

        Returns
        -------
        crval1, crval2, crpix1, crpix2 : tuple of float
            WCS reference position values.
        """
        # Assign keyword values of TELRA and TELDEC to
        # crval1/2. These consist in the RA/DEC position
        # of the chopper midpoint (relative to the SI
        # boresight), and already contain dithering offsets.

        # convert RA from hours to degrees
        crval1 = self.dataout.getheadval("TELRA") * 15.0
        crval2 = self.dataout.getheadval("TELDEC")

        # SIBS_X and SIBS_Y are SOFIA keywords for
        # the boresight pixel in the array.
        # Here we are assuming any transformation
        # needed between the logical coordinate
        # system and the index coordinate system has
        # already been done before (in StepShift).
        # Offsibs_x/y account for small offsets between
        # the SIBS and the actual target position
        # The "+1" is to account for the difference
        # in the SIBS array coordinate system
        # (1st pixel is 0, 0) and the CRPIX convention
        # (1st pixel is 1, 1)
        offsibs_x, offsibs_y = self.read_sibs()
        crpix1 = self.dataout.getheadval("SIBS_X") + offsibs_x + 1.0
        crpix2 = self.dataout.getheadval("SIBS_Y") + offsibs_y + 1.0

        # TelRA and TelDEC refer to the midpoint
        # of the chopper, while the SIBS coincides
        # with the minus beam (in nod A). That means a correction
        # needs to be made to match crpix1/2 to crval1/2.
        # If coord_sys = SIRF, procedure below will correct
        # the values of crpix1/2
        # If coord_sys = ERF, procedure below will correct
        # the values of crval1/2
        coord_sys = self.dataout.getheadval("CHPCRSYS")
        chpamp = self.dataout.getheadval("CHPAMP1")
        chpangle = self.dataout.getheadval("CHPANGLE")
        pixscale_arcsec = self.dataout.getheadval("pixscal")
        if coord_sys == 'sirf':
            # if SIRF, we move the X, Y minus beam
            # pixel to the chopper midpoint pixel
            crpix1 = crpix1 - (chpamp / pixscale_arcsec) * \
                np.sin(np.radians(chpangle))
            crpix2 = crpix2 + (chpamp / pixscale_arcsec) * \
                np.cos(np.radians(chpangle))
        elif coord_sys == 'erf':
            # if ERF, we move the RA/DEC chopper midpoint
            # position to the minus beam position
            corr_dec = np.cos(np.radians(crval2))
            chpamp_deg = chpamp / 3600.
            crval1 = crval1 - (chpamp_deg / corr_dec) * \
                np.sin(np.radians(chpangle))
            crval2 = crval2 + chpamp_deg * \
                np.cos(np.radians(chpangle))
        else:
            msg = '%s is not a valid chopping coordinate ' \
                  'system (CHPCRSYS header keyword)' % coord_sys
            log.warning(msg)
            raise ValueError(msg)

        # NOTE (by NLC): I'm making an assumption here that the shift is small
        # enough that cdelt1 and cdelt2 don't change. I made a test file and
        # shifted crval1, crval2 by 1 degree, and the change in cdelt appeared
        # to be 0.03 arcseconds, so I think this assumption is okay.

        return crval1, crval2, crpix1, crpix2

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Compute the VPA from the 'Array VPA' data column.
           Add 180 degrees if necessary.
        2. Compute reference positions from SIBS boresight
           and chop offset information.
        3. Store header keywords.
        """
        self.dataout = self.datain.copy()
        labmode = self.getarg('labmode')

        if not labmode:
            # real data, astrometry relies on keywords and inputs
            add180vpa = self.getarg('add180vpa')

            try:
                vpa = self.dataout.table['Array VPA'][0]
            except TypeError:
                msg = 'No valid table in input data'
                log.error(msg)
                raise ValueError(msg)
            originalvpa = vpa.copy()
            if add180vpa:
                vpa = vpa + 180.
                # Keep vpa between 0 and 360
                if vpa > 360:
                    vpa = vpa - 360.
                log.debug('Added 180 deg to SIBS_VPA to '
                          'correct for rotation. Changed from '
                          '%s to %s' % (originalvpa, vpa))

            elev = self.dataout.table['Elevation'][0]

            # to get rotations according to Attila's Geometry Memo
            crota2 = -1. * vpa

            self.dataout.setheadval("TEL_AZ", self.dataout.table['Azimuth'][0],
                                    "Average Azimuth over file")
            self.dataout.setheadval("TEL_ELEV", elev,
                                    "Average Elevation over file")
            self.dataout.setheadval("VPOS_ANG", vpa,
                                    "Average Array VPA over file")
            self.dataout.setheadval("CROTA2", crota2,
                                    "Rotation angle of wcs")

            # Sets reference RA/DEC position (CRVAL)
            # as well as the coresponding pixel position (CRPIX)
            crval1, crval2, crpix1, crpix2 = self.compute_wcs()
            self.dataout.setheadval("CRVAL1", crval1,
                                    "Right Ascension of reference pixel")
            self.dataout.setheadval("CRVAL2", crval2,
                                    "Declination of reference pixel")
            self.dataout.setheadval('CRPIX1', crpix1)
            self.dataout.setheadval('CRPIX2', crpix2)

            # set pixel size
            # convert to degrees
            pixscale = self.dataout.getheadval("pixscal") / 3600.
            # to get X axis orientation according to Attila's memo
            self.dataout.setheadval("CDELT1", pixscale,
                                    "Pixel size in x at center [deg]")
            self.dataout.setheadval("CDELT2", pixscale,
                                    "Pixel size in y at center [deg]")

            # assume tangent projection
            self.dataout.setheadval('CTYPE1', 'RA---TAN', 'WCS Projection')
            self.dataout.setheadval('CTYPE2', 'DEC--TAN', 'WCS Projection')

            # now that we have a proper image, need to switch datatype
            self.dataout.setheadval('datatype', 'IMAGE')

        else:
            # lab data: create fake astrometry to make
            # sure other steps still work

            self.dataout.setheadval("TEL_AZ", 0.,
                                    "Average Azimuth over file")
            self.dataout.setheadval("TEL_ELEV", 0.,
                                    "Average Elevation over file")
            self.dataout.setheadval("VPOS_ANG", 0.,
                                    "Average Array VPA over file")
            self.dataout.setheadval("CROTA2", 0.,
                                    "Rotation angle of wcs")
            self.dataout.setheadval("CRVAL1", 1.0,
                                    "Right Ascension of reference pixel")
            self.dataout.setheadval("CRVAL2", 2.0,
                                    "Declination of reference pixel")
            self.dataout.setheadval('CRPIX1', 31.5)
            self.dataout.setheadval('CRPIX2', 19.5)
            pixscale = self.dataout.getheadval("pixscal") / 3600.
            self.dataout.setheadval("CDELT1", pixscale,
                                    "Pixel size in x at center [deg]")
            self.dataout.setheadval("CDELT2", pixscale,
                                    "Pixel size in y at center [deg]")
            self.dataout.setheadval('CTYPE1', 'RA---TAN', 'WCS Projection')
            self.dataout.setheadval('CTYPE2', 'DEC--TAN', 'WCS Projection')
            self.dataout.setheadval('datatype', 'IMAGE')

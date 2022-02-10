# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Array alignment pipeline step."""

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.toolkit.image import adjust

__all__ = ['StepShift']


class StepShift(StepParent):
    """
    Align the R and T arrays.

    This step performs various array manipulations to trim and
    align the detector subarrays. First, the last row is removed,
    since it is unilluminated. Then, the R1 and T1 subarrays are
    rotated to align with R0 and T0, and all subarrays are flipped
    over the y-axis. The first and last chops in the demodulated
    data are removed. Finally, the R array is shifted to align with
    the T array.

    Only linear pixel shifts between R and T are performed here. Any
    additional misalignment (e.g. rotation) is handled in later
    mapping steps. The parameters passed here are set as header keywords,
    to be applied as needed by `sofia_redux.instruments.hawc.steps.StepMerge`.

    This step will also check for the TRCKSTAT keyword in the FITS header.
    If it has been set to BAD, this step will issue an error and halt
    processing.

    Input for this step is a flat-fielded file, where the R Array,
    T Array, and their variances are now images instead of columns in the
    data table. This step is typically run after
    `sofia_redux.instruments.hawc.steps.StepFlat`.

    Output from this step is the same as the input file, except that
    the R and T images, variances, and masks have been trimmed and
    shifted. Additionally, some unneeded columns are removed from the
    DEMODULATED DATA table, and header keywords are set to describe
    R and T relative geometry. These keywords are ALNANG1, ALNANG2,
    ALNMAGX, ALNMAGY, ALNGAPX, ALNGAPY, and ALNROTA.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'shift', and are named with
        the step abbreviation 'SFT'.

        Parameters defined for this step are:

        angle1 : float
            Rotation angle of R0 relative to T0, in degrees
            counter-clockwise, to be stored in keyword ALNANG1.
        angle2 : float
            Rotation angle of R1 relative to T1, in degrees
            counterclockwise, to be stored in keyword ALNANG2.
        mag : list of float
            Magnification of R relative to T, in the x, y pixel
            direction, given as [mx, my]. This will be stored
            in keywords ALNMAGX, ALNMAGY.
        disp1 : list of float
            Pixel displacement of R0 relative to T0, in the x, y
            directions, given as [dx, dy]. This will be applied
            to the data with a linearly interpolated shift.
        disp2 : list of float
            Pixel displacement of R1 relative to T1, in the x, y
            directions, given as [dx, dy]. This will be applied
            to the data with a linearly interpolated shift.
        gapx : float
            Displacement in x pixels between T0 and T1. This
            will be stored in keyword ALNGAPX.
        gapy : float
            Displacement in y pixels between T0 and T1. This
            will be stored in keyword ALNGAPY.
        gapangle : float
            Rotation angle in degrees counter-clockwise between
            T0 and T1. This will be stored in keyword ALNROTA.
        """
        # Name of the pipeline reduction step
        self.name = 'shift'
        self.description = 'Align Arrays'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'sft'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['angle1', 0.0,
                               'Rotation angle of R0 relative to T0, '
                               'in degrees counterclockwise'])
        self.paramlist.append(['angle2', 0.0,
                               'Rotation angle of R1 relative to T1, '
                               'in degrees counterclockwise'])
        self.paramlist.append(['mag', [1.0, 1.0],
                               'Magnification of R relative to T, '
                               'in the x, y pixel direction'])
        self.paramlist.append(['disp1', [0.0, 0.0],
                               "Pixel displacement of R0 relative to T0, "
                               "in the x, y directions"])
        self.paramlist.append(['disp2', [0.0, 0.0],
                               "Pixel displacement of R1 relative to T1, "
                               "in the x, y directions"])
        self.paramlist.append(['gapx', 0.0,
                               'Displacement in x pixels between T0 and T1'])
        self.paramlist.append(['gapy', 0.0,
                               'Displacement in y pixels between T0 and T1'])
        self.paramlist.append(['gapangle', 0.0,
                               'Rotation angle in degrees CCW between '
                               'T0 and T1'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Check for tracking status problems.
        2. Remove unneeded data table columns.
        3. Rotate, flip, and trim data arrays.
        4. Store alignment keywords.
        5. Shift the R array to match the T array.

        Raises
        ------
        ValueError
            If TRCKSTAT is BAD.
        """

        # Check if tracking issues were found in stepDemod
        if 'TRCKSTAT' in self.datain.header:
            trckstatus = self.datain.getheadval('TRCKSTAT')
            if "BAD" in trckstatus:
                msg = 'Bad file due to tracking issues - more ' \
                      'than 1/3 of samples were removed in stepDemod'
                log.error(msg)
                raise ValueError(msg)
        else:
            log.warning('No TRCKSTAT keyword found in the header')

        originalheader = self.datain.header

        # Remove columns
        cols_to_del = ['Chop Offset Imag',
                       'SLOPES R', 'SLOPES T',
                       'RAWMED R', 'RAWMED T',
                       'INTERCEPTS R', 'INTERCEPTS T',
                       'R ARRAY GAIN', 'T ARRAY GAIN',
                       'R ARRAY IMAG', 'T ARRAY IMAG',
                       'R ARRAY IMAG VAR', 'T ARRAY IMAG VAR']
        try:
            tabnames = self.datain.table.names
        except AttributeError:
            msg = 'No valid tables in input data'
            log.error(msg)
            raise ValueError(msg)

        for a in cols_to_del:
            for b in tabnames:
                if a == b:
                    self.datain.tabledelcol(a)

        self.dataout = self.datain.copy()
        # Set the header equal to the input header
        self.dataout.setheader(originalheader)

        # Rotate R1, T1, Flip all arrays
        for datname in ['R array', 'T array',
                        'R array VAR', 'T array VAR',
                        'R BAD PIXEL MASK', 'T BAD PIXEL MASK']:
            data = self.dataout.imageget(datname)
            nrow, ncol = data.shape[-2:]
            # Cut by 1 row (if size longer than 40) and do flips and twists
            # (A flip and rotate is just a flip so
            # each half only needs one flip)
            if len(data.shape) > 2:
                if nrow == 41:
                    data = data[:, :40, :]
                    nrow -= 1
                data[..., :ncol // 2] = \
                    data[:, range(nrow - 1, -1, -1), :ncol // 2]
                data[..., ncol // 2:ncol] = \
                    data[..., range(ncol - 1, ncol // 2 - 1, -1)]
            else:
                if nrow == 41:
                    data = data[:40, :]
                    nrow -= 1
                data[:, :ncol // 2] = \
                    data[range(nrow - 1, -1, -1), :ncol // 2]
                data[:, ncol // 2:ncol] = \
                    data[:, range(ncol - 1, ncol // 2 - 1, -1)]
            self.dataout.imageset(data, datname)

        # Remove first and last chops from images and table
        # (usually produced by demodulation artifacts)
        for datname in ['R array', 'T array', 'R array VAR', 'T array VAR']:
            data = self.dataout.imageget(datname)
            data = data[1:-1, :, :]
            self.dataout.imageset(data, datname)
        table = self.dataout.tableget('demodulated data')
        table = table[1:-1]
        self.dataout.tableset(table, 'demodulated data')

        # Get values and arrays
        disp1 = self.getarg('disp1')
        disp2 = self.getarg('disp2')
        mag1 = self.getarg('mag')
        angle1 = self.getarg('angle1')
        angle2 = self.getarg('angle2')
        gapx = self.getarg('gapx')
        gapy = self.getarg('gapy')
        gaprot = self.getarg('gapangle')
        rmask = self.dataout.imageget('R BAD PIXEL MASK')
        demodr = self.dataout.imageget('R array')
        demodrv = self.dataout.imageget('R array VAR')

        # write to fits header alignment info
        self.dataout.setheadval("ALNANG1", angle1,
                                'Rotation angle in degrees of R1 wrt to T1')
        self.dataout.setheadval("ALNANG2", angle2,
                                'Rotation angle in degrees of R2 wrt to T2')
        self.dataout.setheadval("ALNMAGX", mag1[0],
                                'Mag. of R wrt to T in the X direction')
        self.dataout.setheadval("ALNMAGY", mag1[1],
                                'Mag. of R wrt to T in the Y direction')
        self.dataout.setheadval("ALNGAPX", gapx,
                                'displacement in x pixels between T1 and T2')
        self.dataout.setheadval("ALNGAPY", gapy,
                                'displacement in y pixels between T1 and T2')
        self.dataout.setheadval("ALNROTA", gaprot,
                                'Rotation angle in degrees CCW '
                                'between T1 and T2')

        # Note, we assume the R and T arrays are the same size and that the
        # arrays are cleanly divisible by 2 in number of columns
        nz, nrow, ncol = demodr.shape
        ncol = ncol // 2

        # Apply shifts to R1 relative to T1
        ix1 = disp1[0]
        iy1 = disp1[1]
        datar1 = demodr[:, :, :ncol]
        datarv1 = demodrv[:, :, :ncol]
        maskr1 = rmask[:, :ncol]
        # must be tenth of a pixel or more to bother
        shifted = False
        if not np.allclose([ix1, iy1], 0, atol=1e-2):
            datar1 = adjust.shift(datar1, [0, iy1, ix1])
            datarv1 = adjust.shift(datarv1, [0, iy1, ix1])
            maskr1 = adjust.shift(maskr1, [iy1, ix1], mode='constant',
                                  missing=4, order=0)
            shifted = True

        # Apply shifts to R2 relative to T2
        ix2 = disp2[0]
        iy2 = disp2[1]
        datar2 = demodr[:, :, ncol:]
        datarv2 = demodrv[:, :, ncol:]
        maskr2 = rmask[:, ncol:]
        if not np.allclose([ix2, iy2], 0, atol=1e-2):
            datar2 = adjust.shift(datar2, [0, iy2, ix2])
            datarv2 = adjust.shift(datarv2, [0, iy2, ix2])
            maskr2 = adjust.shift(maskr2, [iy2, ix2], mode='constant',
                                  missing=4, order=0)
            shifted = True

        if shifted:
            # some sort of shift applied
            log.debug('Shifted R array to match T array')

            # combine demodulated data
            newdata = np.concatenate((datar1, datar2), axis=2)

            # combine variance
            newvar = np.concatenate((datarv1, datarv2), axis=2)

            # combine r1 & r2 masks
            newmask = np.concatenate((maskr1, maskr2), axis=1)

            # write out updated data
            self.dataout.imageset(newdata, 'R Array')
            self.dataout.imageset(newvar, 'R Array VAR')
            self.dataout.imageset(newmask, 'R BAD PIXEL MASK')

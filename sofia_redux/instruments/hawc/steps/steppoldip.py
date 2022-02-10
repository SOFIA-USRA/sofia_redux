# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Polarization calibration pipeline step."""

import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepPolDip']


class StepPolDip(StepParent):
    r"""
    Reduce polarized sky dips for instrumental polarization calibration.

    This pipeline step derives the instrumental polarization (q and u)
    from an unchopped, polarized sky dip observation.  This step should
    be run after `sofia_redux.instruments.hawc.steps.StepPrepare`.  The
    output is a DataFits object with several image extensions, used for
    diagnostic purposes. The instrumental polarization fraction is stored
    in the 'q (instrumental)' and 'u (instrumental)' images.

    Notes
    -----
    The flux is fit with this signal model:

       .. math:: F = C + Q_0 cos(4 HWP') + U_0 sin(4 HWP')
       .. math::       + V_0 cos(2 HWP') + W_0 sin(2 HWP')
       .. math::       + G [1 + q\ cos(4 HWP') + u\ sin(4 HWP')] AM
       .. math::       + D (T - T_0)

    The following parameters from the model fit are stored in the output:
       - :math:`C` : FB constant offset
       - :math:`Q_0` : Q0 (background)
       - :math:`U_0` : U0 (background)
       - :math:`G` : radiation gain
       - :math:`D` : thermal gain
       - :math:`q` : q (instrumental)
       - :math:`u` : u (instrumental)

    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'poldip', and are named with
        the step abbreviation 'PDP'.

        Parameters defined for this step are:

        hwp0 : float
            Reference HWP angle.
        temp0 : float
            Reference temperature (ADR set point).
        maxrms : float
            Maximum allowed RMS.  Pixels with fit RMS values higher
            than this threshold are set to NaN.
        """

        # Name of the pipeline reduction step
        self.name = 'poldip'
        self.description = 'Make IP Calibration'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'pdp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['hwp0', 5.0,
                               'Reference HWP angle'])
        self.paramlist.append(['temp0', 0.532,
                               'Reference temperature (ADR setpoint)'])
        self.paramlist.append(['maxrms', 0.1,
                               'Maximum allowed reduced RMS'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Fit the input flux at each pixel with a parametrized model,
           including the instrumental polarization terms.
        2. Store the fit parameters as images in the output file.
        """

        # set up variables
        hwp0 = self.getarg('hwp0')
        temp0 = self.getarg('temp0')
        maxrms = self.getarg('maxrms')

        # make new output data
        self.dataout = DataFits(config=self.config)
        self.dataout.filename = self.datain.filename
        self.dataout.header = self.datain.header.copy()

        # get raw flux data and reshape it
        r_data = self.datain.table['R array']
        t_data = self.datain.table['T array']
        f_data = np.concatenate([r_data, t_data], axis=2)

        shape = f_data.shape
        img_shape = (shape[1], shape[2])
        nsample = shape[0]
        npix = img_shape[0] * img_shape[1]

        f = f_data.reshape((nsample, npix)).transpose()

        # get HWP angles, elevation, and temperature data
        hwp_data = self.datain.table['hwpCounts']
        el_data = self.datain.table['Elevation']
        dtemp_data = self.datain.table['ai23']

        # signal model:
        #  FB = C + Q0*cos(4*HWP') + U0*sin(4*HWP')
        #         + V0*cos(2*HWP') + W0*sin(2*HWP')
        #         + G*{1 + q*cos(4*HWP') + u*sin(4*HWP')}*AM
        #         + D*(T-T0)

        # calculate approximate airmass
        am_data = 1 / np.sin(el_data * np.pi / 180.)

        # subtract ADR set point from temperature
        dtemp_data -= temp0

        # set up some useful quantities from HWP angle
        # (each have dimension nsample)
        c4h = np.cos(4.0 * (hwp_data / 4.0 - hwp0) * np.pi / 180.)
        s4h = np.sin(4.0 * (hwp_data / 4.0 - hwp0) * np.pi / 180.)
        c2h = np.cos(2.0 * (hwp_data / 4.0 - hwp0) * np.pi / 180.)
        s2h = np.sin(2.0 * (hwp_data / 4.0 - hwp0) * np.pi / 180.)

        # constant term
        one = np.ones_like(c4h)

        # set up matrices for linear equations
        # (c is 9 x nsample)
        c = np.array([one, c4h, s4h,
                      am_data, c4h * am_data, s4h * am_data,
                      dtemp_data, c2h, s2h])

        # a is the same for all pixels
        # (a is 9 x 9 matrix)
        a = c.dot(c.T)

        # vector varies by pixel
        # (b is npix x 9 vector)
        b = f.dot(c.T)

        # copy of a for each pixel
        # (aa is npix x 9 x 9)
        aa = np.tile(a, (npix, 1, 1))

        # solve equations
        # returns npix solutions, with 9 parameters each
        solution = np.linalg.solve(aa, b)

        # extract desired constants from solution
        # (each have dimension npix)
        c = solution[:, 0]
        q0 = solution[:, 1]
        u0 = solution[:, 2]
        g = solution[:, 3]
        with np.errstate(invalid='ignore'):
            q = solution[:, 4] / g
            u = solution[:, 5] / g
        d = solution[:, 6]
        v0 = solution[:, 7]
        w0 = solution[:, 8]

        # flux model
        # (model is npix x nsample, same dimensions as f)
        model = np.outer(c, one) + \
            np.outer(q0, c4h) + np.outer(u0, s4h) + \
            np.outer(v0, c2h) + np.outer(w0, s2h) + \
            np.outer(g, am_data) + \
            np.outer(q * g, c4h * am_data) + \
            np.outer(u * g, s4h * am_data) + \
            np.outer(d, dtemp_data)

        # RMS: sqrt of sum over time series of (data - model)^2,
        #   over sqrt(N)
        rms = np.sqrt(np.sum((f - model)**2, axis=1) / nsample)

        # reduced RMS: divide by absolute value of radiation gain
        reduced_rms = rms / np.abs(g)

        # mask deviant data
        with np.errstate(invalid='ignore'):
            idx = (reduced_rms > maxrms)
        q[idx] = np.nan
        u[idx] = np.nan

        # save images in dataout
        dataset = [c.transpose().reshape(img_shape),
                   q0.transpose().reshape(img_shape),
                   u0.transpose().reshape(img_shape),
                   g.transpose().reshape(img_shape),
                   d.transpose().reshape(img_shape),
                   q.transpose().reshape(img_shape),
                   u.transpose().reshape(img_shape),
                   rms.transpose().reshape(img_shape),
                   reduced_rms.transpose().reshape(img_shape),
                   nsample + np.zeros(img_shape)]
        datanames = ['FB constant offset',
                     'Q0 (background)', 'U0 (background)',
                     'radiation gain', 'thermal gain',
                     'q (instrumental)', 'u (instrumental)',
                     'RMS', 'RMS/radiation gain',
                     'number of samples']
        dataunits = ['FB units',
                     'FB units', 'FB units',
                     'FB units/airmass', 'FB units/thermometer units',
                     'dimensionless fraction', 'dimensionless fraction',
                     'FB units', 'dimensionless fraction',
                     'dimensionless number']

        for i, data in enumerate(dataset):
            self.dataout.imageset(data, datanames[i])
            self.dataout.setheadval('BUNIT', dataunits[i],
                                    comment='Data units',
                                    dataname=datanames[i])

        # update SOFIA mandated keywords (since this is first pipe step)
        obsid = 'P_' + self.dataout.getheadval('OBS_ID')
        self.dataout.setheadval('OBS_ID', obsid)
        self.dataout.setheadval('PROCSTAT', 'LEVEL_2')
        self.dataout.setheadval('PIPELINE', 'HAWC_DRP',
                                'Data processing pipeline')

        # set ASSC_AOR and ASSC_MSN value in output header
        try:
            self.dataout.setheadval('ASSC_AOR',
                                    self.dataout.getheadval('AOR_ID'),
                                    'Associated AORs')
        except KeyError:
            pass
        try:
            self.dataout.setheadval('ASSC_MSN',
                                    self.dataout.getheadval('MISSN-ID'),
                                    'Associated Mission IDs')
        except KeyError:
            pass

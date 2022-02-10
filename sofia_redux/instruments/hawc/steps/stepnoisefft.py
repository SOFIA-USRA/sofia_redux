# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Noise FFT pipeline step."""

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepNoiseFFT']


class StepNoiseFFT(StepParent):
    """
    Take the FFT of diagnostic noise data.

    The input to this step is lab data taken with CALMODE = 'NOISE'.
    The output is a FITS file containing the power spectrum image
    with pixels arrayed along the y-axis and frequencies arrayed
    along the x-axis.  Frequency values are stored in WCS keywords
    in the header. The full FFT with linear frequencies is stored
    in the primary image.  Subsequent extensions contain binned
    frequencies, in linear and log scales.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'noisefft', and are named with
        the step abbreviation 'FFT'.

        Parameters defined for this step are:

        truncate : bool
            If set, will truncate to an integer power of 2 number of
            samples before taking the FFT.

        """
        # Name of the pipeline reduction step
        self.name = 'noisefft'
        self.description = 'Noise FFT'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'fft'

        # Clear parameter list
        self.paramlist = []

        # add default parameters
        self.paramlist.append(['truncate', False,
                               'Truncate time samples'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Convert the flux data to Amps/rtHz.
        2. Take the FFT of the flux data for each pixel.
        3. Bin the frequencies and average flux values within
           each bin.
        4. Store the data.

        """
        # get arguments
        truncate = self.getarg('truncate')

        # make new output data
        self.dataout = DataFits(config=self.config)
        self.dataout.filename = self.datain.filename
        self.dataout.header = self.datain.header.copy()

        # get raw flux data and reshape it
        r_data = self.datain.table['R array']
        t_data = self.datain.table['T array']
        f_data = np.concatenate([r_data, t_data], axis=2)

        nsamp, nrow, ncol = f_data.shape
        npix = nrow * ncol

        # magic numbers from hawcp_powspec IDL script,
        # originally developed by J. Vaillancourt, 2011

        # use value from IV curves
        dac2squid = 12600.  # counts / phi0
        coil2squid = 3e-6  # Amp / phi0
        inv_gain = coil2squid / dac2squid  # Amp / count

        # remove bottom 4 bits
        inv_gain /= 4096.

        # invert for proper gain
        gain = 1.0 / inv_gain  # counts / Amp

        # unfiltered
        gain /= 3.361

        # convert counts to Amps
        acdata = f_data / gain

        # sample rate from header
        samprate = self.dataout.getheadval('SMPLFREQ')

        # time in seconds from first sample
        time = np.arange(nsamp, dtype=float) / samprate
        log.info(f'Total record time in file: {time[-1]} seconds')

        # reshape data to 2D
        acdata = acdata.reshape(nsamp, npix)

        # truncate to integer power of 2 number of samples
        # This was originally intended to make FFT more efficient
        # in the IDL implementation, but numpy FFT is plenty fast,
        # so it may no longer be needed.
        if truncate:
            tr = int(np.log(nsamp) / np.log(2))
            nsamp = 2 ** tr
            acdata = acdata[:nsamp, :]
            time = time[:nsamp]
            log.info(f'Truncating total time record to {time[-1]}')

        # frequency vector
        fmax = samprate / 2.0
        df = samprate / nsamp
        fmin = 10.0 * df
        frange = [fmin, fmax]
        freq = np.arange(nsamp, dtype=float) * df

        # hanning window with discrete correction
        h = 0.5 * (1 - np.cos(2 * np.pi
                              * (np.arange(nsamp, dtype=float) + 1.) / nsamp))
        wss = (1. / nsamp) * np.sum(h ** 2)

        # fft of samples for all channels, matching IDL norm convention
        ft = np.fft.fft(h[:, None] * acdata, axis=0, norm='forward')

        # linear frequency fft
        psd = np.abs(ft) / np.sqrt(wss * df) * np.sqrt(2)
        nsamp2 = nsamp // 2
        psd = psd[0:nsamp2, :]
        freq = freq[0:nsamp2]

        # bin linear frequencies
        nbin = 512
        hist, edges = np.histogram(freq, bins=nbin, range=frange)
        freq2 = np.empty(nbin)
        signal = np.empty((nbin, npix))
        for i in range(nbin):
            freq2[i] = (edges[i] + edges[i + 1]) / 2
            f_ind = (freq >= edges[i]) & (freq < edges[i + 1])
            signal[i, :] = np.sum(psd[f_ind, :], axis=0) / np.sum(f_ind)

        # bin log frequencies
        logf_unbin = np.log10(freq)
        hist, edges = np.histogram(logf_unbin, bins=nbin,
                                   range=np.log10(frange))
        logfreq = np.empty(nbin)
        logsignal = np.empty((nbin, npix))
        for i in range(nbin):
            logfreq[i] = (edges[i] + edges[i + 1]) / 2
            f_ind = (logf_unbin >= edges[i]) & (logf_unbin < edges[i + 1])
            logsignal[i, :] = np.sum(psd[f_ind, :], axis=0) / np.sum(f_ind)

        # take the log of the signal too
        loglogsignal = np.log10(logsignal)
        loglogsignal[~np.isfinite(logsignal)] = np.nan

        # save images in dataout

        # linear frequency x pixels
        psd_header = fits.Header({
            'EXTNAME': 'FULL_FFT',
            'CTYPE1': 'FREQ', 'CTYPE2': 'Pixels',
            'CUNIT1': 'Hz', 'CUNIT2': 'pixel',
            'CRPIX1': 1, 'CRPIX2': 1,
            'CRVAL1': freq[0], 'CRVAL2': 1,
            'CDELT1': df, 'CDELT2': 1,
            'BUNIT': 'Amps/rtHz'})
        self.dataout.header.update(psd_header)
        self.dataout.imageset(psd.T, imagename='FULL_FFT')

        # binned linear frequency x pixels
        signal_header = fits.Header({
            'EXTNAME': 'LINEARFREQ',
            'CTYPE1': 'FREQ', 'CTYPE2': 'Pixels',
            'CUNIT1': 'Hz', 'CUNIT2': 'pixel',
            'CRPIX1': 1, 'CRPIX2': 1,
            'CRVAL1': freq2[0], 'CRVAL2': 1,
            'CDELT1': freq2[1] - freq2[0], 'CDELT2': 1,
            'BUNIT': 'Amps/rtHz'})
        self.dataout.imageset(signal.T, imagename='LINEARFREQ',
                              imageheader=signal_header)

        # binned log frequency x pixels
        logsignal_header = fits.Header({
            'EXTNAME': 'LOGFREQ',
            'CTYPE1': 'FREQ-LOG', 'CTYPE2': 'Pixels',
            'CUNIT1': 'Hz', 'CUNIT2': 'pixel',
            'CRPIX1': 1, 'CRPIX2': 1,
            'CRVAL1': logfreq[0], 'CRVAL2': 1,
            'CDELT1': logfreq[1] - logfreq[0], 'CDELT2': 1,
            'BUNIT': 'Amps/rtHz'})
        self.dataout.imageset(logsignal.T, imagename='LOGFREQ',
                              imageheader=logsignal_header)

        # binned log signal, log freq
        loglogsignal_header = fits.Header({
            'EXTNAME': 'LOGFREQ_LOGSIGNAL',
            'CTYPE1': 'FREQ-LOG', 'CTYPE2': 'Pixels',
            'CUNIT1': 'Hz', 'CUNIT2': 'pixel',
            'CRPIX1': 1, 'CRPIX2': 1,
            'CRVAL1': logfreq[0], 'CRVAL2': 1,
            'CDELT1': logfreq[1] - logfreq[0], 'CDELT2': 1,
            'BUNIT': 'log(Amps/rtHz)'})
        self.dataout.imageset(loglogsignal.T, imagename='LOGFREQ_LOGSIGNAL',
                              imageheader=loglogsignal_header)

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
        except KeyError:  # pragma: no cover
            pass
        try:
            self.dataout.setheadval('ASSC_MSN',
                                    self.dataout.getheadval('MISSN-ID'),
                                    'Associated Mission IDs')
        except KeyError:  # pragma: no cover
            pass

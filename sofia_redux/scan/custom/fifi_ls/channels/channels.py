# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.scan.custom.sofia.channels.camera import SofiaCamera
from sofia_redux.scan.channels.modality.correlated_modality import (
    CorrelatedModality)

__all__ = ['FifiLsChannels']


class FifiLsChannels(SofiaCamera):

    def __init__(self, parent=None, info=None, size=0, name='fifi_ls'):
        """
        Initialize FIFI-LS channels.

        Parameters
        ----------
        parent : object, optional
            The owner of the channels such as a Reduction, Scan or Integration.
        info : HawcPlusInfo, optional
            The info object relating to these channels.
        size : int, optional
            The intended size of the channels (number of total data channels).
        name : str, optional
            The name for the channels.
        """
        super().__init__(name=name, parent=parent, info=info, size=size)
        self.n_store_channels = self.detector.pixels

    def copy(self):
        """
        Return a copy of the FIFI-LS channels.

        Returns
        -------
        FifiLsChannels
        """
        return super().copy()

    @property
    def detector(self):
        """
        Return the detector info.

        Returns
        -------
        FifiLsDetectorArrayInfo
        """
        return self.info.detector_array

    @property
    def pixel_sizes(self):
        """
        Return the (x,y) pixel size.

        Returns
        -------
        pixel_sizes : Coordinate2D
            The x, y pixel size in arc seconds.
        """
        return self.info.detector_array.pixel_sizes

    def init_divisions(self):
        """
        Initializes channel divisions.

        Divisions contain sets of channel groups.

        The FIFI-LS channel adds divisions consisting of groups where
        each contains a unique value of a certain data field.  For example,
        the "rows" division contains a group for row 1, a group for row 2, etc.

        Returns
        -------
        None
        """
        super().init_divisions()
        dead_blind = self.flagspace.flags.DEAD | self.flagspace.flags.BLIND

        for division_name, field in [('spexels', 'spexel'),
                                     ('spaxels', 'spaxel'),
                                     ('rows', 'row'),
                                     ('cols', 'col')]:
            self.add_division(self.get_division(
                name=division_name, field=field, discard_flag=dead_blind))

    def init_modalities(self):
        """
        Initializes channel modalities.

        A modality is based of a channel division and contains a mode for each
        channel group in the channel division.

        The FIFI-LS modalities simply contain additional correlated modes
        based on the additional channel fields.  A new coupled modality
        is also created according to polarization arrays.

        Returns
        -------
        None
        """
        super().init_modalities()

        flags = self.flagspace.flags
        builds = [('spexels', 'S', 'spexels', 'spexel_gain', flags.SPEXEL),
                  ('spaxels', 's', 'spaxels', 'spaxel_gain', flags.SPAXEL),
                  ('rows', 'r', 'rows', 'row_gain', flags.ROW),
                  ('cols', 'c', 'cols', 'col_gain', flags.COL)]

        for name, identity, division_name, gain_field, gain_flag in builds:
            division = self.divisions.get(division_name)
            if division is None:  # pragma: no cover
                log.warning(f"Channel division {division_name} not found.")
                continue
            modality = CorrelatedModality(name=name,
                                          identity=identity,
                                          channel_division=division,
                                          gain_provider=gain_field)
            modality.set_gain_flag(gain_flag)
            self.add_modality(modality)

    def load_channel_data(self):
        """
        Load the channel data.

        The channel data is read in directly from the HDU list, not a pixel
        file.

        Returns
        -------
        None
        """
        pass
        # self.detector.initialize_channel_data(self.data)
        # self.set_nominal_pixel_positions()
        # super().load_channel_data()

    def set_nominal_pixel_positions(self):
        """
        Set the channel pixel positions.

        Returns
        -------
        None
        """
        self.data.position = (
            self.info.detector_array.pixel_offsets[self.data.spaxel])

    def max_pixels(self):
        """
        Return the maximum pixels in the detector array.

        Returns
        -------
        count : int
        """
        return self.detector.pixels

    def read_data(self, hdul):
        """
        Read a FITS HDU list to populate channel data.

        Parameters
        ----------
        hdul : fits.HDUList

        Returns
        -------
        None
        """
        self.data.read_hdul(hdul)

    def get_si_pixel_size(self):
        """
        Return the science instrument pixel size

        Returns
        -------
        x, y : Coordinate2D
            The (x, y) pixel sizes
        """
        return self.detector.pixel_sizes

    def write_flat_field(self, filename, include_nonlinear=False):
        """
        Write a flat field file used for chop-nod pipelines.

        Parameters
        ----------
        filename : str
            The filename to write to.
        include_nonlinear : bool, optional
            If `True`, include the nonlinear responses.

        Returns
        -------
        None
        """
        shape = self.detector.n_spexel, self.detector.n_spaxel

        # Set defaults
        gain = np.ones(shape, dtype=float)
        flags = np.full(shape, False)
        flagged = self.data.is_flagged()

        gains = self.data.gain * self.data.coupling
        inverse_gains = np.zeros_like(gains)
        nzi = gains != 0
        inverse_gains[nzi] = 1 / gains[nzi]

        inds = self.data.spexel, self.data.spaxel
        gain[inds] = inverse_gains
        flags[inds] = flagged

        hdul = fits.HDUList()
        hdul.append(fits.ImageHDU(gain, name='Channel gain'))
        hdul.append(fits.ImageHDU(flags.astype(int), name='Bad pixel mask'))
        if include_nonlinear:
            nonlinear = np.zeros(shape, dtype=float)
            nonlinear[inds] = self.data.nonlinearity
            hdul.append(fits.ImageHDU(nonlinear, name='Channel nonlinearity'))

        hdul.writeto(filename, overwrite=True)
        hdul.close()

        log.info(f"Written flat field to {filename}.")

    def add_hdu(self, hdul, hdu, extname):
        """
        Add a FITS HDU to the HDUList.

        Parameters
        ----------
        hdul : fits.HDUList
            The HDUList to append to.
        hdu : fits.ImageHDU or fits.PrimaryHDU or fits.BinTableHDU
            The fits HDU to append.
        extname : str
            The name of the HDU extension.

        Returns
        -------
        None
        """
        hdu.header['EXTNAME'] = extname, 'image content ID'
        self.info.edit_header(hdu.header)
        hdul.append(hdu)

    def calculate_overlaps(self, point_size=None):
        """
        Calculate channel overlaps.

        Parameters
        ----------
        point_size : Coordinate2D1, optional
            The overlap point size (beam FWHM for example).  The default
            is the instrument spatial and spectral resolution.

        Returns
        -------
        None
        """
        if point_size is None:
            point_size = self.info.instrument.get_point_size()

        if point_size == self.overlap_point_size:
            return  # don't need to do anything if already calculated.

        self.data.calculate_overlaps(point_size)
        self.overlap_point_size = point_size

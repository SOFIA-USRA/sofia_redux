# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, log
import numpy as np

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.utils import to_header_float
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.grid_2d import Grid2D
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaDetectorArrayInfo']


class SofiaDetectorArrayInfo(InfoBase):

    subarrays = 0

    def __init__(self):
        super().__init__()
        self.detector_name = None
        self.detector_size_string = None
        self.pixel_size = np.nan * units.Unit('arcsec')
        self.subarray_size = None
        self.boresight_index = Coordinate2D()
        self.grid = None

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'sofscan/array'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.detector_name = options.get_string("DETECTOR")
        self.detector_size_string = options.get_string("DETSIZE")
        self.pixel_size = options.get_float("PIXSCAL") * units.Unit('arcsec')

        subarrays = options.get_int("SUBARRNO", default=0)
        if subarrays > 0:
            self.subarray_size = []
            for i in range(self.subarrays):
                key = f'SUBARR{str(i + 1).zfill(2)}'
                value = options.get_string(key)
                self.subarray_size.append(value)
        else:
            self.subarray_size = []

        self.boresight_index.x = options.get_float("SIBS_X")
        self.boresight_index.y = options.get_float("SIBS_Y")

        header = self.configuration.fits.header
        if 'CTYPE1' in header and 'CTYPE2' in header:
            try:
                self.grid = Grid2D.from_header(header, alt='')
            except Exception as err:
                log.warning(f"Could not read detector array "
                            f"grid system: {err}")
                self.grid = None
        else:
            self.grid = None

    def edit_header(self, header):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        info = [('COMMENT', "<------ SOFIA Array Data ------>"),
                ('DETECTOR', self.detector_name, 'Detector name.'),
                ('DETSIZE', self.detector_size_string, 'Detector size.'),
                ('PIXSCAL', (to_header_float(self.pixel_size, 'arcsec')),
                 '(arcsec) pixel size on sky.')]

        if 0 < self.subarrays == len(self.subarray_size):
            info.append(('SUBARRNO', self.subarrays, 'Number of subarrays.'))
            for subarray in range(self.subarrays):
                value = self.subarray_size[subarray]
                if value is not None:
                    key = f'SUBARR{str(subarray).zfill(2)}'
                    comment = f'Subarray {subarray} location and size.'
                    info.append((key, value, comment))

        if self.boresight_index is None:
            self.boresight_index = Coordinate2D()
        sx, sy = self.boresight_index.coordinates

        info.append(('SIBS_X', to_header_float(sx),
                     '(pixel) boresight pixel x.'))
        info.append(('SIBS_Y', to_header_float(sy),
                     '(pixel) boresight pixel y.'))
        insert_info_in_header(header, info, delete_special=True)

        if self.grid is not None:
            self.grid.edit_header(header)

    def get_table_entry(self, name):
        """
        Given a name, return the parameter stored in the information object.

        Note that names do not exactly match to attribute names.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        if name == 'sibsx':
            return self.boresight_index[0]
        elif name == 'sibsy':
            return self.boresight_index[1]
        else:
            return super().get_table_entry(name)

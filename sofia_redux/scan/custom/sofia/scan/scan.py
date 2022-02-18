# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import units
from astropy.io import fits
import numpy as np
import re

from sofia_redux.scan.scan.scan import Scan
from sofia_redux.scan.coordinate_systems.offset_2d import Offset2D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import to_header_float

__all__ = ['SofiaScan']


class SofiaScan(Scan):

    DEFAULT_FITS_DATE = "1970-01-01T00:00:00.0"

    def __init__(self, channels, reduction=None):
        self.hdul = None
        self.header_extension = 0
        self.header = None
        self.history = None
        super().__init__(channels, reduction=reduction)

    @property
    def referenced_attributes(self):
        """
        Return the names of attributes that are referenced during a copy.

        Returns
        -------
        attribute_names : set (str)
        """
        attributes = super().referenced_attributes
        attributes.add('hdul')
        attributes.add('header')
        return attributes

    @property
    def info(self):
        """
        Return the information object for the scan.

        The information object contains the reduction configuration and various
        parameters pertaining the this scan.

        Returns
        -------
        SofiaInfo
        """
        return super().info

    def copy(self):
        """
        Return a copy of the SofiaScan.

        Returns
        -------
        SofiaScan
        """
        return super().copy()

    @property
    def astrometry(self):
        """
        Return the scan astrometry information.

        Returns
        -------
        info : SofiaAstrometryInfo
        """
        return super().astrometry

    @staticmethod
    def get_lowest_quality(scans):
        """
        Return the lowest quality processing stats from a set of scans.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans.

        Returns
        -------
        QualityFlags.QualityFlagTypes
            The lowest quality flag type.
        """
        lowest_quality = 0
        lowest_scan = None
        for scan in scans:
            if isinstance(scan, SofiaScan):
                if scan.info.processing.quality_level.value > lowest_quality:
                    lowest_quality = scan.info.processing.quality_level.value
                    lowest_scan = scan

        if lowest_scan is not None:
            return lowest_scan.info.processing.flagspace.convert_flag(
                lowest_quality)
        else:
            return None

    @staticmethod
    def get_total_exposure_time(scans):
        """
        Return the total exposure time in a set of scans.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans.

        Returns
        -------
        exposure_time : astropy.units.Quantity
            The total exposure time from all scans in seconds.
        """
        exposure_time = 0.0 * units.Unit('s')
        if scans is None:
            return exposure_time
        for scan in scans:
            if isinstance(scan, SofiaScan):
                exposure_time += scan.info.instrument.exposure_time
        return exposure_time

    @staticmethod
    def has_tracking_error(scans):
        """
        Report whether any scan in a set contains a telescope tracking error.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans.

        Returns
        -------
        tracking_error : bool
            `True` if any scan contains a telescope tracking error.  `False`
            otherwise.
        """
        if scans is None:
            return False
        for scan in scans:
            if isinstance(scan, SofiaScan):
                if scan.info.telescope.has_tracking_error:
                    return True
        else:
            return False

    @staticmethod
    def get_earliest_scan(scans):
        """
        Return the earliest scan in a list of scans determined by MJD.

        Parameters
        ----------
        scans : list (Scan)

        Returns
        -------
        Scan
        """
        return SofiaScan.time_order_scans(scans)[0]

    @staticmethod
    def get_latest_scan(scans):
        """
        Return the latest scan in a list of scans determined by MJD.

        Parameters
        ----------
        scans : list (Scan)

        Returns
        -------
        Scan
        """
        return SofiaScan.time_order_scans(scans)[0]

    def read(self, filename, read_fully=True):
        """
        Read a filename to populate the scan.

        The read should validate the channels before instantiating integrations
        for reading.

        Parameters
        ----------
        filename : str
            The name of the file to read.
        read_fully : bool, optional
            If `True`, perform a full read (default)

        Returns
        -------
        None
        """
        self.hdul = fits.open(filename)
        self.read_hdul(self.hdul, read_fully=read_fully)
        self.close_fits()

    def close_fits(self):
        """
        Close the scan FITS file.

        Returns
        -------
        None
        """
        if self.hdul is None:
            return
        self.hdul.close()
        self.hdul = None

    def read_hdul(self, hdul, read_fully=True):
        """
        Read an open FITS HDUL.

        Parameters
        ----------
        hdul : fits.HDUList
            The FITS HDU list to read.
        read_fully : bool, optional
            If `True` (default), fully read the file.

        Returns
        -------
        None
        """
        self.info.parse_header(hdul[0].header.copy())
        self.channels.read_data(hdul)
        self.channels.validate_scan(self)
        self.integrations = []
        self.add_integrations_from_hdul(self.hdul)
        self.info.instrument.sampling_interval = \
            self[0].info.sampling_interval.copy()
        self.info.instrument.integration_time = \
            self[0].info.integration_time.copy()

    @abstractmethod
    def add_integrations_from_hdul(self, hdul):
        """
        Add integrations to this scan from a HDU list.

        Parameters
        ----------
        hdul : fits.HDUList

        Returns
        -------
        None
        """
        pass

    def is_aor_valid(self):
        """
        Checks whether the observation AOR ID is valid.

        Returns
        -------
        valid : bool
        """
        return self.info.observation.is_aor_valid()

    def is_coordinate_valid(self, coordinate):
        """
        Checks whether coordinates are valid.

        Parameters
        ----------
        coordinate : SphericalCoordinates

        Returns
        -------
        valid : bool
        """
        return self.astrometry.coordinate_valid(coordinate)

    def is_requested_valid(self, header):
        """
        Check if the requested coordinates are valid.

        Parameters
        ----------
        header : astropy.io.fits.Header

        Returns
        -------
        valid : bool
        """
        return self.astrometry.is_requested_valid(header)

    def guess_reference_coordinates(self, header=None):
        """
        Guess the reference coordinates of the scan from the header.

        Parameters
        ----------
        header : astropy.io.fits.Header, optional
            The header from which to guess the coordinates.  If not supplied,
            is read from the stored configuration.

        Returns
        -------
        coordinates : EquatorialCoordinates
        """
        return self.astrometry.guess_reference_coordinates(
            telescope=self.info.telescope, header=header)

    def edit_scan_header(self, header):
        """
        Edit scan FITS header information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to edit.

        Returns
        -------
        None
        """
        line = " ----------------------------------------------------"
        super().edit_scan_header(header)
        header['COMMENT'] = line
        header['COMMENT'] = " Section for preserved SOFIA header data"
        header['COMMENT'] = line

        if self.astrometry.file_date is not None:
            header['DATE'] = (self.astrometry.file_date,
                              'Scan file creation date.')
        if self.info.origin.checksum is not None:
            header['DATASUM'] = (self.info.origin.checksum,
                                 'Data file checksum.')
        if self.info.origin.checksum_version is not None:
            header['CHECKVER'] = (self.info.origin.checksum_version,
                                  'Checksum method version.')

        self.info.edit_header(header)

        header['COMMENT'] = line
        header['COMMENT'] = " Section for scan-specific processing history"
        header['COMMENT'] = line

        self.info.add_history(header, scans=None)

    def validate(self):
        """
        Validate the scan after a read.

        Returns
        -------
        None
        """
        if not self.configuration.get_bool('lab'):
            self.astrometry.validate_astrometry(self)
        super().validate()

    def get_telescope_vpa(self):
        """
        Return the telescope VPA.

        The value represents the midpoint of the first and last frames of the
        first and last integrations respectively.

        Returns
        -------
        angle : astropy.units.Quantity
        """
        return self.frame_midpoint_value('telescope_vpa')

    def get_instrument_vpa(self):
        """
        Return the instrument VPA.

        The value represents the midpoint of the first and last frames of the
        first and last integrations respectively.

        Returns
        -------
        angle : astropy.units.Quantity
        """
        return self.frame_midpoint_value('instrument_vpa')

    def get_id(self):
        """
        Return the scan ID.

        Returns
        -------
        str
        """
        obs_id = self.info.observation.obs_id
        if obs_id is None:
            return f'{self.info.astrometry.date}.UNKNOWN'
        elif obs_id.lower().startswith('unknown'):
            return f'{self.info.astrometry.date}.{obs_id[7:]}'
        else:
            return obs_id

    def get_pointing_data(self):
        """
        Return pointing data information.

        Returns
        -------
        data : dict
        """
        data = super().get_pointing_data()
        relative = self.get_native_pointing(self.pointing)
        si_offset = self.get_si_pixel_offset(relative)
        data['dSIBSX'] = si_offset.x
        data['dSIBSY'] = si_offset.y
        return data

    def get_flight_number(self):
        """
        Return the flight number for the scan.

        Returns
        -------
        flight : int
            Returns the flight number or -1 if not found.
        """
        mission_id = self.info.mission.mission_id
        if mission_id is None:
            return -1
        flight = re.search(r'_F(\d+)', mission_id)
        if flight is None:
            return -1
        return int(flight.groups()[0])

    def get_scan_number(self):
        """
        Return the scan number.

        Returns
        -------
        scan_number : int
            The scan number if found and -1 otherwise.
        """
        scan_number = re.search(r'-(\d+)', self.info.observation.obs_id)
        if scan_number is None:
            return -1
        return int(scan_number.groups()[-1])

    def get_table_entry(self, name):
        """
        Return a parameter value for the given name.

        Parameters
        ----------
        name : str
            The name of the parameter to retrieve.

        Returns
        -------
        value
        """
        if name == 'obstype':
            return self.info.observation.obs_type
        elif name == 'flight':
            return self.get_flight_number()
        elif name == 'scanno':
            return self.get_scan_number()
        elif name == 'date':
            return self.astrometry.date

        for group_name, group in self.info.available_info.items():
            if group is None:
                continue
            prefix = group.log_prefix
            if name.startswith(group.log_prefix):
                return group.get_table_entry(name[len(prefix):])

        return super().get_table_entry(name)

    def get_nominal_pointing_offset(self, native_pointing):
        """
        Get the nominal point offset for a native pointing coordinate.

        The nominal pointing offset ignores the reference coordinate of the
        supplied `native_coordinate` and adds the offset values to the pointing
        offset stored in the configuration.

        Parameters
        ----------
        native_pointing : Offset2D
            The native pointing offset.  The reference position is ignored.

        Returns
        -------
        nominal_pointing_offset: Coordinate2D
        """
        offset = Coordinate2D(native_pointing)
        if self.configuration.is_configured('pointing'):
            pointing_offset = Coordinate2D(
                self.configuration.get_float_list('pointing'), unit='arcsec')
            offset.add(pointing_offset)
        return offset

    def get_si_arcseconds_offset(self, native_pointing):
        """
        Get the offsets of the science instrument in arcseconds.

        Parameters
        ----------
        native_pointing : Offset2D
            The native pointing offset.  The reference position is ignored.

        Returns
        -------
        Coordinate2D
        """
        arc_offset = self.get_nominal_pointing_offset(native_pointing)
        arc_offset.change_unit('arcsec')
        return arc_offset

    def get_si_pixel_offset(self, native_pointing):
        """
        Get the pixel offset of the science instrument.

        Parameters
        ----------
        native_pointing : Offset2D
            The native pointing offset.  The reference position is ignored.

        Returns
        -------
        Coordinate2D
        """
        si_offset = self.get_nominal_pointing_offset(native_pointing)
        angle = self.get_telescope_vpa() - self.get_instrument_vpa()

        Coordinate2D.rotate_offsets(si_offset, angle)
        Coordinate2D.rotate_offsets(si_offset, -self.channels.rotation)
        pixel_size = self.channels.get_si_pixel_size()

        pixel_coordinates = si_offset.coordinates / pixel_size.coordinates
        pixel_coordinates = pixel_coordinates.decompose().value
        reference = np.zeros(2, dtype=float)
        return Offset2D(reference, coordinates=pixel_coordinates)

    def get_pointing_string_from_increment(self, native_pointing):
        """
        Return a string representing the scan pointing.

        Parameters
        ----------
        native_pointing : Offset2D

        Returns
        -------
        str
        """
        si_offset = self.get_si_pixel_offset(native_pointing)
        result = super().get_pointing_string_from_increment(native_pointing)
        result += (f"\n\n  SIBS offset --> "
                   f"{si_offset.x:.4f}, {si_offset.y:.4f} pixels")
        return result

    def edit_pointing_header_info(self, header):
        """
        Edit pointing information in a header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.

        Returns
        -------
        None
        """
        super().edit_pointing_header_info(header)
        native_pointing = self.get_native_pointing_increment(self.pointing)
        si_offset = self.get_si_pixel_offset(native_pointing)
        dx, dy = si_offset.x, si_offset.y
        if isinstance(dx, units.Quantity):
            dx, dy = dx.decompose().value, dy.decompose().value

        offset = self.get_si_arcseconds_offset(native_pointing)
        xel, el = offset.x, offset.y
        header['SIBS_DX'] = dx, '(pixels) SIBS pointing increment in X.'
        header['SIBS_DY'] = dy, '(pixels) SIBS pointing increment in Y.'
        header['SIBS_DXE'] = (
            to_header_float(xel, 'arcsec'),
            "(arcsec) SIBS cross-elevation offset")
        header['SIBS_DE'] = (
            to_header_float(el, 'arcsec'),
            "(arcsec) SIBS elevation offset")

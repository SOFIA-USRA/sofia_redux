# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.coordinates import Angle, FK5
from astropy.io import fits
import numpy as np

from sofia_redux.scan.info.camera.info import CameraInfo
from sofia_redux.scan.custom.example.info.instrument import (
    ExampleInstrumentInfo)
from sofia_redux.scan.custom.example.info.telescope import ExampleTelescopeInfo
from sofia_redux.scan.custom.example.info.astrometry import (
    ExampleAstrometryInfo)
from sofia_redux.scan.custom.example.info.observation import (
    ExampleObservationInfo)
from sofia_redux.scan.custom.example.info.detector_array import (
    ExampleDetectorArrayInfo)
from sofia_redux.scan.utilities.class_provider import (
    frames_instance_for,
    channel_data_class_for)

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import (
    GeodeticCoordinates)
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import (
    HorizontalCoordinates)
from sofia_redux.scan.configuration.dates import DateRange
from sofia_redux.scan.simulation.scan_patterns.daisy import (
    daisy_pattern_equatorial)
from sofia_redux.scan.simulation.scan_patterns.lissajous import (
    lissajous_pattern_equatorial)
from sofia_redux.scan.simulation.source_models.simulated_source import (
    SimulatedSource)

__all__ = ['SimulationInfo']


class SimulationInfo(CameraInfo):

    def __init__(self, configuration_path=None):
        """
        Initialize an ExampleInfo object.

        Parameters
        ----------
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be used
            during the reduction.  The default is
            <package>/data/configurations.
        """
        super().__init__(configuration_path=configuration_path)
        self.name = 'simulation'
        self.instrument = ExampleInstrumentInfo()
        self.telescope = ExampleTelescopeInfo()
        self.astrometry = ExampleAstrometryInfo()
        self.observation = ExampleObservationInfo()
        self.detector_array = ExampleDetectorArrayInfo()
        self.resolution = self.instrument.resolution

    @staticmethod
    def get_file_id():
        """
        Return the file ID.

        Returns
        -------
        str
        """
        return "SIML"

    def max_pixels(self):
        """
        Return the maximum number of pixels in the example instrument.

        Returns
        -------
        int
        """
        return self.detector_array.pixels

    def copy(self):
        """
        Return a copy of the SimulationInfo.

        Returns
        -------
        SimulationInfo
        """
        return super().copy()

    def write_simulated_hdul(self, filename,
                             object_name='Simulation',
                             ra='17h45m39.60213s',
                             dec='-29d00m22.0000s',
                             site_longitude='-122.0644d',
                             site_latitude='37.4089d',
                             date_obs='2021-12-06T18:48:25.876',
                             scan_id=1,
                             scan_pattern='daisy',
                             source_type='single_gaussian',
                             overwrite=True,
                             **kwargs):
        """
        Write a simulated HDU list to file.

        Parameters
        ----------
        filename : str
            The file path to the output file for which to write the simulated
            HDU list.
        object_name : str, optional
            The name of the simulated source.
        ra : str or units.Quantity, optional
            The right-ascension of the simulated source.  The default is the
            Galactic center.
        dec : str or units.Quantity, optional
            The declination of the simulated source.  The default is the
            Galactic center.
        site_longitude : str or units.Quantity, optional
            The site longitude of the simulated observation.  The default is
            NASA Ames.
        site_latitude : str or units.Quantity, optional
            The site latitude of the simulated observation.  The default is
            NASA Ames.
        date_obs : str or Time, optional
            The date of the simulated observation.  String values should be
            provided in ISOT format in UTC scale.
        scan_id : str or int, optional
            The ID of the scan to be pla
        scan_pattern : str, optional
            The scanning pattern type.  Available patterns are {'daisy',
            'lissajous'}.
        source_type : str, optional
            The source type.  Available types are {'single_gaussian'}.
        overwrite : bool, optional
            If `True`, allow `filename` to be overwritten if it already exists.
        kwargs : dict, optional
            Optional keyword arguments that are passed into the scan pattern
            simulation or simulated source data.

        Returns
        -------
        None
        """
        hdul = self.simulated_hdul(object_name=object_name,
                                   ra=ra, dec=dec,
                                   site_longitude=site_longitude,
                                   site_latitude=site_latitude,
                                   date_obs=date_obs,
                                   scan_id=scan_id,
                                   scan_pattern=scan_pattern,
                                   source_type=source_type,
                                   **kwargs)
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()

    def simulated_hdul(self,
                       object_name='Simulation',
                       ra='17h45m39.60213s',
                       dec='-29d00m22.0000s',
                       site_longitude='-122.0644d',
                       site_latitude='37.4089d',
                       date_obs='2021-12-06T18:48:25.876',
                       scan_id=1,
                       scan_pattern='daisy',
                       source_type='single_gaussian',
                       **kwargs):
        """
        Create an HDU list containing simulated data.

        Parameters
        ----------
        object_name : str, optional
            The name of the simulated source.
        ra : str or units.Quantity, optional
            The right-ascension of the simulated source.  The default is the
            Galactic center.
        dec : str or units.Quantity, optional
            The declination of the simulated source.  The default is the
            Galactic center.
        site_longitude : str or units.Quantity, optional
            The site longitude of the simulated observation.  The default is
            NASA Ames.
        site_latitude : str or units.Quantity, optional
            The site latitude of the simulated observation.  The default is
            NASA Ames.
        date_obs : str or Time, optional
            The date of the simulated observation.  String values should be
            provided in ISOT format in UTC scale.
        scan_id : str or int, optional
            The ID of the scan to be pla
        scan_pattern : str, optional
            The scanning pattern type.  Available patterns are {'daisy',
            'lissajous'}.
        source_type : str, optional
            The source type.  Available types are {'single_gaussian'}.
        kwargs : dict, optional
            Optional keyword arguments that are passed into the scan pattern
            simulation or simulated source data.

        Returns
        -------
        hdul : fits.HDUList
            A FITS HDU list where the first (hdul[0]) HDU contains no data and
            the primary header.  The second HDU (hdul[1]) contains a FITS
            binary table with the following columns: DMJD, LST, RA, DEC, AZ,
            EL, and DAC.
        """
        header = self.simulated_observation_header(
            object_name=object_name, ra=ra, dec=dec,
            site_latitude=site_latitude, site_longitude=site_longitude,
            date_obs=date_obs, scan_id=scan_id, scan_pattern=scan_pattern,
        )
        scan_hdu = self.scan_hdu_from_header(header, **kwargs)
        source_model = SimulatedSource.get_source_model(source_type, **kwargs)
        data_hdu = self.simulated_data(scan_hdu, header, source_model,
                                       **kwargs)
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(header=header, data=np.empty(0)))
        hdul.append(data_hdu)
        return hdul

    @staticmethod
    def simulated_observation_header(object_name='Simulation',
                                     ra='17h45m39.60213s',
                                     dec='-29d00m22.0000s',
                                     site_longitude='-122.0644d',
                                     site_latitude='37.4089d',
                                     date_obs='2021-12-06T18:48:25.876',
                                     scan_id=1,
                                     scan_pattern='daisy'):
        """
        Create a simulated observation primary header.

        Parameters
        ----------
        object_name : str
            The name of the observed object.
        ra : str or units.Quantity
            The right-ascension of the observation.
        dec : str or units.Quantity
            The declination of the observation.
        site_longitude : str or units.Quantity
            The site longitude.
        site_latitude : str or units.Quantity
            The site latitude.
        date_obs : str
            The date-time of the observation in ISOT format, UTC scale.
        scan_id : str or int
            The scan identifier.
        scan_pattern : str
            The scan pattern type.  Allowable values are
            {'daisy', 'lissajous'}.

        Returns
        -------
        header : fits.Header
        """
        obs_header = fits.Header()
        obs_header['OBJECT'] = object_name, 'Object catalog name.'
        obs_header['SCANID'] = scan_id, 'Scan identifier.'
        date_obs = DateRange.to_time(date_obs)
        obs_header['DATE-OBS'] = date_obs.isot, 'Observation start (UTC).'
        obs_header['SCANPATT'] = scan_pattern, 'Scanning pattern.'

        if isinstance(ra, str):
            center = FK5(ra=ra, dec=dec)
            center = EquatorialCoordinates([center.ra, center.dec])
        else:
            center = EquatorialCoordinates([ra, dec])

        obs_header['OBSRA'] = (
            center.ra.to('hourangle').value, '(hour) Requested RA.')
        obs_header['OBSDEC'] = (
            center.dec.to('degree').value, '(deg) Requested DEC.')

        if isinstance(site_latitude, str):
            site_latitude = Angle(site_latitude)
            site_longitude = Angle(site_longitude)
        site = GeodeticCoordinates([site_longitude, site_latitude],
                                   unit='degree')
        obs_header['SITELON'] = site.longitude.value, '(deg) Site longitude.'
        obs_header['SITELAT'] = site.latitude.value, '(deg) Site latitude.'

        lst = date_obs.sidereal_time('mean', longitude=site.longitude)
        horizontal = center.to_horizontal(site, lst)
        obs_header['LST'] = lst.value, '(hour) Local sidereal time.'
        obs_header['OBSAZ'] = (horizontal.az.to('degree').value,
                               '(deg) Observation Azimuth.')
        obs_header['OBSEL'] = (horizontal.el.to('degree').value,
                               '(deg) Observation Elevation.')

        return obs_header

    def scan_hdu_from_header(self, header, **kwargs):
        """
        Create an HDU containing the simulated scan pattern from a header.

        Parameters
        ----------
        header : fits.Header
            The simulated observation primary header.
        kwargs : dict, optional
            Optional keyword arguments to pass into the scan pattern simulator.

        Returns
        -------
        hdu : fits.BinTableHDU
            The header data unit containing DMJD, LST, RA, DEC, AZ, and EL
            columns.
        """
        scan_type = header['SCANPATT'].upper().strip()
        ra = header['OBSRA'] * units.Unit('hourangle')
        dec = header['OBSDEC'] * units.Unit('degree')
        center = EquatorialCoordinates([ra, dec])
        site = GeodeticCoordinates([header['SITELON'], header['SITELAT']],
                                   unit='degree')
        dt = self.instrument.sampling_interval

        if scan_type == 'DAISY':
            equatorial = daisy_pattern_equatorial(center, dt, **kwargs)
        elif scan_type == 'LISSAJOUS':
            equatorial = lissajous_pattern_equatorial(center, dt, **kwargs)
        else:
            raise NotImplementedError(
                f"{scan_type} scanning pattern not supported.")

        date_obs = DateRange.to_time(header['DATE-OBS'])
        obs_time = dt * np.arange(equatorial.size) + date_obs
        lst = obs_time.sidereal_time('mean', longitude=site.longitude)
        horizontal = equatorial.to_horizontal(site, lst)

        ra = fits.Column(name='RA', format='1D', unit='hourangle',
                         array=equatorial.ra.to('hourangle').value)
        dec = fits.Column(name='DEC', format='1D', unit='degree',
                          array=equatorial.dec.to('degree').value)
        dmjd = fits.Column(name='DMJD', format='1D', unit='day',
                           array=obs_time.mjd)
        lst = fits.Column(name='LST', format='1D', unit='hour',
                          array=obs_time.sidereal_time(
                              'mean', longitude=site.longitude))
        az = fits.Column(name='AZ', format='1D', unit='degree',
                         array=horizontal.az.to('degree').value)
        el = fits.Column(name='EL', format='1D', unit='degree',
                         array=horizontal.el.to('degree').value)
        columns = fits.ColDefs([dmjd, lst, ra, dec, az, el])
        hdu = fits.BinTableHDU.from_columns(columns)
        return hdu

    def set_frames_coordinates(self, frames, table):

        hourangle = units.Unit('hourangle')
        deg = units.Unit('degree')

        if frames.info is None:
            frames.default_info = self

        if frames.valid is None:
            frames.set_frame_size(table.size)

        frames.mjd[:] = table['DMJD']
        frames.lst[:] = table['LST'] * hourangle
        ra = table['RA'] * hourangle
        dec = table['DEC'] * deg

        equatorial = EquatorialCoordinates(
            np.stack((ra, dec)), epoch=frames.info.astrometry.epoch,
            copy=False)
        frames.equatorial[:] = equatorial

        horizontal = HorizontalCoordinates(np.stack(
            (table['AZ'], table['EL'])), unit='degree', copy=False)
        frames.horizontal[:] = horizontal

        frames.calculate_parallactic_angle()
        horizontal_offset = equatorial.get_native_offset_from(
            frames.info.astrometry.equatorial)
        frames.equatorial_native_to_horizontal_offset(
            horizontal_offset, in_place=True)
        frames.horizontal_offset[:] = horizontal_offset
        frames.set_rotation(0)

    def simulated_data(self, scan_hdu, header, source_model, **kwargs):
        """
        Create an HDU containing modelled source data.

        Parameters
        ----------
        scan_hdu : fits.BinTableHDU
            A FITS table containing scanning data which must include the
            columns RA, DEC, AZ, and EL.
        header : fits.Header
            The primary FITS header.  Must contain the keys OBSRA, OBSDEC,
            SITELON, and SITELAT.
        source_model : SimulatedSource
        kwargs : dict, optional
            Optional keyword arguments to be applied to the simulated data.

        Returns
        -------
        data_hdu : fits.BinTableHDU
            An updated FITS table including the simulated data in the DAC
            column.
        """
        data = self.create_data(scan_hdu, header, source_model, **kwargs)
        n_records, n_row, n_col = data.shape

        data_column = fits.Column(
            name='DAC',
            format=f'{n_records}D',
            unit='count',
            array=data,
            dim=f'({n_row}, {n_col})')

        new_cols = scan_hdu.columns + data_column
        hdu = fits.BinTableHDU.from_columns(new_cols)
        return hdu

    def create_data(self, scan_hdu, header, source_model, **kwargs):
        """
        Create modelled source data.

        Parameters
        ----------
        scan_hdu : fits.BinTableHDU
            A FITS table containing scanning data which must include the
            columns RA, DEC, AZ, and EL.
        header : fits.Header
            The primary FITS header.  Must contain the keys OBSRA, OBSDEC,
            SITELON, and SITELAT.
        source_model : SimulatedSource
        kwargs : dict, optional
            Optional keyword arguments that will be applied to the data.

        Returns
        -------
        data : numpy.ndarray
            Simulated data of the shape (n_records, n_row, n_col).
        """
        table = scan_hdu.data

        info = self.copy()
        lon = header['SITELON'] * units.Unit('degree')
        lat = header['SITELAT'] * units.Unit('degree')
        ra = header['OBSRA'] * units.Unit('hourangle')
        dec = header['OBSDEC'] * units.Unit('degree')
        info.astrometry.equatorial = EquatorialCoordinates([ra, dec])
        info.astrometry.site = GeodeticCoordinates([lon, lat])

        frames = frames_instance_for('example')
        info.set_frames_coordinates(frames, table)

        channel_data = channel_data_class_for('example')()
        channel_data.default_info = info

        info.detector_array.initialize_channel_data(channel_data)
        offsets = frames.get_equatorial_native_offset(channel_data.position)

        # Create the output data
        n_records = table.size
        n_row, n_col = info.detector_array.ROWS, info.detector_array.COLS
        data = np.empty((n_records, n_row, n_col), dtype=float)
        data[:, channel_data.row, channel_data.col] = source_model(offsets)
        self.modify_data(data, **kwargs)
        return data

    def modify_data(self, data, **kwargs):
        """
        Add various properties to simulated scan data.

        Parameters
        ----------
        data : numpy.ndarray (float)
            The simulated data of shape (n_records, n_row, n_col).  Will be
            modified in-place.

        Returns
        -------
        None
        """
        s2n = kwargs.get('s2n')
        seed = kwargs.get('seed')
        if seed is not None:
            rand = np.random.RandomState(seed)
        else:
            rand = np.random
        if s2n is not None and float(s2n) > 0:
            max_data = np.nanmax(np.abs(data))
            noise_level = max_data / float(s2n)
            data += rand.randn(*data.shape) * noise_level

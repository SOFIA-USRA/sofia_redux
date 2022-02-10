from abc import ABC
from astropy import constants
from astropy import units
import numpy as np

from sofia_redux.scan.configuration.dates import DateRange
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates


class AircraftSimulation(ABC):

    def __init__(self):
        units.imperial.enable()
        self.start_altitude = 41 * units.Unit('ft') * 1000
        self.end_altitude = 41 * units.Unit('ft') * 1000
        self.heading = None
        self.airspeed = 500 * units.Unit('knot')
        self.ground_speed = None  # Calculate via subtracting wind
        self.start_utc = None
        self.end_utc = None
        self.flight_time = None
        self.source = None
        self.start_location = None
        self.end_location = None
        self.start_lst = None
        self.end_lst = None
        self.start_horizontal = None
        self.end_horizontal = None

    def initialize_from_header(self, header):
        """
        Initialize the aircraft from settings in a primary FITS header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        if 'ALTI_STA' in header:
            self.start_altitude = float(header['ALTI_STA']) * units.Unit('ft')
        if 'ALTI_END' in header:
            self.end_altitude = float(header['ALTI_END']) * units.Unit('ft')
        if 'AIRSPEED' in header:
            self.airspeed = float(header['AIRSPEED']) * units.Unit('knot')
        if 'GRDSPEED' in header:
            self.ground_speed = float(header['GRDSPEED']) * units.Unit('knot')
        else:
            self.ground_speed = self.airspeed  # Assume no wind

        if 'DATE-OBS' in header:
            self.start_utc = DateRange.to_time(header['DATE-OBS'])

        if 'EXPTIME' in header:
            self.end_utc = self.start_utc + header['EXPTIME'] * units.Unit('s')
        elif 'UTCEND' in header and 'DATE-OBS' in header:
            day = header['DATE-OBS'].split('T')[0]
            utc_end = f"{day}T{header['UTCEND']}"
            self.end_utc = DateRange.to_time(utc_end)

        if self.end_utc is None:
            raise ValueError("Cannot determine flight length (time) "
                             "from header.")

        self.flight_time = (self.end_utc - self.start_utc).to('second')

        self.source = EquatorialCoordinates([
            header['OBJRA'] * units.Unit('hourangle'),
            header['OBJDEC'] * units.Unit('degree')])

        self.start_location = GeodeticCoordinates([
            header['LON_STA'] * units.Unit('degree'),
            header['LAT_STA'] * units.Unit('degree')])

        self.start_lst = self.start_utc.sidereal_time(
            'mean', longitude=self.start_location.longitude)

        self.start_horizontal = self.source.to_horizontal(
            self.start_location, self.start_lst)

        self.orient_to_source()
        self.calculate_end_position()

    def orient_to_source(self):
        """
        Orient the aircraft wrt the source.

        Alters the heading of the aircraft so the source is at 90 degrees
        to the side.

        Returns
        -------
        None
        """
        self.heading = self.start_horizontal.az - (90 * units.Unit('degree'))
        self.heading = self.heading.to('degree')

    def calculate_end_position(self):
        """
        Calculate the end position of the flight using Vicenty's formula.

        Returns
        -------
        None
        """
        d = (self.airspeed * self.flight_time).decompose()  # distance
        mean_height = (self.start_altitude + self.end_altitude) / 2
        r = (constants.R_earth + mean_height).decompose()  # radius
        a = self.heading  # bearing
        dr = (d / r).decompose().value * units.Unit('radian')

        lon_1 = self.start_location.longitude
        lat_1 = self.start_location.latitude

        lat_2 = np.arcsin(np.sin(lat_1) * np.cos(dr)
                          + np.cos(lat_1) * np.sin(dr) * np.cos(a))

        lon_2 = lon_1 + np.arctan2(np.sin(a) * np.sin(dr) * np.cos(lat_1),
                                   np.cos(dr) - (np.sin(lat_1)
                                                 * np.sin(lat_2)))

        self.end_location = GeodeticCoordinates([lon_2, lat_2], unit='degree')
        self.end_lst = self.end_utc.sidereal_time(
            'mean', longitude=self.end_location.longitude)
        self.end_horizontal = self.source.to_horizontal(
            self.end_location, self.end_lst)

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.epoch.epoch import (
    JulianEpoch, Epoch, J2000)
from sofia_redux.scan.coordinate_systems.epoch.precession import Precession
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.focal_plane_coordinates import \
    FocalPlaneCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.info.astrometry import AstrometryInfo


class TestAstrometryInfo(object):
    def test_init(self):
        info = AstrometryInfo()
        # spot check defaults
        assert np.isnan(info.mjd)
        assert np.isnan(info.lst)
        assert info.time_stamp is None
        assert info.date is None
        assert info.pointing is None
        assert info.ground_based is True

    def test_calculate_precessions(self):
        info = AstrometryInfo()
        info.configuration = Configuration()
        info.set_mjd(12345)
        assert info.mjd == 12345
        assert info.configuration.dates.current_date == 12345

        info.calculate_precessions(2000)

        assert isinstance(info.epoch, Epoch)
        assert info.epoch.equinox.value == 'J2000.000'
        assert isinstance(info.apparent_epoch, JulianEpoch)
        assert info.apparent_epoch.equinox.value == 12345
        assert isinstance(info.from_apparent, Precession)
        assert isinstance(info.to_apparent, Precession)

    def test_precess(self, populated_scan):
        info = AstrometryInfo()
        info.mjd = 12345
        info.equatorial = populated_scan.equatorial.copy()

        info.precess(1900)
        assert info.epoch.equinox.value == 'B1900.000'
        assert info.equatorial.epoch.equinox.value == 'B1900.000'
        info.precess(J2000)
        assert info.epoch.equinox.value == 'J2000.000'
        assert info.equatorial.epoch.equinox.value == 'J2000.000'
        info.precess('2020')
        assert info.epoch.equinox.value == 'J2020.000'
        assert info.equatorial.epoch.equinox.value == 'J2020.000'

        # pass a scan
        for integ in populated_scan.integrations:
            assert integ.frames.equatorial.epoch.equinox.value == 'J2000.000'
        info.precess(1900, scan=populated_scan)
        assert info.epoch.equinox.value == 'B1900.000'
        assert info.equatorial.epoch.equinox.value == 'B1900.000'
        for integ in populated_scan.integrations:
            assert integ.frames.equatorial.epoch.equinox.value == 'B1900.000'

    def test_calculate_equatorial(self, populated_scan):
        populated_scan.validate()
        info = populated_scan.astrometry
        info.from_apparent = None
        info.equatorial = None

        info.calculate_equatorial()
        assert isinstance(info.from_apparent, Precession)
        assert isinstance(info.equatorial, EquatorialCoordinates)
        assert info.equatorial.epoch.equinox.value == 'J2000.000'

    def test_calculate_apparent(self, populated_scan):
        populated_scan.validate()
        info = populated_scan.astrometry
        info.apparent = None
        info.to_apparent = None

        info.calculate_apparent()
        assert isinstance(info.to_apparent, Precession)
        assert isinstance(info.apparent, EquatorialCoordinates)
        assert info.apparent.epoch.equinox.value == info.mjd

    def test_calculate_horizontal(self, populated_scan):
        populated_scan.validate()
        info = populated_scan.astrometry
        info.apparent = None
        info.horizontal = None

        info.calculate_horizontal()
        assert isinstance(info.apparent, EquatorialCoordinates)
        assert info.apparent.epoch.equinox.value == info.mjd
        assert isinstance(info.horizontal, HorizontalCoordinates)

        assert info.get_native_coordinates() is info.horizontal

    def test_get_position_reference(self, capsys, populated_scan):
        populated_scan.validate()
        info = populated_scan.astrometry

        assert info.get_position_reference() is info.equatorial
        assert info.get_position_reference('equatorial') is info.equatorial
        assert info.get_position_reference('horizontal') is info.horizontal
        assert info.get_position_reference('native') is info.horizontal

        ref = info.get_position_reference('focalplane')
        assert isinstance(ref, FocalPlaneCoordinates)
        assert ref.x == [0]
        assert ref.y == [0]

        info.is_nonsidereal = True
        ref = info.get_position_reference()
        assert ref is info.equatorial

        # add reference RA/Dec
        info.configuration.set_option('reference.ra', 10)
        ref = info.get_position_reference()
        assert ref is info.equatorial
        assert 'reference.ra or reference.dec configuration was ' \
               'given without the other' in capsys.readouterr().err

        info.configuration.set_option('reference.dec', 10)
        ref = info.get_position_reference()
        assert ref is not info.equatorial
        assert isinstance(ref, EquatorialCoordinates)
        assert ref.ra == 10 * units.hourangle
        assert ref.dec == 10 * units.deg

    def test_apply_scan(self):
        info = AstrometryInfo()
        info.configuration = Configuration()
        assert info.is_nonsidereal is False

        info.apply_scan('test')
        assert info.is_nonsidereal is False

        info.configuration.set_option('moving', True)
        info.apply_scan('test')
        assert info.is_nonsidereal is True

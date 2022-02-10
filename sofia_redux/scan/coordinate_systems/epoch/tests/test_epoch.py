# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.coordinates import FK4
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.epoch.epoch import (
    Epoch, J2000, B1950, B1900, JulianEpoch, BesselianEpoch)


def test_init():
    epoch = Epoch()
    assert epoch.equinox.jyear == 2000
    assert not epoch.immutable


def test_copy():
    epoch = Epoch()
    e2 = epoch.copy()
    assert e2 == epoch and e2 is not epoch


def test_empty_copy():
    epoch = Epoch()
    e2 = epoch.empty_copy()
    assert e2.immutable is epoch.immutable
    assert epoch.equinox is not None and e2.equinox is None


def test_eq():
    epoch = Epoch()
    assert epoch is not None
    assert epoch == epoch
    ej = epoch.get_julian_epoch()
    assert epoch != ej
    e2 = Epoch(equinox=np.arange(5) + 2000)
    assert e2 != epoch
    e2 = Epoch(equinox=2001)
    assert e2 != epoch
    e1 = Epoch(equinox=np.arange(5) + 2000)
    e2 = Epoch(equinox=np.arange(6) + 2000)
    assert e1 != e2
    e2 = e1.copy()
    assert e2 == e1


def test_getitem():
    epoch = Epoch(np.arange(5) + 2000)
    e1 = epoch[1]
    assert e1.singular and e1.year == 2001


def test_singular():
    epoch = Epoch(np.arange(5) + 2000)
    assert not epoch.singular
    epoch = Epoch()
    assert epoch.singular


def test_ndim():
    assert Epoch().ndim == 0
    assert Epoch(np.arange(5) + 2000).ndim == 1
    assert Epoch(np.arange(10).reshape(2, 5) + 2000).ndim == 2


def test_shape():
    assert Epoch().shape == ()
    assert Epoch(np.arange(10)).shape == (10,)
    assert Epoch(np.arange(12).reshape(3, 4)).shape == (3, 4)
    epoch = Epoch()
    epoch.equinox = None
    assert epoch.shape == ()


def test_size():
    assert Epoch().size == 1
    assert Epoch(np.arange(10).reshape(2, 5)).size == 10


def test_year():
    assert Epoch().year == 2000
    assert np.isclose(Epoch('B1950').year, 1949.9997904422999)
    epoch = Epoch()
    epoch.year = 2001
    assert epoch.year == 2001


def test_julian_year():
    epoch = Epoch('B1950')
    assert np.isclose(epoch.julian_year, 1949.9997904422999)
    epoch = Epoch('J2000')
    assert epoch.julian_year == 2000


def test_besselian_year():
    epoch = Epoch('B1950')
    assert epoch.besselian_year == 1950
    epoch = Epoch('J2000')
    assert np.isclose(epoch.besselian_year, 2000.001277513665)


def test_mjd():
    epoch = Epoch('J2000')
    assert epoch.mjd == 51544.5
    epoch.mjd += 1
    assert np.isclose(epoch.year, 2000.0027398846553)


def test_is_julian():
    epoch = Epoch()
    assert epoch.is_julian
    epoch = Epoch('J2000')
    assert epoch.is_julian
    epoch = Epoch('B1950')
    assert not epoch.is_julian


def test_str():
    epoch = Epoch()
    assert str(epoch) == '2000.0'
    epoch = Epoch([2001, 2002])
    assert str(epoch) == 'MJD 51909.75 -> 52275.0'


def test_get_equinox():
    assert Epoch.get_equinox(None).jyear == 2000
    assert Epoch.get_equinox(FK4()).byear == 1950
    assert Epoch.get_equinox(Time(2021, format='jyear')).jyear == 2021
    h = fits.Header({'EQUINOX': 'J2020'})
    assert Epoch.get_equinox(h).jyear == 2020
    assert Epoch.get_equinox('B1960').byear == 1960
    assert Epoch.get_equinox('2022').jyear == 2022

    assert Epoch.get_equinox(1983).byear == 1983
    assert Epoch.get_equinox(1984).jyear == 1984

    assert np.allclose(Epoch.get_equinox(np.arange(3) + 2000).jyear,
                       [2000, 2001, 2002])


def test_get_epoch():
    assert Epoch.get_epoch('J2000') == J2000
    assert Epoch.get_epoch(J2000) == J2000
    assert Epoch.get_epoch(B1950) == B1950


def test_get_equinox_from_header():
    h = fits.Header()
    h['EQUINOX'] = 'J2001.0'
    assert Epoch.get_equinox_from_header(h).jyear == 2001


def test_edit_header():
    h = fits.Header()
    epoch = Epoch(np.arange(5) + 2000)
    epoch.edit_header(h)
    assert len(h) == 0
    epoch = Epoch('J2000')
    epoch.edit_header(h)
    assert h['EQUINOX'] == 2000.0
    assert h.comments['EQUINOX'] == 'The epoch of the quoted coordinates'


def test_set_year():
    epoch = Epoch(immutable=True)
    with pytest.raises(ValueError) as err:
        epoch.set_year(2000)
    assert "Cannot alter" in str(err.value)
    epoch = Epoch()
    epoch.set_year(Time(2001, format='jyear'))
    assert epoch.year == 2001
    epoch.set_year(2002)
    assert epoch.year == 2002


def test_set_mjd():
    epoch = Epoch(immutable=True)
    with pytest.raises(ValueError) as err:
        epoch.set_mjd(51544.5)
    assert "Cannot alter" in str(err.value)
    epoch = Epoch()
    epoch.set_mjd(51545.5)
    # Conversion from UTC to TT scale
    assert np.isclose(epoch.mjd, 51545.50074287037)
    epoch.set_mjd(Time(51545.5, format='mjd', scale='tt'))
    assert epoch.mjd == 51545.5


def test_get_julian_epoch():
    epoch = Epoch(immutable=True)
    j = epoch.get_julian_epoch()
    assert j.immutable and j.year == 2000
    epoch = Epoch('B1950')
    j = epoch.get_julian_epoch()
    assert not j.immutable and np.isclose(j.year, 1949.9995808890756)


def test_get_besselian_epoch():
    epoch = Epoch(immutable=True)
    b = epoch.get_besselian_epoch()
    assert b.immutable and np.isclose(b.year, 2000.0025550546166)
    epoch = Epoch('B1950')
    b = epoch.get_besselian_epoch()
    assert not b.immutable and b.year == 1950


def test_get_indices():
    epoch = Epoch()
    with pytest.raises(KeyError) as err:
        _ = epoch.get_indices(1)
    assert "singular epochs" in str(err.value)

    epoch = Epoch(np.arange(2000, 2004))
    e = epoch.get_indices(None)
    assert e.equinox is None
    e = epoch.get_indices(np.asarray(1))
    assert e.year == 2001


def test_julian_epoch_init():
    epoch = JulianEpoch()
    assert epoch.year == 2000
    assert epoch.equinox.jyear == 2000
    assert not epoch.immutable


def test_julian_epoch_copy():
    epoch = JulianEpoch()
    e2 = epoch.copy()
    assert e2 == epoch and e2 is not epoch


def test_julian_epoch_year():
    epoch = JulianEpoch([1960, 1961])
    epoch.year = np.arange(1962, 1964)
    assert np.allclose(epoch.year, [1962, 1963])
    assert np.allclose(epoch.equinox.jyear, [1962, 1963])


def test_julian_epoch_str():
    epoch = JulianEpoch()
    assert str(epoch) == 'J2000.0'
    epoch = JulianEpoch([2000, 2001])
    assert str(epoch) == 'Julian MJD 51544.5 -> 51909.75'


def test_julian_epoch_get_equinox():
    assert JulianEpoch.get_equinox(1983).jyear == 1983


def test_julian_epoch_get_indices():
    epoch = JulianEpoch([2000, 2001])
    e1 = epoch.get_indices(1)
    assert e1.year == 2001


def test_besselian_epoch_init():
    epoch = BesselianEpoch(2000)
    assert epoch.year == 2000
    assert epoch.equinox.byear == 2000
    assert not epoch.immutable


def test_besselian_epoch_copy():
    epoch = BesselianEpoch()
    b = epoch.copy()
    assert b == epoch and b is not epoch


def test_besselian_epoch_year():
    epoch = BesselianEpoch([2000, 2001])
    epoch.year = np.arange(2002, 2004)
    assert np.allclose(epoch.year, [2002, 2003])
    assert np.allclose(epoch.equinox.byear, [2002, 2003])


def test_besselian_epoch_str():
    epoch = BesselianEpoch()
    assert str(epoch) == 'B1950.0'
    epoch = BesselianEpoch([2000, 2001])
    assert str(epoch) == 'Besselian MJD 51544.0333981 -> 51909.27559688'


def test_besselian_epoch_get_equinox():
    assert BesselianEpoch.get_equinox(2000).byear == 2000


def test_besselian_epoch_get_indices():
    epoch = BesselianEpoch([1950, 1951])
    e1 = epoch.get_indices(1)
    assert e1.year == 1951


def test_j2000():
    j = J2000
    assert j.year == 2000
    assert j.equinox.jyear == 2000


def test_b1900():
    b = B1900
    assert b.year == 1900
    assert b.equinox.byear == 1900


def test_b1950():
    b = B1950
    assert b.year == 1950
    assert b.equinox.byear == 1950

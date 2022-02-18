# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.source_models.beams.gaussian_source import GaussianSource
from sofia_redux.scan.source_models.beams.elliptical_source import \
    EllipticalSource


arcsec = units.Unit('arcsec')
degree = units.Unit('degree')


@pytest.fixture
def gaussian2d():
    g = Gaussian2D(x_fwhm=10, y_fwhm=5, theta=30, peak=2.0, peak_unit='Jy',
                   position_unit=arcsec)
    return g


@pytest.fixture
def gaussian_source(gaussian2d):
    return GaussianSource(gaussian_model=gaussian2d)


@pytest.fixture
def elliptical_source(gaussian2d):
    return EllipticalSource(gaussian_model=gaussian2d)


def test_init(gaussian_source):
    g = EllipticalSource()
    assert g.elongation == 0
    assert g.angle_weight is None

    g = EllipticalSource(gaussian_model=gaussian_source)
    assert np.isclose(g.elongation, 1/3)
    assert np.isclose(g.fwhm, 7.07106781 * arcsec, atol=1e-6)
    assert g.fwhm == g.x_fwhm
    assert g.y_fwhm == g.y_fwhm


def test_major_fwhm(elliptical_source):
    e = elliptical_source.copy()
    assert np.isclose(e.major_fwhm, 10 * arcsec)


def test_minor_fwhm(elliptical_source):
    e = elliptical_source.copy()
    assert np.isclose(e.minor_fwhm, 5 * arcsec)


def test_major_fwhm_weight(elliptical_source):
    e = elliptical_source.copy()
    assert e.major_fwhm_weight == np.inf
    e.elongation_weight = 0.25
    e.fwhm_weight = 1.0
    assert np.isclose(e.major_fwhm_weight, 0.00441989 / arcsec ** 2, atol=1e-6)
    e.fwhm_weight = 0.0
    assert np.isclose(e.major_fwhm_weight, 0 / arcsec ** 2, atol=1e-6)


def test_minor_fwhm_weight(elliptical_source):
    e = elliptical_source.copy()
    assert e.minor_fwhm_weight == np.inf
    e.elongation_weight = 0.25
    e.fwhm_weight = 0.5
    assert np.isclose(e.minor_fwhm_weight, 0.0043956 / arcsec ** 2, atol=1e-6)
    e.fwhm_weight = 0.0
    assert np.isclose(e.minor_fwhm_weight, 0 / arcsec ** 2, atol=1e-6)


def test_major_fwhm_rms(elliptical_source):
    e = elliptical_source.copy()
    e.elongation_weight = 10.0
    assert np.isclose(e.major_fwhm_rms, 2.622022 * arcsec, atol=1e-6)


def test_minor_fwhm_rms(elliptical_source):
    e = elliptical_source.copy()
    e.elongation_weight = 20.0
    assert np.isclose(e.minor_fwhm_rms, 2.015564 * arcsec, atol=1e-6)


def test_angle(elliptical_source):
    e = elliptical_source.copy()
    assert e.angle == 30 * degree
    e.angle = 45 * degree
    assert e.angle == 45 * degree


def test_angle_rms(elliptical_source):
    e = elliptical_source.copy()
    assert e.angle_rms == 0 * degree
    e.angle_weight = 0.25 / (degree ** 2)
    assert e.angle_rms == 2 * degree


def test_elongation_rms(elliptical_source):
    e = elliptical_source.copy()
    assert e.elongation_rms == 0
    e.elongation_weight = 0.25
    assert e.elongation_rms == 2


def test_set_elongation(elliptical_source):
    e = elliptical_source.copy()
    assert np.isclose(e.elongation, 1/3)
    e.set_elongation()
    assert np.isclose(e.elongation, 1/3)
    assert e.elongation_weight == np.inf

    e.set_xy_fwhm(0 * arcsec, 0 * arcsec)
    e.set_elongation(weight=None)
    assert e.elongation == 0
    assert e.elongation_weight == np.inf

    e.set_elongation(major=1 * arcsec, minor=3 * arcsec, weight=4.0,
                     angle=45 * degree)
    assert np.isclose(e.elongation, 0.5)
    assert np.isclose(e.major_fwhm, 3 * arcsec)
    assert np.isclose(e.minor_fwhm, 1 * arcsec)
    assert np.isclose(e.position_angle, 135 * degree)
    assert np.isclose(e.elongation_weight, 4.0)

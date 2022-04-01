# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates \
    import EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D
from sofia_redux.scan.frames.horizontal_frames \
    import HorizontalFrames, HorizontalCoordinates


@pytest.fixture
def initialized_frames(populated_integration):
    frames = HorizontalFrames()
    nframe = 10
    frames.initialize(populated_integration, nframe)
    return frames


@pytest.fixture
def validated_frames(initialized_frames):
    initialized_frames.validate()
    return initialized_frames


@pytest.fixture
def offsets():
    x, y = np.meshgrid([-2, -1, 0, 1, 2], [-60, -45, -30, 0, 30, 45, 60])
    offsets = EquatorialCoordinates([x.ravel(), y.ravel()], unit='degree')
    return offsets


@pytest.fixture
def projector():
    projection = DefaultProjection2D()
    projection.reference = Coordinate2D([5, 5])
    projector = AstroProjector(projection)
    return projector


class TestHorizontalFrames(object):

    def test_property_defaults(self):
        frames = HorizontalFrames()

        # quick checks on property defaults for unpopulated frames
        assert 'zenith_tau' in frames.default_field_types
        assert 'horizontal' in frames.default_field_types
        assert frames.site is None

    def test_site(self, initialized_frames):
        assert initialized_frames.site is initialized_frames.astrometry.site

    def test_validate(self, initialized_frames):
        # no telescope info
        initialized_frames.validate()
        assert initialized_frames.valid.all()
        assert isinstance(initialized_frames.equatorial, EquatorialCoordinates)
        assert isinstance(initialized_frames.horizontal, HorizontalCoordinates)

        # equatorial None: is calculated
        initialized_frames.equatorial = None
        initialized_frames.validate()
        assert isinstance(initialized_frames.equatorial, EquatorialCoordinates)
        assert isinstance(initialized_frames.horizontal, HorizontalCoordinates)

        # horizontal None: is calculated
        initialized_frames.horizontal = None
        initialized_frames.validate()
        assert isinstance(initialized_frames.equatorial, EquatorialCoordinates)
        assert isinstance(initialized_frames.horizontal, HorizontalCoordinates)

    def test_get_equatorial(self, validated_frames, offsets):
        nframe = validated_frames.size
        eq = validated_frames.get_equatorial(offsets)
        assert isinstance(eq, EquatorialCoordinates)
        assert eq.x.shape == (nframe, offsets.x.size)
        assert eq.y.shape == (nframe, offsets.y.size)

        # indices
        eq = validated_frames.get_equatorial(offsets, indices=np.arange(4))
        assert isinstance(eq, EquatorialCoordinates)
        assert eq.x.shape == (4, offsets.x.size)
        assert eq.y.shape == (4, offsets.y.size)

        # singular
        eq = validated_frames[0].get_equatorial(offsets)
        assert isinstance(eq, EquatorialCoordinates)
        assert eq.x.shape == (1, offsets.x.size)
        assert eq.y.shape == (1, offsets.y.size)

    def test_get_horizontal(self, validated_frames, offsets):
        nframe = validated_frames.size
        hz = validated_frames.get_horizontal(offsets)
        assert isinstance(hz, HorizontalCoordinates)
        assert hz.x.shape == (nframe, offsets.x.size)
        assert hz.y.shape == (nframe, offsets.y.size)

        # indices
        hz = validated_frames.get_horizontal(offsets, indices=np.arange(4))
        assert isinstance(hz, HorizontalCoordinates)
        assert hz.x.shape == (4, offsets.x.size)
        assert hz.y.shape == (4, offsets.y.size)

        # singular
        hz = validated_frames[0].get_horizontal(offsets)
        assert isinstance(hz, HorizontalCoordinates)
        assert hz.x.shape == (1, offsets.x.size)
        assert hz.y.shape == (1, offsets.y.size)

    def test_get_native_offset(self, validated_frames, offsets):
        nframe = validated_frames.size

        nt = validated_frames.get_native_offset(offsets.copy())
        assert isinstance(nt, Coordinate2D)
        assert nt.x.shape == (nframe, offsets.x.size)
        assert nt.y.shape == (nframe, offsets.y.size)
        assert not np.allclose(nt.x, 0)
        assert not np.allclose(nt.y, 0)

    def test_get_absolute_native_offsets(self, validated_frames):
        offset = validated_frames.get_absolute_native_offsets()
        assert offset is validated_frames.horizontal_offset

    def test_get_base_native_offset(self, validated_frames):
        offset = validated_frames.get_base_native_offset()
        assert offset is not validated_frames.horizontal_offset
        assert np.allclose(offset.x, validated_frames.horizontal_offset.x)
        assert np.allclose(offset.y, validated_frames.horizontal_offset.y)

    def test_project(self, mocker, validated_frames, projector):
        positions = Coordinate2D([1, 4], unit=units.arcsec)

        # placeholder projector always returns 5, 5
        offset = validated_frames.project(positions, projector)
        assert offset.x == 5
        assert offset.y == 5

        # mock horizontal projector, check that functions are called
        mocker.patch.object(projector, 'is_horizontal', return_value=True)
        projector.coordinates.add_native_offset = lambda *args: None
        m1 = mocker.patch.object(validated_frames, 'get_horizontal_offset')
        offset = validated_frames.project(positions, projector)
        assert offset.x == 5
        assert offset.y == 5
        m1.assert_called_once()

    def test_parallactic_angle(self, validated_frames):

        angle = validated_frames.get_parallactic_angle()
        assert angle.size == validated_frames.size
        assert np.allclose(angle, 0 * units.radian)

        validated_frames.set_parallactic_angle(45 * units.deg)
        angle = validated_frames.get_parallactic_angle()
        assert np.allclose(angle, 45 * units.deg)

        # indices
        angle = validated_frames.get_parallactic_angle(indices=np.arange(4))
        assert angle.size == 4
        assert np.allclose(angle, 45 * units.deg)

        # singular
        frames = validated_frames[0]
        frames.set_parallactic_angle(20 * units.deg)
        angle = frames.get_parallactic_angle()
        assert np.isclose(angle, 20 * units.deg)

        # no quantities
        frames.set_parallactic_angle(0.1)
        angle = frames.get_parallactic_angle()
        assert np.isclose(angle, 0.1 * units.rad)

    def test_calculate_parallactic_angle(self, validated_frames):
        # set angle from site
        validated_frames.calculate_parallactic_angle()
        angle = validated_frames.get_parallactic_angle()
        assert np.allclose(angle, 0)

        # set from site + lst
        validated_frames.calculate_parallactic_angle(lst=10 * units.deg)
        angle = validated_frames.get_parallactic_angle()
        assert np.allclose(angle, 0.2 * units.rad, atol=0.03)

    def test_calculate_horizontal(self, validated_frames):
        validated_frames.horizontal = None
        validated_frames.calculate_horizontal()
        assert isinstance(validated_frames.horizontal, HorizontalCoordinates)

    def test_calculate_equatorial(self, validated_frames, mocker):
        validated_frames.equatorial = None
        validated_frames.calculate_equatorial()
        assert isinstance(validated_frames.equatorial, EquatorialCoordinates)
        eq = validated_frames.equatorial.copy()

        # is tracking: adds tracking coordinates to scan
        validated_frames.equatorial = None
        validated_frames.scan.info.telescope.is_tracking = True
        validated_frames.calculate_equatorial()
        assert isinstance(validated_frames.equatorial, EquatorialCoordinates)
        assert not np.allclose(validated_frames.equatorial.x, eq.x,
                               equal_nan=True)
        assert not np.allclose(validated_frames.equatorial.y, eq.y,
                               equal_nan=True)

    def test_pointing_at(self, mocker):
        frames = HorizontalFrames()

        # calls parent function + calculate equatorial
        m1 = mocker.patch('sofia_redux.scan.frames.frames.Frames.pointing_at')
        m2 = mocker.patch.object(frames, 'calculate_equatorial')
        frames.pointing_at(Coordinate2D())
        m1.assert_called_once()
        m2.assert_called_once()

    def test_set_zenith_tau(self):
        frames = HorizontalFrames()

        # is singular
        frames.fixed_index = 1
        frames.set_default_values()
        frames.horizontal.sin_lat = 1
        frames.set_zenith_tau(0.1)
        assert frames.zenith_tau == 0.1
        assert np.isclose(frames.transmission, 0.904837)

        # is not singular
        frames.fixed_index = np.arange(10)
        frames.set_default_values()
        frames.horizontal.sin_lat[:] = 1
        frames.set_zenith_tau(0.1)
        assert frames.zenith_tau.shape == (10,)
        assert np.allclose(frames.transmission, 0.904837)

        # specify indices
        frames.set_zenith_tau(0.2, indices=np.arange(4))
        assert frames.zenith_tau.shape == (10,)
        assert np.all(frames.zenith_tau[:4] == 0.2)
        assert np.all(frames.zenith_tau[4:] == 0.1)
        assert np.allclose(frames.transmission[:4], 0.818731)
        assert np.allclose(frames.transmission[4:], 0.904837)

    def test_horizontal_to_native_equatorial_offset(self, validated_frames):
        f = validated_frames
        f.set_parallactic_angle(90 * units.deg)
        offset = Coordinate2D([np.arange(f.size) + 1., np.arange(f.size) + 2.])
        test = offset.copy()

        # in place
        eq_off = f.horizontal_to_native_equatorial_offset(test)
        assert np.allclose(eq_off.x, -1 * offset.y)
        assert np.allclose(eq_off.y, offset.x)
        assert eq_off is test

        # not in place
        test = offset.copy()
        eq_off = f.horizontal_to_native_equatorial_offset(test, in_place=False)
        assert np.allclose(eq_off.x, -1 * offset.y)
        assert np.allclose(eq_off.y, offset.x)
        assert eq_off is not test

        # indices: offsets should be provided to match
        test = Coordinate2D([np.arange(4) + 1., np.arange(4) + 2.])
        eq_off = f.horizontal_to_native_equatorial_offset(
            test, indices=np.arange(4))
        assert np.allclose(eq_off.x, -1 * offset.y[:4])
        assert np.allclose(eq_off.y, offset.x[:4])

        # singular
        f = validated_frames[0]
        test = Coordinate2D([1, 2])
        eq_off = f.horizontal_to_native_equatorial_offset(test)
        assert np.allclose(eq_off.x, -2)
        assert np.allclose(eq_off.y, 1)

    def test_horizontal_to_equatorial_offset(self, mocker):
        frames = HorizontalFrames()
        m1 = mocker.patch.object(frames,
                                 'horizontal_to_native_equatorial_offset',
                                 return_value=Coordinate2D([1, 2]))
        # calls horizontal to native function, then scales x by -1
        offset = frames.horizontal_to_equatorial_offset(Coordinate2D())
        m1.assert_called_once()
        assert offset.x == -1
        assert offset.y == 2

    def test_equatorial_native_to_horizontal_offset(self, validated_frames):
        f = validated_frames
        f.set_parallactic_angle(-90 * units.deg)
        offset = Coordinate2D([np.arange(f.size) + 1., np.arange(f.size) + 2.])
        test = offset.copy()

        # in place
        eq_off = f.equatorial_native_to_horizontal_offset(test)
        assert np.allclose(eq_off.x, -1 * offset.y)
        assert np.allclose(eq_off.y, offset.x)
        assert eq_off is test

        # not in place
        test = offset.copy()
        eq_off = f.equatorial_native_to_horizontal_offset(test, in_place=False)
        assert np.allclose(eq_off.x, -1 * offset.y)
        assert np.allclose(eq_off.y, offset.x)
        assert eq_off is not test

        # indices: offsets should be provided to match
        test = Coordinate2D([np.arange(4) + 1., np.arange(4) + 2.])
        eq_off = f.equatorial_native_to_horizontal_offset(
            test, indices=np.arange(4))
        assert np.allclose(eq_off.x, -1 * offset.y[:4])
        assert np.allclose(eq_off.y, offset.x[:4])

        # singular
        f = validated_frames[0]
        test = Coordinate2D([1, 2])
        eq_off = f.equatorial_native_to_horizontal_offset(test)
        assert np.allclose(eq_off.x, -2)
        assert np.allclose(eq_off.y, 1)

    def test_equatorial_to_horizontal_offset(self, mocker):
        frames = HorizontalFrames()
        frames.fixed_index = 1
        frames.set_default_values()

        # scales x by -1 then calls equatorial to horizontal offset function

        # in place
        test = Coordinate2D([1, 2])
        m1 = mocker.patch.object(test, 'scale_x')
        offset = frames.equatorial_to_horizontal_offset(test, in_place=True)
        assert offset is test
        m1.assert_called_once()

        # not in place
        test = Coordinate2D([1, 2])
        m2 = mocker.patch.object(frames,
                                 'equatorial_native_to_horizontal_offset',
                                 return_value=Coordinate2D([1, 2]))

        offset = frames.equatorial_to_horizontal_offset(test, in_place=False)
        m2.assert_called_once()
        assert offset.x == 1
        assert offset.y == 2
        assert offset is not test

    def test_native_to_native_equatorial_offset(self, mocker):
        # just calls horizontal_to_native_equatorial_offset
        frames = HorizontalFrames()
        m1 = mocker.patch.object(frames,
                                 'horizontal_to_native_equatorial_offset',
                                 return_value='test')
        offset = frames.native_to_native_equatorial_offset(
            Coordinate2D([1., 2.]))
        m1.assert_called_once()
        assert offset == 'test'

    def test_native_equatorial_to_native_offset(self, mocker):
        # just calls equatorial_native_to_horizontal_offset
        frames = HorizontalFrames()
        m1 = mocker.patch.object(frames,
                                 'equatorial_native_to_horizontal_offset',
                                 return_value='test')
        offset = frames.native_equatorial_to_native_offset(
            Coordinate2D([1., 2.]))
        m1.assert_called_once()
        assert offset == 'test'

    def test_native_to_equatorial(self, validated_frames):
        frames = validated_frames
        frames.lst[:] = 10 * units.deg
        native = frames.horizontal

        eq = frames.native_to_equatorial(native)
        assert eq is not native
        assert isinstance(eq, EquatorialCoordinates)

        # equatorial to native inverts native_to_equatorial
        nn = frames.equatorial_to_native(eq)
        assert nn is not native
        assert nn is not eq
        assert isinstance(nn, HorizontalCoordinates)
        assert np.allclose(nn.x, native.x)
        assert np.allclose(nn.y, native.y)

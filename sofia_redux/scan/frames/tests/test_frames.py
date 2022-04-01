# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.coordinate_systems.equatorial_coordinates \
    import EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D
from sofia_redux.scan.frames.frames import Frames


class FramesCheck(Frames):
    """An un-abstracted frames class for testing"""

    def __init__(self):
        super().__init__()

    def get_absolute_native_coordinates(self):
        return self.equatorial

    def get_absolute_native_offsets(self):
        return self.equatorial


class DefaultsCheck(FramesCheck):

    @property
    def default_channel_fields(self):
        # override default fields to test some other types
        fields = super().default_channel_fields
        fields.update({
            'quantity': 1.0 * units.arcsec,
            'tuple_coord_quantity': (Coordinate2D, 1.0 * units.arcsec),
            'tuple_coord_unit': (Coordinate2D, units.arcsec),
            'tuple_coord_str': (Coordinate2D, 'arcsec')
        })
        return fields

    @property
    def readout_attributes(self):
        return {'data', 'transmission', 'equatorial', 'test'}


@pytest.fixture
def offsets():
    x, y = np.meshgrid([-2, -1, 0, 1, 2], [-60, -45, -30, 0, 30, 45, 60])
    offsets = EquatorialCoordinates([x.ravel(), y.ravel()], unit='degree')
    return offsets


@pytest.fixture
def validated_frames(populated_integration):
    frames = FramesCheck()
    nframe = 10
    frames.initialize(populated_integration, nframe)
    frames.validate()
    return frames


@pytest.fixture
def projector():
    projection = DefaultProjection2D()
    projection.reference = Coordinate2D([5, 5])
    projector = AstroProjector(projection)
    return projector


class TestFrames(object):

    def test_init(self):
        # can't init abstract class
        with pytest.raises(TypeError):
            Frames()

        # okay with native coordinate functions implemented
        FramesCheck()

    def test_property_defaults(self):
        frames = FramesCheck()

        # quick checks on property defaults for unpopulated frames
        assert 'integration' in frames.referenced_attributes
        assert 'data' in frames.readout_attributes
        assert np.isnan(frames.default_field_types['mjd'])
        assert frames.default_channel_fields['sample_flag'] == 0
        assert 'frame_fixed_channels' in frames.internal_attributes
        assert frames.channel_size == 0
        assert frames.scan is None
        assert frames.info is None
        assert frames.astrometry is None
        assert frames.scan_equatorial is None
        assert frames.channels is None
        assert frames.configuration is None
        assert 'integration' in frames.special_fields
        assert frames.channel_fixed_index is None

    def test_channel_size(self):
        frames = FramesCheck()
        frames.data = np.zeros((121, 10))
        assert frames.channel_size == 10

    def test_populated_properties(self, populated_integration):
        integ = populated_integration
        frames = FramesCheck()
        frames.integration = integ
        assert frames.scan is integ.scan
        assert frames.info is integ.scan.info
        assert frames.astrometry is integ.scan.info.astrometry
        assert frames.scan_equatorial is integ.scan.info.astrometry.equatorial
        assert frames.channels is integ.channels
        assert frames.configuration is integ.scan.info.configuration
        assert frames.channel_fixed_index is integ.channels.data.fixed_index

    def test_insert_blanks(self):
        frames = FramesCheck()
        nframe = 10
        nchannel = 20
        arr = np.arange(nframe * nchannel).reshape(nframe, nchannel)
        frames.data = arr
        frames.valid = np.full(nframe, True)

        frames.insert_blanks(np.array([2, 4, 6]))
        assert frames.data.shape == (nframe + 3, nchannel)
        assert np.all(frames.data[0] == arr[0])
        assert np.all(frames.data[2] == 0)
        assert np.all(frames.data[4 + 1] == 0)
        assert np.all(frames.data[6 + 2] == 0)
        assert np.all(frames.data[-1] == arr[-1])

        assert frames.valid.shape == (nframe + 3,)
        assert np.sum(frames.valid) == nframe
        assert not frames.valid[2]
        assert not frames.valid[4 + 1]
        assert not frames.valid[6 + 2]

    def test_set_default_channel_values(self):
        frames = DefaultsCheck()
        nframe = 10
        nchannel = 20
        frames.set_frame_size(nframe)

        frames.set_default_channel_values(nchannel)

        assert frames.data.shape == (nframe, nchannel)
        assert frames.sample_flag.shape == (nframe, nchannel)
        assert frames.source_index.shape == (nframe, nchannel)
        assert frames.map_index.shape == (nframe, nchannel)
        assert frames.sample_equatorial.shape == (nframe, nchannel)
        assert frames.tuple_coord_quantity.shape == (nframe, nchannel)
        assert frames.tuple_coord_unit.shape == (nframe, nchannel)
        assert frames.tuple_coord_str.shape == (nframe, nchannel)

    def test_set_channels(self, populated_integration):
        nframe = 20
        channels = populated_integration.channels
        nchannel = channels.data.size

        frames = FramesCheck()
        frames.set_frame_size(nframe)

        with pytest.raises(ValueError) as err:
            frames.set_channels()
        assert 'No channels supplied' in str(err)

        frames.integration = populated_integration
        frames.set_channels()

        # data initialized
        assert frames.data.shape == (nframe, nchannel)
        assert np.allclose(frames.frame_fixed_channels, np.arange(nchannel))

        # call again: indices validated
        frames.set_channels()
        assert np.allclose(frames.frame_fixed_channels, np.arange(nchannel))

    def test_find_channel_fixed_indices(self, populated_integration):
        channels = populated_integration.channels
        nchannel = channels.data.size
        fixed_indices = np.arange(nchannel)
        frames = FramesCheck()

        # zero size array if no channels
        idx = frames.find_channel_fixed_indices(fixed_indices)
        assert idx.size == 0

        # returns index array if all requested
        frames.integration = populated_integration
        idx = frames.find_channel_fixed_indices(fixed_indices)
        assert idx.shape == (nchannel,)
        assert np.allclose(idx, fixed_indices)

        # otherwise selects from indices, removing inappropriate ones
        fixed_indices = np.array([0, 1, 2, 3, nchannel + 4])
        idx = frames.find_channel_fixed_indices(fixed_indices)
        assert idx.shape == (4,)
        assert np.allclose(idx, [0, 1, 2, 3])

        # unless cull = False: then bad ones are replaced with -1
        idx = frames.find_channel_fixed_indices(fixed_indices, cull=False)
        assert idx.shape == (5,)
        assert np.allclose(idx, [0, 1, 2, 3, -1])

    def test_get_frame_count(self):
        frames = FramesCheck()
        assert frames.get_frame_count() == 0

        nframe = 20
        frames.set_frame_size(nframe)
        assert frames.get_frame_count() == nframe

        assert frames.get_frame_count(keep_flag=1) == 0
        assert frames.get_frame_count(discard_flag=1) == nframe
        assert frames.get_frame_count(match_flag=1) == 0

    def test_set_transmission(self):
        frames = FramesCheck()

        # is singular
        frames.fixed_index = 1
        frames.set_default_values()
        frames.set_transmission(0.1)
        assert frames.transmission == 0.1

        # is not singular
        frames.fixed_index = np.arange(10)
        frames.set_default_values()
        frames.set_transmission(0.1)
        assert frames.transmission.shape == (10,)
        assert np.all(frames.transmission == 0.1)

        # specify indices
        frames.set_transmission(0.2, indices=np.arange(4))
        assert frames.transmission.shape == (10,)
        assert np.all(frames.transmission[:4] == 0.2)
        assert np.all(frames.transmission[4:] == 0.1)

    def test_slim(self, populated_integration):
        # no integration: raises error in validation
        frames = FramesCheck()
        with pytest.raises(ValueError) as err:
            frames.slim()
        assert 'Channel fixed indices for frames not set' in str(err)

        # integration but channels not set: same
        frames.integration = populated_integration
        with pytest.raises(ValueError) as err:
            frames.slim()
        assert 'Channel fixed indices for frames not set' in str(err)

        # works if channels specified
        frames.source_index = 0
        frames.map_index = 0
        frames.slim(channels=populated_integration.channels)
        assert frames.source_index is None
        assert frames.map_index is None

        # works if channels previously set
        frames.source_index = 0
        frames.map_index = 0
        frames.slim()
        assert frames.source_index is None
        assert frames.map_index is None

    def test_validate_channel_indices(self, populated_integration):
        frames = FramesCheck()
        frames.integration = populated_integration
        frames.set_channels()
        nchannel = populated_integration.channels.data.size

        # no effect if fixed indices are same
        frames.validate_channel_indices()
        assert np.allclose(frames.channel_fixed_index,
                           frames.frame_fixed_channels)
        assert frames.channel_fixed_index.shape == (nchannel,)
        assert frames.data.shape == (0, nchannel)
        assert frames.sample_flag.shape == (0, nchannel)

        # if fixed indices not the same, they are fixed to be the same,
        # data is reset, None attributes are left alone
        frames.frame_fixed_channels = np.arange(10)
        frames.sample_flag = None
        frames.validate_channel_indices()
        assert np.allclose(frames.channel_fixed_index,
                           frames.frame_fixed_channels)
        assert frames.channel_fixed_index.shape == (121,)
        assert frames.data.shape == (0, 10)
        assert frames.sample_flag is None

    def test_jackknife(self, mocker):
        frames = FramesCheck()
        nframe = 20
        frames.set_frame_size(nframe)
        assert frames.sign.size == nframe
        assert np.allclose(frames.sign, 1)

        # mock random to always return < 0.5
        mocker.patch('sofia_redux.scan.frames.frames.'
                     'np.random.random', return_value=np.full(nframe, 0.1))

        frames.jackknife()
        assert np.allclose(frames.sign, -1)

    def test_get_source_gain(self):
        frames = FramesCheck()
        nframe = 20
        frames.set_frame_size(nframe)
        frames.set_transmission(0.5)

        with pytest.raises(ValueError) as err:
            frames.get_source_gain(1)
        assert 'does not define' in str(err)

        gain = frames.get_source_gain(frames.flagspace.flags.TOTAL_POWER)
        assert gain.shape == (20,)
        assert np.allclose(gain, 0.5)

    def test_validate(self, mocker, populated_integration):
        frames = FramesCheck()
        frames.integration = populated_integration
        frames.set_channels()
        frames.set_frame_size(20)

        # just check that validate_frames numba function is called
        m1 = mocker.patch('sofia_redux.scan.frames.frames.'
                          'frames_numba_functions.validate_frames')
        frames.validate()
        m1.assert_called_once()

    def test_get_equatorial(self, validated_frames, offsets):
        nframe = validated_frames.size
        eq = validated_frames.get_equatorial(offsets)
        assert eq.x.shape == (nframe, offsets.x.size)
        assert eq.y.shape == (nframe, offsets.y.size)

    def test_get_equatorial_native_offset(self, validated_frames, offsets):
        nframe = validated_frames.size

        eq = validated_frames.get_equatorial_native_offset(offsets)
        assert eq.x.shape == (nframe, offsets.x.size)
        assert eq.y.shape == (nframe, offsets.y.size)

        # singular
        eq = validated_frames[0].get_equatorial_native_offset(offsets)
        assert eq.x.shape == (offsets.x.size,)
        assert eq.y.shape == (offsets.y.size,)

    def test_get_native_offset(self, validated_frames, offsets):
        nframe = validated_frames.size

        nt = validated_frames.get_native_offset(offsets)
        assert nt.x.shape == (nframe, offsets.x.size)
        assert nt.y.shape == (nframe, offsets.y.size)

    def test_get_focal_plane_offset(self, validated_frames, offsets):
        nframe = validated_frames.size

        fp = validated_frames.get_focal_plane_offset(offsets)
        assert fp.x.shape == (nframe, offsets.x.size)
        assert fp.y.shape == (nframe, offsets.y.size)

        # singular
        fp = validated_frames[0].get_focal_plane_offset(offsets)
        assert fp.x.shape == (offsets.x.size,)
        assert fp.y.shape == (offsets.y.size,)

    def test_get_apparent_equatorial(self, validated_frames):
        ap = validated_frames.get_apparent_equatorial()
        assert isinstance(ap, EquatorialCoordinates)

        # singular
        ap = validated_frames[0].get_apparent_equatorial()
        assert isinstance(ap, EquatorialCoordinates)

        # with apparent
        ap = validated_frames.get_apparent_equatorial(apparent=ap)
        assert isinstance(ap, EquatorialCoordinates)

    def test_pointing_at(self, mocker, validated_frames):
        # for FrameCheck, absolute native functions just return equatorial
        # these should have their subtract functions called
        m1 = mocker.patch.object(validated_frames.equatorial, 'subtract')
        m2 = mocker.patch.object(validated_frames.equatorial,
                                 'subtract_offset')

        offset = Coordinate2D([0, 0])
        validated_frames.pointing_at(offset)
        m1.assert_called_once()
        m2.assert_called_once()

        # specify indices
        validated_frames.pointing_at(offset, indices=np.arange(4))

        # singular: indices ignored
        validated_frames[0].pointing_at(offset, indices=np.arange(4))

    def test_scale(self):
        frames = FramesCheck()

        # non singular
        arr = np.arange(10 * 20, dtype=float).reshape(10, 20)
        frames.data = arr.copy()
        frames.fixed_index = np.arange(10)
        frames.scale(2)
        assert np.allclose(frames.data, arr * 2)
        frames.scale(2, indices=np.arange(4))
        assert np.allclose(frames.data[4:], arr[4:] * 2)
        assert np.allclose(frames.data[:4], arr[:4] * 4)
        frames.invert()
        assert np.allclose(frames.data[4:], arr[4:] * -2)
        assert np.allclose(frames.data[:4], arr[:4] * -4)
        frames.scale(0)
        assert np.allclose(frames.data, 0)

        # singular
        frames.data = 4
        frames.fixed_index = 0
        frames.scale(2)
        assert frames.data == 8
        frames.scale(2, indices=0)
        assert frames.data == 16
        frames.invert()
        assert frames.data == -16
        frames.scale(0)
        assert frames.data == 0

    def test_rotation(self, validated_frames):
        angle = validated_frames.get_rotation()
        assert angle.size == validated_frames.size
        assert np.allclose(angle, 0 * units.radian)

        validated_frames.set_rotation(45 * units.deg)
        angle = validated_frames.get_rotation()
        assert np.allclose(angle, 45 * units.deg)

        # indices
        angle = validated_frames.get_rotation(indices=np.arange(4))
        assert angle.size == 4
        assert np.allclose(angle, 45 * units.deg)

        # singular
        frames = validated_frames[0]
        frames.set_rotation(20 * units.deg)
        angle = frames.get_rotation()
        assert np.isclose(angle, 20 * units.deg)

        # no quantities
        frames.set_rotation(0.1)
        angle = frames.get_rotation()
        assert np.isclose(angle, 0.1 * units.rad)

    def test_get_native_xy(self, validated_frames):
        positions = Coordinate2D([[1, 2, 3], [4, 5, 6]])

        # no rotation
        x = validated_frames.get_native_x(positions)
        y = validated_frames.get_native_y(positions)
        xy = validated_frames.get_native_xy(positions)

        assert np.allclose(x, positions.x)
        assert np.allclose(y, positions.y)
        assert np.allclose(x * units.arcsec, xy.x)
        assert np.allclose(y * units.arcsec, xy.y)

        # rotation
        validated_frames.set_rotation(90 * units.deg)
        x = validated_frames.get_native_x(positions)
        y = validated_frames.get_native_y(positions)
        xy = validated_frames.get_native_xy(positions)

        assert np.allclose(x, -1 * positions.y)
        assert np.allclose(y, positions.x)
        assert np.allclose(x * units.arcsec, xy.x)
        assert np.allclose(y * units.arcsec, xy.y)

        # singular frames
        validated_frames.set_rotation(90 * units.deg)
        x = validated_frames[0].get_native_x(positions)
        y = validated_frames[0].get_native_y(positions)
        xy = validated_frames[0].get_native_xy(positions)

        assert np.allclose(x, -1 * positions.y)
        assert np.allclose(y, positions.x)
        assert np.allclose(x * units.arcsec, xy.x)
        assert np.allclose(y * units.arcsec, xy.y)

        # single position
        positions = Coordinate2D([1, 4])
        x = validated_frames[0].get_native_x(positions)
        y = validated_frames[0].get_native_y(positions)
        xy = validated_frames[0].get_native_xy(positions)
        assert np.isclose(x, -4)
        assert np.isclose(y, 1)
        assert np.isclose(xy.x, -4 * units.arcsec)
        assert np.isclose(xy.y, 1 * units.arcsec)

    def test_add_data_from(self):
        frames = FramesCheck()
        arr = np.arange(10 * 20, dtype=float).reshape(10, 20)
        frames.data = arr.copy()
        frames.sample_flag = np.full(10, 0)
        frames.fixed_index = np.arange(10)

        other_frames = FramesCheck()
        other_frames.data = arr * 2.
        other_frames.sample_flag = np.full(10, 1)
        other_frames.fixed_index = np.arange(10)

        frames.add_data_from(other_frames)
        assert np.allclose(frames.data, arr + arr * 2)
        assert np.all(frames.sample_flag == 1)

        # with scaling
        frames.add_data_from(other_frames, scaling=2)
        assert np.allclose(frames.data, arr + arr * 2 + arr * 4)
        assert np.all(frames.sample_flag == 1)

        # singular
        frames.fixed_index = 0
        frames.data = 4
        frames.sample_flag = 0
        other_frames.data = 6
        other_frames.sample_flag = 1
        frames.add_data_from(other_frames)
        assert frames.data == 10
        assert frames.sample_flag == 1

        # with scaling
        frames.add_data_from(other_frames, scaling=2)
        assert frames.data == 22
        assert frames.sample_flag == 1

    def test_project(self, mocker, validated_frames, projector):
        positions = Coordinate2D([1, 4], unit=units.arcsec)

        # placeholder projector always returns 5, 5
        offset = validated_frames.project(positions, projector)
        assert offset.x == 5
        assert offset.y == 5

        # nonsidereal
        validated_frames.info.astrometry.is_nonsidereal = True
        offset = validated_frames.project(positions, projector)
        assert offset.x == 5
        assert offset.y == 5

        # mock focal plane projector, check that functions are called
        mocker.patch.object(projector, 'is_focal_plane', return_value=True)
        projector.coordinates.add_native_offset = lambda *args: None
        m1 = mocker.patch.object(validated_frames, 'get_focal_plane_offset')
        offset = validated_frames.project(positions, projector)
        assert offset.x == 5
        assert offset.y == 5
        m1.assert_called_once()

    def test_native_to_native_equatorial_offset(self):
        frames = FramesCheck()
        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        test = offset.copy()

        # just returns input for generic Frames
        offset_reference = frames.native_to_native_equatorial_offset(
            offset, in_place=True)
        assert offset_reference is offset
        assert np.allclose(offset_reference.x, test.x)
        assert np.allclose(offset_reference.y, test.y)

        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        offset_copy = frames.native_to_native_equatorial_offset(
            offset, in_place=False)
        assert offset_copy is not offset
        assert np.allclose(offset_copy.x, test.x)
        assert np.allclose(offset_copy.y, test.y)

    def test_native_to_equatorial_offset(self):
        frames = FramesCheck()
        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        test = offset.copy()

        # returns input with x sign inverted
        offset_reference = frames.native_to_equatorial_offset(
            offset, in_place=True)
        assert offset_reference is offset
        assert np.allclose(offset_reference.x, -1 * test.x)
        assert np.allclose(offset_reference.y, test.y)

        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        offset_copy = frames.native_to_equatorial_offset(
            offset, in_place=False)
        assert offset_copy is not offset
        assert np.allclose(offset_copy.x, -1 * test.x)
        assert np.allclose(offset_copy.y, test.y)

    def test_native_equatorial_to_native_offset(self):
        frames = FramesCheck()
        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        test = offset.copy()

        # just returns input for generic Frames
        offset_reference = frames.native_equatorial_to_native_offset(
            offset, in_place=True)
        assert offset_reference is offset
        assert np.allclose(offset_reference.x, test.x)
        assert np.allclose(offset_reference.y, test.y)

        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        offset_copy = frames.native_equatorial_to_native_offset(
            offset, in_place=False)
        assert offset_copy is not offset
        assert np.allclose(offset_copy.x, test.x)
        assert np.allclose(offset_copy.y, test.y)

    def test_equatorial_to_native_offset(self):
        frames = FramesCheck()
        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        test = offset.copy()

        # returns input with x sign inverted
        offset_reference = frames.equatorial_to_native_offset(
            offset, in_place=True)
        assert offset_reference is offset
        assert np.allclose(offset_reference.x, -1 * test.x)
        assert np.allclose(offset_reference.y, test.y)

        offset = Coordinate2D([[1, 2, 3], [4, 5, 6]])
        offset_copy = frames.equatorial_to_native_offset(
            offset, in_place=False)
        assert offset_copy is not offset
        assert np.allclose(offset_copy.x, -1 * test.x)
        assert np.allclose(offset_copy.y, test.y)

    def test_native_to_equatorial(self):
        frames = FramesCheck()
        native = EquatorialCoordinates([[1, 2, 3], [4, 5, 6]])

        # for default frames, native and equatorial are the same
        eq = frames.native_to_equatorial(native)
        assert eq is not native
        assert isinstance(eq, EquatorialCoordinates)
        assert np.allclose(eq.x, native.x)
        assert np.allclose(eq.y, native.y)

        nn = frames.equatorial_to_native(eq)
        assert nn is not native
        assert nn is not eq
        assert isinstance(nn, EquatorialCoordinates)
        assert np.allclose(nn.x, native.x)
        assert np.allclose(nn.y, native.y)

    def test_get_native_offset_from(self, mocker):
        frames = FramesCheck()

        # just calls the equatorial native offset
        m1 = mocker.patch.object(frames, 'get_equatorial_native_offset_from')
        frames.get_native_offset_from(EquatorialCoordinates())
        m1.assert_called_once()

    def test_get_equatorial_native_offset_from(self, validated_frames):
        reference = EquatorialCoordinates([0, 0], unit=units.deg)
        offset = validated_frames.get_equatorial_native_offset_from(reference)
        assert offset.x.size == validated_frames.size
        assert offset.y.size == validated_frames.size

        # indices
        offset = validated_frames.get_equatorial_native_offset_from(
            reference, indices=np.arange(4))
        assert offset.x.size == 4
        assert offset.y.size == 4

        # singular
        validated_frames.set_frame_size(1)
        validated_frames.fixed_index = 0
        offset = validated_frames.get_equatorial_native_offset_from(
            reference)
        assert offset.x.size == 1
        assert offset.y.size == 1

    @pytest.mark.parametrize(['reference', 'first', 'last'],
                             [(0, 0, None), (1, 1, 0),
                              (10, 10, 9),
                              (-1, 19, 18), (-2, 18, 17),
                              (18, 18, 17), (19, 19, 18)])
    def test_get_frame_index_value(self, reference, first, last):
        nframe = 20
        frames = FramesCheck()
        frames.set_frame_size(nframe)
        frames.transmission = np.arange(nframe)

        assert frames.get_first_frame_index() == 0
        assert frames.get_last_frame_index() == nframe - 1
        assert frames.get_first_frame_value('transmission') == 0
        assert frames.get_last_frame_value('transmission') == nframe - 1

        assert frames.get_first_frame_index(reference) == first
        assert frames.get_first_frame_value_from(
            reference, 'transmission') == first
        if last is not None:
            assert frames.get_last_frame_index(reference) == last
            assert frames.get_last_frame_value_from(
                reference, 'transmission') == last

        if first < nframe - 1:
            frames.valid[reference] = False
            assert frames.get_first_frame_index(reference) == first + 1
            assert frames.get_first_frame_value_from(
                reference, 'transmission') == first + 1
        if last is not None and last > 0:
            frames.valid[reference - 1] = False
            assert frames.get_last_frame_index(reference) == last - 1
            assert frames.get_last_frame_value_from(
                reference, 'transmission') == last - 1

    def test_frame_value_errors(self):
        frames = FramesCheck()
        frames.set_frame_size(20)
        with pytest.raises(ValueError) as err:
            frames.get_first_frame_value('test')
        assert 'does not contain test field' in str(err)

        with pytest.raises(ValueError) as err:
            frames.get_last_frame_value('test')
        assert 'does not contain test field' in str(err)

        with pytest.raises(ValueError) as err:
            frames.get_first_frame_value_from(-1, 'test')
        assert 'does not contain test field' in str(err)

        with pytest.raises(ValueError) as err:
            frames.get_last_frame_value_from(-1, 'test')
        assert 'does not contain test field' in str(err)

    def test_get_frame(self, validated_frames):
        first = validated_frames.get_first_frame()
        assert isinstance(first, FramesCheck)
        assert first.fixed_index == 0

        last = validated_frames.get_last_frame()
        assert isinstance(last, FramesCheck)
        assert last.fixed_index == validated_frames.fixed_index[-1]

        # invalidate some
        validated_frames.valid[0] = False
        validated_frames.valid[-1] = False
        first = validated_frames.get_first_frame()
        assert first.fixed_index == 1
        last = validated_frames.get_last_frame()
        assert last.fixed_index == validated_frames.fixed_index[-2]

        # specify reference
        first = validated_frames.get_first_frame(reference=5)
        assert first.fixed_index == 5
        last = validated_frames.get_last_frame(reference=5)
        assert last.fixed_index == 4

    def test_add_dependents(self, mocker):
        frames = FramesCheck()

        # just check that numba function is called
        m1 = mocker.patch('sofia_redux.scan.frames.frames.'
                          'frames_numba_functions.add_dependents')

        frames.add_dependents(None)
        m1.assert_called_with(dependents=None, dp=None, frame_valid=None,
                              start_frame=None, end_frame=None,
                              subtract=False)

        frames.remove_dependents(None)
        m1.assert_called_with(dependents=None, dp=None, frame_valid=None,
                              start_frame=None, end_frame=None,
                              subtract=True)

    def test_shift_frames(self):
        frames = DefaultsCheck()
        nframes = 20
        frames.set_frame_size(nframes)
        assert np.all(frames.valid)
        frames.data = np.arange(nframes, dtype=float)

        # add some attributes to test
        frames.set_transmission(0.1)
        assert np.allclose(frames.transmission, 0.1)

        frames.equatorial = EquatorialCoordinates([np.arange(nframes),
                                                   np.arange(nframes)])
        assert np.all(~np.isnan(frames.equatorial.x))
        assert np.all(~np.isnan(frames.equatorial.y))

        frames.test = None
        assert frames.test is None

        # shift forward 1
        frames.shift_frames(1)

        # first frame is invalid: attributes are filled as appropriate
        assert not frames.valid[0]
        assert frames.transmission[0] == 0.1
        assert np.isnan(frames.equatorial.x[0])
        assert np.isnan(frames.equatorial.y[0])
        assert frames.test is None
        assert frames.data[0] == 0
        assert np.allclose(frames.data[1:], np.arange(nframes - 1))

        # shift back 1: does not re-validate first frame, but does
        # shift values back
        frames.shift_frames(-1)
        assert not frames.valid[0]
        assert frames.transmission[0] == 0.1
        assert not np.isnan(frames.equatorial.x[0])
        assert not np.isnan(frames.equatorial.y[0])
        assert frames.test is None
        assert not frames.valid[-1]
        assert frames.transmission[0] == 0.1
        assert not np.isnan(frames.equatorial.x[0])
        assert not np.isnan(frames.equatorial.y[0])
        assert frames.test is None
        assert np.allclose(frames.data[:-1], np.arange(nframes - 1))
        assert frames.data[-1] == nframes - 2

    def test_set_from_downsampled(self, mocker):
        frames = FramesCheck()

        # just check that numba function is called
        m1 = mocker.patch('sofia_redux.scan.frames.frames.'
                          'frames_numba_functions.downsample_data',
                          return_value=(1, 2))
        frames.set_from_downsampled(frames[0], None, [3], None)
        m1.assert_called_once()
        # data and flag set from numba function return values
        assert frames.data == 1
        assert frames.sample_flag == 2
        # valid set from input
        assert frames.valid == [3]

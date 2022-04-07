# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pandas
import pytest
import numpy as np

from sofia_redux.scan.channels.channel_data.channel_data import ChannelData
from sofia_redux.scan.custom.example.channels.channel_data.channel_data \
    import ExampleChannelData


class ChannelDataCheck(ChannelData):
    """An un-abstracted class for testing"""

    def __init__(self):
        super().__init__()

    def read_channel_data_file(self, filename):
        pass

    def get_overlap_indices(self, radius):
        pass

    def get_overlap_distances(self, overlap_indices):
        pass

    def get_pixel_count(self):
        pass

    def get_pixels(self):
        pass

    def get_mapping_pixels(self, indices=None, name=None, keep_flag=None,
                           discard_flag=None, match_flag=None):
        pass


class TestChannelData(object):

    def test_init(self):
        # can't init abstract class
        with pytest.raises(TypeError):
            ChannelData()

        # okay with abstract functions implemented
        ChannelDataCheck()

    def test_property_defaults(self):
        data = ChannelDataCheck()

        # quick checks on property defaults for unpopulated object
        assert data.info is None
        assert data.configuration is None
        assert 'channels' in data.referenced_attributes
        assert np.isnan(data.default_field_types['one_over_f_stat'])

    def test_instance_from_instrument_name(self):
        data = ChannelData.instance_from_instrument_name('example')
        assert isinstance(data, ExampleChannelData)

    def test_read_pixel_data(self, mocker):
        # verify steps called: read, set flags, set data, validate
        data = ChannelDataCheck()
        data.channel_id = [1, 2, 3]
        m1 = mocker.patch.object(data, 'read_channel_data_file',
                                 return_value={1: 1, 2: 2, 3: 3})
        m2 = mocker.patch.object(data, 'set_flags')
        m3 = mocker.patch.object(data, 'set_channel_data')
        m4 = mocker.patch.object(data, 'validate_pixel_data')

        data.read_pixel_data('test.file')
        assert m1.called_with('test.file')
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert m3.call_count == 3
        assert m4.call_count == 1

    def test_set_channel_data(self):
        data = ChannelDataCheck()
        nchannel = 10
        data.gain = np.full(nchannel, 1.0)
        data.weight = np.full(nchannel, 1.0)
        data.flag = np.full(nchannel, 0)

        # no change for None info
        data.set_channel_data(5, None)
        assert data.gain[5] == 1.0
        assert data.weight[5] == 1.0
        assert data.flag[5] == 0

        # pass info to set
        info = {'gain': 2.0, 'weight': 0.0, 'flag': 1}
        data.set_channel_data(5, info)
        assert data.gain[5] == 2.0
        assert data.weight[5] == 0.0
        assert data.flag[5] == 1

    def test_validate_pixel_data(self, mocker, populated_data):
        data = populated_data
        flag = data.flag.copy()
        weight = data.weight.copy()

        # no change for default options
        data.validate_pixel_data()
        assert np.allclose(data.flag, flag)
        assert np.allclose(data.weight, weight)

        # set critical flags to None: no change
        data.configuration.set_option('pixels.criticalflags', None)
        data.validate_pixel_data()
        assert np.allclose(data.flag, flag)
        assert np.allclose(data.weight, weight)

        # same if critical flags function returns None
        del data.configuration['pixels.criticalflags']
        m1 = mocker.patch.object(data.flagspace, 'critical_flags',
                                 return_value=None)
        data.validate_pixel_data()
        m1.assert_called_once()
        assert np.allclose(data.flag, flag)
        assert np.allclose(data.weight, weight)

    def test_set_hardware_gain(self, populated_data):
        nchannel = populated_data.size
        populated_data.set_hardware_gain(populated_data.info)
        assert populated_data.hardware_gain.size == nchannel
        assert np.allclose(populated_data.hardware_gain,
                           populated_data.info.instrument.gain)

    def test_kill_channels(self, populated_data):
        # no flagged channels
        assert np.all(populated_data.flag == 0)
        populated_data.kill_channels()
        assert np.all(populated_data.flag == 0)

        # flag some channels
        populated_data.flag[:4] = 4
        populated_data.kill_channels(flag=4)
        # specified data is marked DEAD (flag=1)
        assert np.all(populated_data.flag[:4] == 1)
        assert np.all(populated_data.flag[4:] == 0)

    def test_flag_field(self, capsys, populated_data):
        # flag non existent field
        populated_data.flag_field('test', ['5:*'])
        assert np.all(populated_data.flag == 0)
        assert 'does not have test attribute' in capsys.readouterr().err

        # set gain so it can be flagged
        nchannel = populated_data.size
        populated_data.gain = np.arange(nchannel, dtype=float)
        populated_data.flag_field('gain', ['*:5', '110-*', '50', 'b-a-d'])
        # dead at 5 and below, 110 and above, and at 50
        assert np.all(populated_data.flag[:6] == 1)
        assert np.all(populated_data.flag[110:] == 1)
        assert np.all(populated_data.flag[50] == 1)
        # otherwise okay
        assert np.all(populated_data.flag[6:50] == 0)
        assert np.all(populated_data.flag[51:110] == 0)
        # bad range ignored
        assert 'Could not parse flag: gain (b-a-d)' in capsys.readouterr().err

    def test_flag_fields(self, populated_data):
        # set some data to test
        nchannel = populated_data.size
        populated_data.gain = np.arange(nchannel, dtype=float)
        populated_data.weight = np.arange(nchannel, dtype=float)

        # ignored if not a dict
        fields = ['gain', 5]
        populated_data.flag_fields(fields)
        assert np.all(populated_data.flag == 0)

        # flag gain and weight
        fields = {'gain': '5', 'weight': '6-8'}
        populated_data.flag_fields(fields)
        assert np.all(populated_data.flag[:5] == 0)
        assert np.all(populated_data.flag[5:9] == 1)
        assert np.all(populated_data.flag[9:] == 0)

    def test_set_blind_channels(self, populated_data):
        # set some blind channels by index
        populated_data.set_blind_channels(np.arange(4))
        assert np.all(populated_data.flag[:4] == 2)

        # calling again sets previously blind channels to dead
        populated_data.set_blind_channels(np.arange(4) + 4)
        assert np.all(populated_data.flag[:4] == 1)
        assert np.all(populated_data.flag[4:8] == 2)

    def test_flag_channel_list(self, populated_data):
        # kill channels by index list or string
        populated_data.flag_channel_list(['0-3', '4'])
        assert np.all(populated_data.flag[:5] == 1)
        populated_data.flag_channel_list('5-7,8,9,10')
        assert np.all(populated_data.flag[:11] == 1)

    def test_set_uniform_gains(self, populated_data):
        nchannel = populated_data.size
        populated_data.gain = np.arange(nchannel, dtype=float)
        populated_data.coupling = np.arange(nchannel, dtype=float)
        populated_data.weight = np.arange(nchannel, dtype=float)

        # gain and coupling by default
        populated_data.set_uniform_gains()
        assert populated_data.gain.size == nchannel
        assert np.all(populated_data.gain == 1)
        assert populated_data.coupling.size == nchannel
        assert np.all(populated_data.coupling == 1)

        # other fields allowed
        populated_data.set_uniform_gains(field='weight')
        assert np.all(populated_data.weight == 1)
        assert populated_data.weight.size == nchannel

    def test_flatten_weights(self, populated_data):
        nchannel = populated_data.size
        populated_data.gain = np.arange(nchannel, dtype=float)
        populated_data.weight = np.arange(nchannel, dtype=float)

        populated_data.flatten_weights()
        assert np.allclose(populated_data.weight, 90.37344)

        # if all gains 0, weight set to 1
        populated_data.gain[:] = 0
        populated_data.flatten_weights()
        assert np.allclose(populated_data.weight, 1.0)

    def test_get_filtering(self, populated_data, populated_integration):
        nchannel = populated_data.size
        filt = populated_data.get_filtering(populated_integration)
        assert filt.size == nchannel
        assert np.allclose(filt, 1.0)

        populated_data.direct_filtering = np.arange(nchannel)
        filt = populated_data.get_filtering(populated_integration)
        assert np.allclose(filt, np.arange(nchannel))

    def test_apply_info(self, mocker):
        data = ChannelDataCheck()
        m1 = mocker.patch.object(data, 'set_hardware_gain')
        data.apply_info('test')
        assert m1.called_with('test')
        assert m1.call_count == 1

    def test_get_typical_gain_magnitude(self, populated_data):
        nchannel = populated_data.size
        gains = np.arange(nchannel, dtype=float)

        # without units, or with dimensionless units
        mag = populated_data.get_typical_gain_magnitude(gains)
        assert np.isclose(mag, 52.28311)
        mag = populated_data.get_typical_gain_magnitude(
            gains * units.dimensionless_unscaled)
        assert np.isclose(mag, 52.28311)

        # with units
        mag = populated_data.get_typical_gain_magnitude(gains * units.Jy)
        assert np.isclose(mag, 52.28311 * units.Jy)

        # with no kept data
        mag = populated_data.get_typical_gain_magnitude(gains, keep_flag=1)
        assert np.isclose(mag, 1)
        mag = populated_data.get_typical_gain_magnitude(gains * units.Jy,
                                                        discard_flag=0)
        assert np.isclose(mag, 1 * units.Jy)

    def test_clear_overlaps(self, overlaps):
        data = ChannelDataCheck()

        # no overlaps: no op
        data.overlaps = None
        data.clear_overlaps()
        assert data.overlaps is None

        # set a sparse overlaps matrix
        data.overlaps = overlaps
        assert not np.allclose(data.overlaps.data, 0)
        assert data.overlaps.size > 0

        # clear it
        data.clear_overlaps()
        assert np.allclose(data.overlaps.data, 0)
        assert np.allclose(data.overlaps.data.size, 0)

    def test_calculate_overlaps(self, mocker):
        data = ChannelDataCheck()

        # check for call sequence: overlap indices,
        # overlap distances, overlap values
        m1 = mocker.patch.object(data, 'get_overlap_indices',
                                 return_value=[1, 2, 3])
        m2 = mocker.patch.object(data, 'get_overlap_distances',
                                 return_value=('test', units.arcsec))
        m3 = mocker.patch.object(data, 'calculate_overlap_values')

        data.calculate_overlaps(1.0 * units.arcsec)
        assert m1.call_count == 1
        assert m1.called_with(2.0 * units.arcsec)
        assert m2.call_count == 1
        assert m2.called_with([1, 2, 3])
        assert m3.call_count == 1
        assert m3.called_with('test', 1.0 * units.arcsec)

    def test_add_dependents(self):
        data = ChannelDataCheck()
        data.dependents = np.arange(10)
        data.add_dependents(np.arange(10))
        assert np.allclose(data.dependents, 2 * np.arange(10))
        data.remove_dependents(np.arange(10))
        assert np.allclose(data.dependents, np.arange(10))
        data.remove_dependents(np.arange(10))
        assert np.allclose(data.dependents, 0)

    def test_to_string(self):
        data = ChannelDataCheck()
        data.channel_id = np.arange(3)
        data.gain = np.arange(3)
        data.weight = np.arange(3)
        data.flag = np.arange(3)
        s = data.to_string()
        assert "ch\tgain\tweight\tflag" in s
        assert "0\t0.000\t0.000e+00\t-" in s
        assert "1\t1.000\t1.000e+00\tX" in s
        assert "2\t2.000\t2.000e+00\tB" in s

        df = data.to_string(frame=True)
        assert isinstance(df, pandas.DataFrame)
        assert len(df) == 3

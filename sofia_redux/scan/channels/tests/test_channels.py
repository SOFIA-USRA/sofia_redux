# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.channels import Channels
from sofia_redux.scan.channels.division.division import ChannelDivision
from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.custom.example.channels.channel_data.channel_data \
    import ExampleChannelData
from sofia_redux.scan.custom.example.scan.scan import ExampleScan
from sofia_redux.toolkit.utilities.fits import set_log_level


@pytest.fixture
def empty_channels():
    info = ExampleInfo()
    channels = Channels(info=info)
    return channels


@pytest.fixture
def populated_channels(populated_scan):
    return populated_scan.channels


class TestChannels(object):

    def test_init(self):
        # bare init is allowed
        channels = Channels()
        assert channels.info is None
        assert channels.parent is None
        assert channels.data is None

        # can also specify info and parent
        info = ExampleInfo()
        parent = 'test'
        channels = Channels(parent=parent, info=info)
        assert channels.info is info
        assert channels.parent == 'test'
        assert channels.info.name == 'example'
        assert isinstance(channels.data, ExampleChannelData)

        # and name override if info supplied
        channels = Channels(info=info, name='test')
        assert channels.info is info
        assert channels.info.name == 'test'

    def test_empty_copy(self, empty_channels):
        new = empty_channels.copy()
        assert new.data is not empty_channels.data
        assert new.data.size == empty_channels.data.size
        assert new.groups is None
        assert new.divisions is None
        assert new.modalities is None
        assert not new.is_initialized

    def test_populated_copy(self, mocker, populated_channels):
        # set some extra attributes to copy
        populated_channels.parent = ExampleInfo()
        populated_channels.overlap_point_size = 0.1
        populated_channels.test = ExampleInfo()
        mocker.patch.object(populated_channels, 'calculate_overlaps')

        mocker.patch('sofia_redux.scan.channels.channels.'
                     'Channels.reference_attributes',
                     return_value={'parent', 'test'},
                     new_callable=mocker.PropertyMock)
        assert 'test' in populated_channels.reference_attributes

        new = populated_channels.copy()

        # data copy
        assert new.data is not populated_channels.data
        assert new.data.size == populated_channels.data.size

        # references
        assert new.parent is populated_channels.parent
        assert new.test is populated_channels.test

        # point size reset, overlaps recalculated
        assert np.isnan(new.overlap_point_size)
        new.calculate_overlaps.assert_called_once()

        # initialized
        assert new.is_initialized

    def test_fifi_copy(self, fifi_simulated_channels):
        chan_1 = fifi_simulated_channels
        chan_2 = fifi_simulated_channels.copy()
        assert chan_1.overlap_point_size == chan_2.overlap_point_size

    def test_blank_properties(self):
        channels = Channels()
        assert channels.flagspace is None
        assert channels.configuration is None
        assert channels.size == 0
        assert channels.sourceless_flags is None
        assert channels.non_detector_flags is None

        assert channels.n_store_channels == 0
        channels.n_store_channels = 1
        assert channels.n_store_channels == 0
        assert channels.n_mapping_channels == 0
        channels.n_mapping_channels = 1
        assert channels.n_mapping_channels == 0

        assert channels.parent is None
        channels.parent = 'test'
        assert channels.parent is None

        assert channels.has_option('test') is False
        with pytest.raises(IndexError) as err:
            channels[0]
        assert 'No data' in str(err)

        assert channels.get_name() is None
        channels.set_name('test')
        assert channels.get_name() is None

        assert channels.get_pixel_count() == 0
        assert channels.get_pixels() is None

    def test_populated_properties(self, populated_channels):
        channels = populated_channels
        assert channels.flagspace is channels.data.flagspace
        assert channels.configuration is channels.info.configuration
        assert channels.size == channels.data.size
        assert channels.sourceless_flags == \
            channels.data.flagspace.sourceless_flags()
        assert channels.non_detector_flags == \
            channels.data.flagspace.non_detector_flags()

        channels.n_store_channels = 11
        assert channels.n_store_channels == 11
        assert channels.info.instrument.n_store_channels == 11

        channels.n_mapping_channels = 11
        assert channels.n_mapping_channels == 11
        assert channels.info.instrument.n_mapping_channels == 11

        channels.configuration.set_option('test', True)
        assert channels.has_option('test')

        assert channels.size == 121
        get_item = channels[0]
        assert isinstance(get_item, ExampleChannelData)
        assert get_item.size == 1

    def test_apply_configuration(self, mocker, capsys, populated_channels):
        # check that appropriate functions are called
        m1 = mocker.patch.object(populated_channels, 'scramble')
        m2 = mocker.patch.object(populated_channels, 'flatten_weights')
        m3 = mocker.patch.object(populated_channels.data,
                                 'set_uniform_gains')
        m4 = mocker.patch.object(populated_channels, 'set_fixed_source_gains')
        m5 = mocker.patch.object(populated_channels, 'jackknife')
        m6 = mocker.patch.object(populated_channels, 'normalize_array_gains')
        m7 = mocker.patch.object(populated_channels.data,
                                 'validate_weights')

        populated_channels.configuration.set_option('scramble', True)
        populated_channels.configuration.set_option('flatweights', True)
        populated_channels.configuration.set_option('uniform', True)
        populated_channels.configuration.set_option('source.fixedgains', True)
        populated_channels.configuration.set_option('jackknife.channels', True)

        populated_channels.apply_configuration()
        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_called_once()
        m4.assert_called_once()
        m5.assert_called_once()
        m6.assert_called_once()
        m7.assert_called_once()

        # spikes and dof are reset
        assert np.allclose(populated_channels.data.spikes, 0)
        assert np.allclose(populated_channels.data.dof, 1)

        # optionally add noise to gains
        gains = populated_channels.data.gain.copy()
        populated_channels.configuration.set_option('gainnoise', 0.5)
        populated_channels.apply_configuration()
        assert np.std(populated_channels.data.gain) > np.std(gains)

        # optionally incorporate coupling into gains and reset coupling
        gains = populated_channels.data.gain.copy()
        populated_channels.data.coupling.fill(2.0)
        populated_channels.configuration.set_option('gainnoise', np.nan)
        populated_channels.configuration.set_option('sourcegains', True)
        populated_channels.apply_configuration()
        assert np.allclose(populated_channels.data.gain, gains * 2)
        assert np.allclose(populated_channels.data.coupling, 1.0)

    def test_jackknife(self, mocker, populated_channels):
        nchannel = populated_channels.size
        assert np.allclose(populated_channels.data.coupling, 1)

        # mock random to always return < 0.5
        mocker.patch('sofia_redux.scan.channels.channels.'
                     'np.random.random', return_value=np.full(nchannel, 0.1))
        populated_channels.jackknife()
        assert np.allclose(populated_channels.data.coupling, -1)

    def test_flag_channels(self, populated_channels):
        # no channels flagged by default
        populated_channels.flag_channels()
        assert not np.any(populated_channels.data.weight == 0)
        assert not np.any(populated_channels.data.flag
                          == populated_channels.flagspace.flags.BLIND.value)
        assert not np.any(populated_channels.data.flag
                          == populated_channels.flagspace.flags.DEAD.value)

        # flag some in configuration
        populated_channels.configuration.set_option('blind', [0, 1, 2])
        populated_channels.configuration.set_option('flag', [3, 4, 5])
        populated_channels.flag_channels()
        assert np.sum(populated_channels.data.weight == 0) == 3
        assert np.all(populated_channels.data.flag[:3]
                      == populated_channels.flagspace.flags.BLIND.value)
        assert np.all(populated_channels.data.flag[3:6]
                      == populated_channels.flagspace.flags.DEAD.value)

    def test_scramble(self, capsys, mocker, populated_channels):
        px = populated_channels.data.position.x.copy()
        py = populated_channels.data.position.y.copy()

        populated_channels.scramble()
        assert 'Scrambling pixel position data' in capsys.readouterr().err
        assert not np.allclose(populated_channels.data.position.x, px)
        assert not np.allclose(populated_channels.data.position.y, py)

    def test_get_channel_data_instance(self, populated_channels):
        channels = Channels()
        with pytest.raises(ValueError) as err:
            channels.get_channel_data_instance()
        assert "Info must be set" in str(err)

        channel_data = populated_channels.get_channel_data_instance()
        assert isinstance(channel_data, ExampleChannelData)

    def test_get_scan_instance(self, populated_channels):
        channels = Channels()
        with pytest.raises(ValueError) as err:
            channels.get_scan_instance()
        assert "Info must be set" in str(err)

        scan = populated_channels.get_scan_instance()
        assert isinstance(scan, ExampleScan)

    def test_get_pixels(self, populated_channels):
        nchannel = populated_channels.size
        channels = Channels()
        channels.data = populated_channels.data
        assert channels.data is not None
        assert channels.size == nchannel

        assert channels.get_pixel_count() == nchannel
        pixels = channels.get_pixels()
        assert pixels.shape == (nchannel,)

    def test_get_perimeter_pixels(self, populated_channels):
        perimeter = populated_channels.get_perimeter_pixels()
        assert isinstance(perimeter, ExampleChannelGroup)
        assert perimeter.size == 63
        assert perimeter.position.x.min() == -10 * units.arcsec
        assert perimeter.position.y.min() == -10 * units.arcsec
        assert perimeter.position.x.max() == 10 * units.arcsec
        assert perimeter.position.y.max() == 10 * units.arcsec

        populated_channels.configuration.set_option('perimeter', 4)
        perimeter = populated_channels.get_perimeter_pixels()
        assert isinstance(perimeter, ExampleChannelGroup)
        assert perimeter.size == 4
        assert np.allclose(perimeter.position.x,
                           np.array([10, -10, 10, -10]) * units.arcsec)
        assert np.allclose(perimeter.position.y,
                           np.array([10, 10, -10, -10]) * units.arcsec)

        populated_channels.configuration.set_option('perimeter', 0)
        perimeter = populated_channels.get_perimeter_pixels()
        assert isinstance(perimeter, ExampleChannelGroup)
        assert perimeter.size == populated_channels.size
        assert np.allclose(perimeter.position.x,
                           populated_channels.data.position.x)
        assert np.allclose(perimeter.position.y,
                           populated_channels.data.position.y)

    def test_find_fixed_indices(self, mocker, populated_channels):
        assert np.allclose(populated_channels.find_fixed_indices([1, 2, 3]),
                           [1, 2, 3])
        populated_channels.data.fixed_index += 1
        assert np.allclose(populated_channels.find_fixed_indices([1, 2, 3]),
                           [0, 1, 2])

    def test_get_division(self, populated_channels):
        # non-existent field in channel data
        with pytest.raises(ValueError) as err:
            populated_channels.get_division('test', 'test')
        assert "does not contain 'test'" in str(err)

        # good field, divided by unique values
        populated_channels.data.gain[:10] = 1
        populated_channels.data.gain[10:] = 2
        division = populated_channels.get_division('test', 'gain')
        assert len(division.groups) == 2
        assert division.groups[0].name == 'gain-1.0'
        assert division.groups[1].name == 'gain-2.0'

    def test_add_group(self, populated_channels):
        # reset groups
        populated_channels.groups = {}

        # add no group
        populated_channels.add_group(None)
        assert len(populated_channels.groups) == 0

        # add bad group
        with pytest.raises(ValueError) as err:
            populated_channels.add_group('test')
        assert 'can only contain' in str(err)
        assert len(populated_channels.groups) == 0

        # add good group
        group = ExampleChannelGroup(populated_channels.data, name='test1')
        populated_channels.add_group(group)
        assert len(populated_channels.groups) == 1
        assert populated_channels.groups['test1'] is group

        # specify name
        group = ExampleChannelGroup(populated_channels.data)
        populated_channels.add_group(group, name='test2')
        assert len(populated_channels.groups) == 2
        assert populated_channels.groups['test2'] is group

    def test_add_division(self, populated_channels):
        populated_channels.divisions = None
        assert populated_channels.list_divisions() == []

        # reset groups and divisions
        populated_channels.groups = {}
        populated_channels.divisions = {}
        assert populated_channels.list_divisions() == []

        # add no division
        populated_channels.add_division(None)
        assert len(populated_channels.groups) == 0

        # add bad division
        with pytest.raises(ValueError) as err:
            populated_channels.add_division('test')
        assert 'can only contain' in str(err)
        assert len(populated_channels.groups) == 0

        # add good division
        group1 = ExampleChannelGroup(populated_channels.data)
        group2 = ExampleChannelGroup(populated_channels.data)
        division = ChannelDivision('test', [group1, group2])
        populated_channels.add_division(division)

        # added to divisions and groups
        assert len(populated_channels.divisions) == 1
        assert populated_channels.divisions['test'] is division

        assert len(populated_channels.groups) == 2
        assert populated_channels.groups['test-1'] is group1
        assert populated_channels.groups['test-2'] is group2

        assert populated_channels.list_divisions() == ['test']

    def test_add_modality(self, populated_channels):
        populated_channels.modalities = None
        assert populated_channels.list_modalities() == []

        # reset modalities
        populated_channels.modalities = {}
        assert populated_channels.list_modalities() == []

        # add no modality
        populated_channels.add_modality(None)
        assert len(populated_channels.modalities) == 0

        # add bad modality
        with pytest.raises(ValueError) as err:
            populated_channels.add_modality('test')
        assert 'can only contain' in str(err)
        assert len(populated_channels.modalities) == 0

        # add good modality
        modality = Modality(name='test')
        populated_channels.add_modality(modality)
        assert len(populated_channels.modalities) == 1
        assert populated_channels.modalities['test'] is modality

        assert populated_channels.list_modalities() == ['test']

    def test_flat_weights_gains(self, mocker, populated_channels):
        # just check channel data functions are called
        m1 = mocker.patch.object(populated_channels.data, 'flatten_weights')
        m2 = mocker.patch.object(populated_channels.data, 'set_uniform_gains')
        populated_channels.flatten_weights()
        m1.assert_called_once()
        populated_channels.uniform_gains()
        m2.assert_called_once()

    def test_set_fixed_source_gains(self, populated_channels):
        populated_channels.data.gain[:] = 2.0
        populated_channels.data.coupling[:] = 3.0
        populated_channels.set_fixed_source_gains()
        assert np.allclose(populated_channels.data.gain, 2)
        assert np.allclose(populated_channels.data.coupling, 6)
        assert populated_channels.fixed_source_gains is True

        # won't redo the operation if already done
        populated_channels.set_fixed_source_gains()
        assert np.allclose(populated_channels.data.gain, 2)
        assert np.allclose(populated_channels.data.coupling, 6)

    def test_get_channel_flag_key(self, empty_channels):
        key = empty_channels.get_channel_flag_key()
        assert key == ["'X' - 1 - Dead",
                       "'B' - 2 - Blind",
                       "'d' - 4 - Discarded",
                       "'g' - 8 - Gain",
                       "'n' - 16 - Noisy",
                       "'f' - 32 - Degrees-of-freedom",
                       "'s' - 64 - Spiky",
                       "'r' - 128 - Railing/Saturated",
                       "'F' - 256 - Insufficient phase degrees-of-freedom",
                       "'t' - 512 - Time weighting",
                       "'b' - 1024 - Bad TES bias gain",
                       "'m' - 2048 - Bad MUX gain",
                       "'R' - 4096 - Bad detector row gain",
                       "'T' - 8192 - Flicker noise"]

    def test_write_channel_data(self, tmpdir, populated_channels):
        nchannel = populated_channels.size
        filename = str(tmpdir.join('channel.txt'))
        populated_channels.write_channel_data(filename, header='test header')
        assert os.path.isfile(filename)

        with open(filename) as fh:
            lines = fh.readlines()
        assert len(lines) > nchannel

        # header present
        assert lines[2] == 'test header\n'

        # spot check contents
        lines = ''.join(lines)
        assert '0,0\t1.000\t3.162e+00\t-\t1.000' \
               '\t1.000\t1.000\t0\t0\t0' in lines
        assert '10,10\t1.000\t3.162e+00\t-\t1.000' \
               '\t1.000\t1.000\t120\t10\t10' in lines

    def test_print_correlated_modalities(self, capsys, populated_channels):
        populated_channels.is_initialized = False
        populated_channels.print_correlated_modalities()
        capt = capsys.readouterr()
        # * for configured modalities only
        assert 'Available pixel divisions for example:' in capt.out
        assert '(*) obs-channels' in capt.out
        assert '    sky' in capt.out

        # same for response modalities
        populated_channels.print_response_modalities()
        capt = capsys.readouterr()
        # * for configured modalities only
        assert 'Available pixel divisions for example:' in capt.out
        assert '(*) obs-channels' in capt.out
        assert '    sky' in capt.out

    def test_reindex(self, mocker, populated_channels):
        m1 = mocker.patch('sofia_redux.scan.channels.'
                          'channel_group.channel_group.ChannelGroup.reindex')
        populated_channels.reindex()

        # reindex called for all groups, divisions, modalities
        total = len(populated_channels.groups)
        for n, d in populated_channels.divisions.items():
            total += len(d.groups)
        for n, m in populated_channels.modalities.items():
            total += len(m.modes)
        assert m1.call_count == total

        # data is reset to channels.data
        for name, group in populated_channels.groups.items():
            assert group.data is populated_channels.data

    def test_slim(self, populated_channels):
        nchannel = populated_channels.size

        # nothing flagged
        assert populated_channels.slim() is False
        assert populated_channels.size == nchannel

        # flag some channels
        populated_channels.configuration.set_option('blind', [0, 1, 2])
        populated_channels.configuration.set_option('flag', [3, 4, 5])
        populated_channels.flag_channels()

        # discards the dead ones
        assert populated_channels.slim() is True
        assert populated_channels.size == nchannel - 3
        for name, group in populated_channels.groups.items():
            assert group.data.size == nchannel - 3

    def test_load_temporary_hardware_gains(self, populated_channels):
        nchannel = populated_channels.size
        populated_channels.data.hardware_gain = None
        populated_channels.data.temp = None

        populated_channels.load_temporary_hardware_gains()
        assert populated_channels.data.hardware_gain.size == nchannel
        assert populated_channels.data.temp.size == nchannel
        assert np.allclose(populated_channels.data.hardware_gain,
                           populated_channels.data.temp)
        assert (populated_channels.data.temp
                is not populated_channels.data.hardware_gain)

    def test_get_source_gains(self, populated_channels):
        populated_channels.data.gain[:] = 2.0
        populated_channels.data.coupling[:] = 3.0
        populated_channels.data.source_filtering[:] = 4.0

        populated_channels.configuration.set_option('source.fixedgains', True)
        gain = populated_channels.get_source_gains(filter_corrected=False)
        assert np.allclose(gain, 3)

        populated_channels.configuration.set_option('source.fixedgains', False)
        gain = populated_channels.get_source_gains(filter_corrected=False)
        assert np.allclose(gain, 6)

        populated_channels.configuration.set_option('source.fixedgains', False)
        gain = populated_channels.get_source_gains(filter_corrected=True)
        assert np.allclose(gain, 24)

    def test_get_fwhm(self, populated_channels):
        nchannel = populated_channels.size
        channels = Channels()
        channels.data = populated_channels.data
        channels.data.resolution = np.arange(nchannel, dtype=float)
        channels.data.resolution[:2] = np.nan
        channels.data.resolution[-2:] = np.nan

        assert channels.get_min_beam_fwhm() == 2
        assert channels.get_max_beam_fwhm() == nchannel - 3
        assert channels.get_average_beam_fwhm() == (nchannel - 1) / 2

    def test_get_average_filtering(self, populated_channels):
        nchannel = populated_channels.size

        # all gains 1.0, all channels unflagged
        gain = populated_channels.get_average_filtering()
        assert np.allclose(gain, 1)

        populated_channels.data.gain[:] = 2.0
        populated_channels.data.coupling[:] = 3.0
        populated_channels.data.source_filtering[:] = 4.0
        populated_channels.data.weight = np.arange(nchannel)
        gain = populated_channels.get_average_filtering()
        assert np.allclose(gain, 4)

    def test_flag_weights(self, mocker, capsys, populated_channels):
        nchannel = populated_channels.size

        # check that numba function is called, trigger some error conditions
        flag = np.arange(nchannel)
        m1 = mocker.patch('sofia_redux.scan.channels.channels.'
                          'cnf.flag_weights', return_value=(0, 0, flag))

        populated_channels.flag_weights()
        assert m1.call_count == 1
        assert np.allclose(populated_channels.data.flag, flag)
        assert 'No valid channels' in capsys.readouterr().err

        m2 = mocker.patch('sofia_redux.scan.channels.channels.'
                          'cnf.flag_weights', return_value=(nchannel, 0, flag))

        populated_channels.flag_weights()
        assert m2.call_count == 1
        assert 'All channels flagged' in capsys.readouterr().err

        populated_channels.n_mapping_channels = 0
        populated_channels.flag_weights()
        assert m2.call_count == 2
        assert 'No mapping channels' in capsys.readouterr().err

    def test_get_source_nefd(self, mocker, populated_channels):
        # just check that numba function is called
        mocker.patch('sofia_redux.scan.channels.channels.'
                     'cnf.get_source_nefd', return_value='test')

        assert populated_channels.get_source_nefd() == 'test'

    def test_get_stability(self, empty_channels):
        del empty_channels.configuration['stability']
        assert empty_channels.get_stability() == 10 * units.s
        empty_channels.configuration.set_option('stability', 4)
        assert empty_channels.get_stability() == 4 * units.s

    def test_get_one_over_f_stat(self, mocker, populated_channels):
        # just check that numba function is called
        mocker.patch('sofia_redux.scan.channels.channels.'
                     'cnf.get_one_over_f_stat', return_value='test')

        assert populated_channels.get_one_over_f_stat() == 'test'

    def test_get_fits_data(self, populated_channels):
        data = {}

        # todo: check on when/how n_store_channels should be set
        # it currently seems to be set only if size is specified on
        # initialization of the Channels instance
        populated_channels.n_store_channels = populated_channels.size

        populated_channels.get_fits_data(data)
        np.allclose(data['Channel_Gains'], populated_channels.data.gain)
        np.allclose(data['Channel_Weights'], populated_channels.data.weight)
        np.allclose(data['Channel_Flags'], populated_channels.data.flag)

    def test_edit_scan_header(self, mocker, empty_channels):
        # check that flagspace method is called if configured
        m1 = mocker.patch.object(empty_channels.flagspace, 'edit_header')

        empty_channels.configuration.set_option(
            'write.scandata.details', False)
        empty_channels.edit_scan_header({})
        assert m1.call_count == 0

        empty_channels.configuration.set_option(
            'write.scandata.details', True)
        empty_channels.edit_scan_header({})
        assert m1.call_count == 1

    def test_calculate_overlaps(self, mocker, populated_channels):
        m1 = mocker.patch.object(populated_channels.data, 'calculate_overlaps')

        res = populated_channels.info.resolution
        assert np.isnan(populated_channels.overlap_point_size)
        populated_channels.calculate_overlaps()
        assert populated_channels.overlap_point_size == res
        assert m1.call_count == 1

        # calling again does nothing
        populated_channels.calculate_overlaps()
        assert populated_channels.overlap_point_size == res
        assert m1.call_count == 1

        # set point size explicitly: redoes overlaps
        populated_channels.calculate_overlaps(point_size=1)
        assert populated_channels.overlap_point_size == 1 * units.arcsec
        assert m1.call_count == 2

        populated_channels.calculate_overlaps(point_size=2 * units.arcsec)
        assert populated_channels.overlap_point_size == 2 * units.arcsec
        assert m1.call_count == 3

    def test_get_table_entry(self, populated_channels):
        c = populated_channels
        assert c.get_table_entry('gain') == 1.0
        assert c.get_table_entry('sampling') == 0.1
        assert c.get_table_entry('rate') == 10.0
        assert c.get_table_entry('okchannels') == c.size
        assert c.get_table_entry('maxchannels') == 0
        assert c.get_table_entry('mount') == 'CASSEGRAIN'
        assert c.get_table_entry('resolution') == 10.0
        assert c.get_table_entry('sizeunit') == 'arcsec'
        assert c.get_table_entry('ptfilter') == 1.0
        assert np.isnan(c.get_table_entry('FWHM'))
        assert np.isnan(c.get_table_entry('minFWHM'))
        assert np.isnan(c.get_table_entry('maxFWHM'))
        assert np.isnan(c.get_table_entry('stat1f'))
        assert c.get_table_entry('test') is None

    def test_str(self, empty_channels):
        assert str(empty_channels) == 'Instrument example'

    def test_troubleshoot_few_pixels(self, empty_channels):
        # set some configured options to get all suggestions
        empty_channels.configuration.set_option('correlated.test1', True)
        empty_channels.configuration.set_option('correlated.test2', True)
        empty_channels.configuration.set_option('correlated.test2.nogains',
                                                True)
        empty_channels.configuration.set_option('gains', True)
        empty_channels.configuration.set_option('despike', True)
        empty_channels.configuration.set_option('weighting.noiserange',
                                                [0.1, 1])

        empty_channels.modalities = {'test1': Modality(),
                                     'test2': Modality(),
                                     'test3': Modality()}

        suggestion = empty_channels.troubleshoot_few_pixels()
        suggestion = '\n'.join(suggestion)

        assert 'Disable gain estimation' in suggestion
        assert 'correlated.test1.nogains' in suggestion
        assert 'correlated.test2.nogains' not in suggestion
        assert 'correlated.test3.nogains' not in suggestion
        assert 'forget=gains' in suggestion
        assert 'forget=despike' in suggestion
        assert 'weighting.noiserange' in suggestion

    def test_data_functions(self, mocker, populated_channels):
        channels = Channels()
        channels.data = populated_channels.data

        # flag fields
        m1 = mocker.patch.object(channels.data, 'flag_field')
        channels.flag_field('test', None)
        assert m1.call_count == 1
        assert m1.called_with('test', None)

        # not called for non-dict fields
        fields = ['test']
        channels.flag_fields(fields)
        assert m1.call_count == 1

        # called for each if fields is dict
        fields = {'test1': None, 'test2': None}
        channels.flag_fields(fields)
        assert m1.call_count == 3
        assert m1.called_with('test1', None)
        assert m1.called_with('test2', None)

        # flag channel list
        m2 = mocker.patch.object(channels.data, 'flag_channel_list')
        channels.flag_channel_list(None)
        assert m2.call_count == 1
        assert m2.called_with(None)

        # kill channels
        m3 = mocker.patch.object(channels.data, 'kill_channels')
        channels.kill_channels(None)
        assert m3.call_count == 1
        assert m3.called_with(flag=None)

        # set blind channels
        m4 = mocker.patch.object(channels.data, 'set_blind_channels')
        channels.set_blind_channels(None)
        assert m4.call_count == 1
        assert m4.called_with(None)

        # get gain magnitude
        m5 = mocker.patch.object(channels.data,
                                 'get_typical_gain_magnitude')
        channels.get_typical_gain_magnitude(None)
        assert m5.call_count == 1
        assert m5.called_with(None)

        # get mapping pixels
        m6 = mocker.patch.object(channels.data,
                                 'get_mapping_pixels')
        channels.get_mapping_pixels(None)
        assert m6.call_count == 1
        assert m6.called_with(discard_flag=None)

        # remove dependents
        m7 = mocker.patch.object(channels.data,
                                 'remove_dependents')
        channels.remove_dependents(None)
        assert m7.call_count == 1
        assert m7.called_with(None)

        # add dependents
        m8 = mocker.patch.object(channels.data,
                                 'add_dependents')
        channels.add_dependents(None)
        assert m8.call_count == 1
        assert m8.called_with(None)

        # get filtering
        m9 = mocker.patch.object(channels.data,
                                 'get_filtering')
        channels.get_filtering(None)
        assert m9.call_count == 1
        assert m9.called_with(None)

    def test_load_channel_data(self, mocker, tmpdir, capsys,
                               populated_channels):
        channels = Channels()
        channels.info = populated_channels.info
        channels.data = populated_channels.data

        # pixel and wiring data not configured
        channels.configuration.set_option('pixeldata', ' ')
        channels.configuration.set_option('wiring', ' ')
        m1 = mocker.patch.object(channels.data, 'read_pixel_data')
        channels.load_channel_data()
        assert m1.call_count == 0
        capt = capsys.readouterr()
        assert 'Cannot read pixel data' in capt.err
        assert 'Cannot read wiring data' in capt.err

        # make pix and wiring files to read
        pix = tmpdir.join('pixeldata.dat')
        pix.write('test pixel data')
        wiring = tmpdir.join('wiring.dat')
        wiring.write('test wiring data')

        channels.configuration.set_option('pixeldata', str(pix))
        channels.configuration.set_option('wiring', str(wiring))
        channels.load_channel_data()

        # pixel data is loaded; wiring data is no-op for default channels
        assert m1.call_count == 1
        capt = capsys.readouterr()
        assert f'Loading pixel data from {str(pix)}' in capt.out
        assert 'wiring' not in capt.out

    def test_set_weights(self, populated_channels):
        weight = populated_channels.data.weight.copy()
        time = populated_channels.info.instrument.integration_time.value
        assert not populated_channels.standard_weights

        populated_channels.set_standard_weights()
        assert np.allclose(populated_channels.data.weight,
                           weight / np.sqrt(time))
        assert populated_channels.standard_weights

        # no op if repeated
        populated_channels.set_standard_weights()
        assert np.allclose(populated_channels.data.weight,
                           weight / np.sqrt(time))
        assert populated_channels.standard_weights

        # set_sample_weights undoes the operation
        populated_channels.set_sample_weights()
        assert np.allclose(populated_channels.data.weight, weight)
        assert not populated_channels.standard_weights

        # no op if repeated
        populated_channels.set_sample_weights()
        assert np.allclose(populated_channels.data.weight, weight)
        assert not populated_channels.standard_weights

    def test_get_groups(self, populated_channels):
        # observing channels
        group = populated_channels.get_observing_channels()
        assert isinstance(group, ExampleChannelGroup)
        assert group.name == 'obs-channels-1'

        # live channels
        group = populated_channels.get_live_channels()
        assert isinstance(group, ExampleChannelGroup)
        assert group.name == 'live-1'

        # detector channels
        group = populated_channels.get_detector_channels()
        assert isinstance(group, ExampleChannelGroup)
        assert group.name == 'detectors-1'

        # sensitive channels
        group = populated_channels.get_sensitive_channels()
        assert isinstance(group, ExampleChannelGroup)
        assert group.name == 'sensitive-1'

        # blind channels
        group = populated_channels.get_blind_channels()
        assert isinstance(group, ExampleChannelGroup)
        assert group.name == 'blinds-1'

        # should all return None if no groups
        populated_channels.groups = None
        assert populated_channels.get_observing_channels() is None
        assert populated_channels.get_live_channels() is None
        assert populated_channels.get_detector_channels() is None
        assert populated_channels.get_sensitive_channels() is None
        assert populated_channels.get_blind_channels() is None

    def test_census(self, capsys, populated_channels):
        nchannel = populated_channels.size
        populated_channels.census()
        assert populated_channels.n_mapping_channels == nchannel

        populated_channels.modalities = None
        with set_log_level('DEBUG'):
            populated_channels.census(report=True)
        assert populated_channels.n_mapping_channels == 0
        capt = capsys.readouterr()
        assert 'Mapping channels: 0' in capt.out
        assert f'Mapping pixels: {nchannel}' in capt.out

        with set_log_level('DEBUG'):
            populated_channels.census(report=False)
        capt = capsys.readouterr()
        assert 'Mapping channels' not in capt.out
        assert 'Mapping pixels' not in capt.out

    def test_create_channel_group(self, populated_channels):
        nchannel = populated_channels.size

        # group from all pixels
        group = populated_channels.create_channel_group()
        assert isinstance(group, ExampleChannelGroup)
        assert group.size == nchannel

        # group from specified pixels
        group = populated_channels.create_channel_group(indices=np.arange(4))
        assert isinstance(group, ExampleChannelGroup)
        assert group.size == 4

        # group from flagged data
        populated_channels.configuration.set_option('flag', [3, 4, 5])
        dead = populated_channels.flagspace.flags.DEAD
        populated_channels.flag_channels()

        group = populated_channels.create_channel_group(keep_flag=dead)
        assert isinstance(group, ExampleChannelGroup)
        assert group.size == 3
        group = populated_channels.create_channel_group(discard_flag=dead)
        assert isinstance(group, ExampleChannelGroup)
        assert group.size == nchannel - 3
        group = populated_channels.create_channel_group(match_flag=dead)
        assert isinstance(group, ExampleChannelGroup)
        assert group.size == 3

    def test_init_groups(self, populated_channels):
        populated_channels.init_groups()
        assert len(populated_channels.groups) == 6

        # add some more from configuration
        populated_channels.configuration.set_option('group.test1', '0,1,2')
        populated_channels.configuration.set_option('group.test2', '3-6')
        populated_channels.init_groups()
        assert len(populated_channels.groups) == 8
        assert populated_channels.groups['test1'].size == 3
        assert populated_channels.groups['test2'].size == 4

        populated_channels.configuration.set_option('group.test3', None)
        with pytest.raises(ValueError) as err:
            populated_channels.init_groups()
        assert 'Could not parse group' in str(err)

    def test_init_divisions(self, capsys, populated_channels):
        populated_channels.init_divisions()
        assert len(populated_channels.divisions) == 8

        # add some more from configuration
        populated_channels.configuration.set_option('division.test1', 'live')
        populated_channels.configuration.set_option('division.test2',
                                                    'detectors')
        populated_channels.configuration.set_option('division.test3', 'bad')
        populated_channels.init_divisions()
        assert len(populated_channels.divisions) == 11
        assert len(populated_channels.divisions['test1'].groups) == 1
        assert len(populated_channels.divisions['test2'].groups) == 1
        assert len(populated_channels.divisions['test3'].groups) == 0
        assert 'Channel group bad is undefined ' \
               'for division test3' in capsys.readouterr().err

    def test_init_modalities(self, capsys, populated_channels):
        populated_channels.init_modalities()
        assert len(populated_channels.modalities) == 28

        # trigger error in one
        del populated_channels.divisions['detectors']
        populated_channels.init_modalities()
        assert len(populated_channels.modalities) == 27
        assert "Could not create modality from " \
               "detectors" in capsys.readouterr().err

        # add a blind modality: fails by default
        populated_channels.configuration.set_option('blind')
        populated_channels.init_modalities()
        assert len(populated_channels.modalities) == 27
        assert 'blinds-1' not in populated_channels.modalities
        assert "has no 'temperature_gain'" in capsys.readouterr().err

        # add some more from configuration
        populated_channels.configuration.set_option('division.test1', 'live')
        populated_channels.configuration.set_option('division.test2',
                                                    'detectors')
        populated_channels.configuration.set_option('division.test3', 'bad')
        # does not succeed unless divisions already added
        populated_channels.init_modalities()
        assert len(populated_channels.modalities) == 27
        assert 'Configuration division test1 ' \
               'does not exist' in capsys.readouterr().err

        populated_channels.init_divisions()
        populated_channels.init_modalities()
        assert len(populated_channels.modalities) == 30
        assert 'Configuration division test3 does not ' \
               'contain any channel groups' in capsys.readouterr().err

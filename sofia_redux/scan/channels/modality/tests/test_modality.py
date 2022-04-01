# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.division.division import ChannelDivision
from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.channels.mode.field_response import FieldResponse
from sofia_redux.scan.channels.mode.mode import Mode
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.flags.instrument_flags import InstrumentFlags


class TestModality(object):
    def setup_modality(self, integ):
        g1 = ExampleChannelGroup(integ.channels.data)
        division = ChannelDivision('test_division', groups=[g1])
        modality = Modality(name='test', mode_class=FieldResponse,
                            gain_provider='gain', channel_division=division)
        for mode in modality.modes:
            mode.field = 'transmission'
            mode.gain_flag = mode.flagspace.flags(1)
            signal = mode.get_signal(integ)
            integ.add_signal(signal)
        return modality

    def test_init(self, example_modality):
        # bare init allowed
        modality = Modality()
        assert modality.modes is None
        assert modality.name is None
        assert modality.size == 0
        assert modality.flagspace is None
        assert len(modality.fields) == 0
        assert modality.mode_class is Mode

        # can also specify types, names
        assert example_modality.name == 'test'
        assert len(example_modality.modes) == 2
        assert example_modality.size == 2
        assert example_modality.flagspace is not None
        assert len(example_modality.fields) == 37
        assert example_modality.mode_class is Mode

    def test_flagspace(self):
        modality = Modality()
        modality.modes = None
        assert modality.flagspace is None
        modality.modes = []
        assert modality.flagspace is None
        modality.modes = ['bad']
        assert modality.flagspace is None

    def test_get_mode_name_index(self, example_modality):
        m = example_modality
        assert m.get_mode_name_index('test_g1') is None
        assert m.get_mode_name_index('test_division-1') == 0
        assert m.get_mode_name_index('test:test_division-1') == 0
        assert m.get_mode_name_index('test:test_division-2') == 1

    def test_validate_mode_index(self, example_modality):
        m = Modality()
        with pytest.raises(KeyError) as err:
            m.validate_mode_index(0)
        assert "No modes" in str(err)

        m = example_modality
        assert m.validate_mode_index(0) == 0
        assert m.validate_mode_index(-1) == 1
        assert m.validate_mode_index('test_division-1') == 0

        with pytest.raises(KeyError) as err:
            m.validate_mode_index('bad')
        assert 'does not exist' in str(err)

        with pytest.raises(ValueError) as err:
            m.validate_mode_index([1])
        assert 'Invalid index type' in str(err)

        with pytest.raises(IndexError) as err:
            m.validate_mode_index(-3)
        assert 'Cannot use index' in str(err)

        with pytest.raises(IndexError) as err:
            m.validate_mode_index(3)
        assert 'out of range' in str(err)

    def test_string(self, example_modality):
        modality = Modality()
        assert str(modality) == 'Modality (name=None id=None): 0 mode(s)'
        assert modality.to_string() == "Modality 'None':"

        # short description
        expected = 'Modality (name=test id=None): 2 mode(s)\n' \
                   'Mode (test:test_division-1): 121 channels\n' \
                   'Mode (test:test_division-2): 121 channels'
        assert str(example_modality) == expected

        # long description: includes channel list for all modes
        description = example_modality.to_string()
        assert description.startswith("Modality 'None':")
        assert description.endswith('10,8 10,9 10,10')

    def test_get_set(self, example_modality):
        m1 = example_modality[0]
        assert isinstance(m1, Mode)
        assert m1.name == 'test:test_division-1'
        example_modality[1] = m1
        assert example_modality[1].name == 'test:test_division-1'

        with pytest.raises(ValueError) as err:
            example_modality[1] = 'bad'
        assert 'must be of <class' in str(err)
        assert isinstance(example_modality[1], Mode)

    def test_set_mode_class(self):
        modality = Modality()
        modality.set_mode_class(None)
        assert modality.mode_class is Mode
        modality.set_mode_class(Mode)
        assert modality.mode_class is Mode
        with pytest.raises(ValueError) as err:
            modality.set_mode_class(str)
        assert 'Mode class must be' in str(err)

    def test_set_default_names(self, populated_data):
        modality = Modality()
        modality.set_default_names()
        assert modality.modes is None

        mode = Mode(
            channel_group=ExampleChannelGroup(populated_data, name='test'))
        modality.modes = [mode]
        modality.set_default_names()
        assert modality.modes[0].name == 'None:test'

    @pytest.mark.parametrize('as_dict', [True, False])
    def test_set_options(self, example_modality, as_dict):
        # no op if None passed
        example_modality.set_options(None)

        # error if bad type
        with pytest.raises(ValueError) as err:
            example_modality.set_options('bad')
        assert 'Configuration must be <class' in str(err)

        # set options via dict or config
        options = {'resolution': 10,
                   'trigger': 'test_trigger',
                   'nogains': True,
                   'phasegains': True,
                   'gainrange': '0:1',
                   'signed': as_dict,
                   'nofield': True}
        if as_dict:
            flag = InstrumentFlags.flags.GAINS_SIGNED
            example_modality.set_options(options)
        else:
            flag = InstrumentFlags.flags.GAINS_BIDIRECTIONAL

            config = Configuration()
            for key, value in options.items():
                config.set_option(f'test.{key}', value)
            example_modality.set_options(config)

        assert example_modality.resolution == 10 * units.s
        assert example_modality.trigger == 'test_trigger'
        assert example_modality.solve_gains is False

        for i in range(example_modality.size):
            assert example_modality.modes[i].gain_range.min == 0
            assert example_modality.modes[i].gain_range.max == 1
            assert example_modality.modes[i].gain_provider is None
            assert example_modality.modes[i].gain_type == flag
            assert example_modality.modes[i].phase_gains is True

    def test_no_op_config(self):
        m = Modality()

        # no op configurations for no modes
        m.set_gain_range('0:1')
        m.set_gain_direction('GAINS_SIGNED')
        m.set_gain_flag('bad')
        m.set_phase_gains(False)
        m.set_gain_provider('test')

        assert m.modes is None

    def test_update_all_gains(self, capsys, example_modality,
                              populated_integration):
        integ = populated_integration

        # no op if not solve_gains or if no modes
        m = Modality()
        m.solve_gains = False
        assert not m.update_all_gains(integ)
        m.solve_gains = True
        assert not m.update_all_gains(integ)

        m = example_modality
        m.solve_gains = False
        assert not m.update_all_gains(integ)
        assert capsys.readouterr().err == ''

        # also no op for fixed gains
        m.solve_gains = True
        for mode in m.modes:
            mode.fixed_gains = True
        assert not m.update_all_gains(integ)
        assert capsys.readouterr().err == ''

        # even still: no op for current modes because the signal is not
        # set in the integration
        for mode in m.modes:
            mode.fixed_gains = False
            mode.gain_flag = mode.flagspace.flags(1)
        assert not m.update_all_gains(integ)
        assert 'Could not update gains' in capsys.readouterr().err

        # set up a modality with signal and flagging
        modality = self.setup_modality(integ)
        assert modality.update_all_gains(integ)
        assert capsys.readouterr().err == ''

    def test_average_gains(self, mocker, populated_integration):
        integ = populated_integration
        modality = self.setup_modality(integ)

        nchannel = modality.modes[0].size
        gains = np.arange(nchannel, dtype=float)
        weights = np.arange(nchannel, dtype=float)

        mocker.patch.object(modality.modes[0], 'derive_gains',
                            return_value=(np.full(nchannel, 1.0),
                                          np.full(nchannel, 1.0)))
        modality.average_gains(integ, gains, weights)
        expected = ((np.arange(nchannel, dtype=float) ** 2 + 1)
                    / (np.arange(nchannel, dtype=float) + 1))

        assert np.allclose(gains, expected)
        assert np.allclose(weights, np.arange(nchannel, dtype=float) + 1)

        # no op if mode.fixed_gains
        gains = np.arange(nchannel, dtype=float)
        weights = np.arange(nchannel, dtype=float)
        modality.modes[0].fixed_gains = True
        modality.average_gains(integ, gains, weights)
        assert np.allclose(gains, np.arange(nchannel))
        assert np.allclose(weights, np.arange(nchannel))

    def test_apply_gains(self, mocker, populated_integration):
        integ = populated_integration
        modality = self.setup_modality(integ)

        nchannel = modality.modes[0].size
        gains = np.arange(nchannel, dtype=float)
        weights = np.arange(nchannel, dtype=float)

        assert modality.apply_gains(integ, gains, weights)

        # fixed gains: no op
        modality.modes[0].fixed_gains = True
        assert not modality.apply_gains(integ, gains, weights)

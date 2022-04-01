# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.channels.division.division import ChannelDivision
from sofia_redux.scan.channels.modality.correlated_modality \
    import CorrelatedModality
from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


@pytest.fixture
def example_correlated_modality(populated_data):
    g1 = ExampleChannelGroup(populated_data, name='test_g1')
    g2 = ExampleChannelGroup(populated_data, name='test_g2')
    division = ChannelDivision('test_division', groups=[g1, g2])
    modality = CorrelatedModality(name='test', mode_class=CorrelatedMode,
                                  channel_division=division,
                                  gain_provider='gain')
    return modality


class TestCorrelatedModality(object):
    def test_init(self, example_correlated_modality):
        modality = CorrelatedModality()
        assert modality.modes is None
        assert modality.solve_signal is True

        assert isinstance(example_correlated_modality, CorrelatedModality)
        assert len(example_correlated_modality.modes) == 2
        assert example_correlated_modality.solve_signal is True

    @pytest.mark.parametrize('as_dict', [True, False])
    def test_set_options(self, example_correlated_modality, as_dict):
        modality = example_correlated_modality

        # no op if None passed
        modality.set_options(None)

        # error if bad type
        with pytest.raises(ValueError) as err:
            modality.set_options('bad')
        assert 'Configuration must be <class' in str(err)

        # set options via dict or config
        options = {'resolution': 10,
                   'trigger': 'test_trigger',
                   'nogains': True,
                   'phasegains': True,
                   'gainrange': '0:1',
                   'signed': as_dict,
                   'nofield': True,
                   'nosignals': True}
        if as_dict:
            modality.set_options(options)
        else:

            config = Configuration()
            for key, value in options.items():
                config.set_option(f'correlated.test.{key}', value)
            modality.set_options(config)

        assert modality.resolution == 10 * units.s
        assert modality.trigger == 'test_trigger'
        assert modality.solve_gains is False
        assert modality.solve_signal is False

        for i in range(modality.size):
            assert modality.modes[i].gain_range.min == 0
            assert modality.modes[i].gain_range.max == 1
            assert modality.modes[i].gain_provider is None
            assert modality.modes[i].phase_gains is True

    def test_set_skip_flags(self, example_correlated_modality):
        example_correlated_modality.set_skip_flags(1)
        for m in example_correlated_modality.modes:
            assert m.skip_flags == m.flagspace.flags(1)
        example_correlated_modality.set_skip_flags('GAIN')
        for m in example_correlated_modality.modes:
            assert m.skip_flags == m.flagspace.flags.GAIN

    def test_update_signals(self, mocker, example_correlated_modality,
                            populated_integration):

        # no op if resolution is NaN
        example_correlated_modality.resolution = np.nan
        example_correlated_modality.update_signals('test')

        m1 = mocker.patch.object(CorrelatedMode, 'update_signals')
        example_correlated_modality.resolution = 10 * units.s
        example_correlated_modality.update_signals(populated_integration)
        assert m1.call_count == 2

        m2 = mocker.patch.object(CorrelatedMode, 'update_signals',
                                 side_effect=ValueError('bad'))
        with pytest.raises(ValueError):
            example_correlated_modality.update_signals(populated_integration)
        assert m2.call_count == 1

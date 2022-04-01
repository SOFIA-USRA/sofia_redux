# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.scan.channels.mode.position_response \
    import PositionResponse
from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.signal.signal import Signal


class TestPositionResponse(object):

    def test_init(self):
        response = PositionResponse()
        assert isinstance(response, Response)
        assert response.size == 0
        assert response.type is None

    def test_set_type(self):
        response = PositionResponse(position_type='Telescope')
        assert str(response.type) == 'MotionFlagTypes.TELESCOPE'

        with pytest.raises(ValueError) as err:
            response.set_type('x')
        assert 'Position type must be' in str(err)

    def test_get_signal_from_direction(self, mocker, populated_integration):
        integ = populated_integration
        group = ExampleChannelGroup(populated_integration.channels.data,
                                    name='test_group')
        mode = PositionResponse(channel_group=group, name='test',
                                position_type='Telescope')

        signal = mode.get_signal_from_direction(integ, 'x')
        assert isinstance(signal, Signal)
        assert signal.size == integ.size

        # verify it is a position signal retrieved
        m1 = mocker.patch.object(integ, 'get_position_signal')
        mode.get_signal_from_direction(integ, 'y')
        m1.assert_called_once()

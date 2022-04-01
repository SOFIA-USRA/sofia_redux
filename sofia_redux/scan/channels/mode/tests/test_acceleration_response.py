# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.mode.acceleration_response \
    import AccelerationResponse
from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.signal.signal import Signal


class TestAccelerationResponse(object):

    def test_init(self):
        response = AccelerationResponse()
        assert isinstance(response, Response)
        assert response.size == 0

    def test_get_signal_from_direction(self, mocker, populated_integration):
        integ = populated_integration
        group = ExampleChannelGroup(populated_integration.channels.data,
                                    name='test_group')
        mode = AccelerationResponse(channel_group=group, name='test')

        signal = mode.get_signal_from_direction(integ, 'x')
        assert isinstance(signal, Signal)
        assert signal.size == integ.size

        # verify it is an acceleration signal retrieved
        m1 = mocker.patch.object(integ, 'get_acceleration_signal')
        mode.get_signal_from_direction(integ, 'y')
        m1.assert_called_once()

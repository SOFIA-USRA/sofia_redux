# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.mode.motion_response \
    import MotionResponse
from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


class MotionResponseCheck(MotionResponse):
    """Implement abstract function for testing."""
    def get_signal_from_direction(self, integration, direction):
        return str(direction)


class TestMotionResponse(object):

    def test_init(self):
        response = MotionResponseCheck()
        assert isinstance(response, Response)
        assert response.size == 0

    def test_get_signal(self, mocker, populated_integration):
        integ = populated_integration
        group = ExampleChannelGroup(populated_integration.channels.data,
                                    name='test_group')
        mode = MotionResponseCheck(channel_group=group, name='test')

        mode.set_direction('x')
        signal = mode.get_signal(integ)
        assert signal == 'MotionFlags: MotionFlagTypes.X'

        mode.set_name('test-y')
        signal = mode.get_signal(integ)
        assert signal == 'MotionFlags: MotionFlagTypes.Y'

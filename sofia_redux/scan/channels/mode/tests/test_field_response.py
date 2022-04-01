# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.channels.mode.field_response \
    import FieldResponse
from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.signal.signal import Signal


class TestFieldResponse(object):

    def test_init(self):
        response = FieldResponse(field='gain', floating=True)
        assert isinstance(response, Response)
        assert response.size == 0
        assert response.field == 'gain'
        assert response.is_floating

    def test_get_signal(self, capsys, populated_integration):
        integ = populated_integration
        group = ExampleChannelGroup(populated_integration.channels.data,
                                    name='test_group')
        mode = FieldResponse(channel_group=group, name='test',
                             field='transmission', floating=True,
                             derivative_order=0)

        integ.frames.transmission = np.arange(integ.size, dtype=float)

        signal = mode.get_signal(integ)
        assert isinstance(signal, Signal)
        assert np.allclose(signal.value, np.arange(integ.size))

        mode.derivative_order = 1
        signal = mode.get_signal(integ)
        assert isinstance(signal, Signal)
        assert np.allclose(signal.value, 10)

        mode.field = 'test'
        signal = mode.get_signal(integ)
        assert isinstance(signal, Signal)
        assert np.allclose(signal.value, 0)
        assert 'No field named test' in capsys.readouterr().err

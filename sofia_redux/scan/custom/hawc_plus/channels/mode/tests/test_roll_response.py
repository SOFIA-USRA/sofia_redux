# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.custom.hawc_plus.channels.channel_group.channel_group \
    import HawcPlusChannelGroup
from sofia_redux.scan.custom.hawc_plus.channels.mode.roll_response import \
    RollResponse


def test_init(hawc_plus_channel_data):
    group = HawcPlusChannelGroup(hawc_plus_channel_data.copy(),
                                 indices=np.arange(10))
    response = RollResponse(channel_group=group)
    assert response.field == 'roll'
    assert response.is_floating
    assert response.derivative_order == 2

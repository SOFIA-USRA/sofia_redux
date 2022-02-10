# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.channel_flags import ChannelFlags


def test_channel_flags():
    flags = ChannelFlags
    for letter in ['X', 'B', 'd', 'g', 'n', 'f', 's', 'r', 'F', 't']:
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter

    assert flags.hardware_flags() & flags.flags.DEAD
    assert flags.hardware_flags() & flags.flags.BLIND
    assert not flags.software_flags() & flags.hardware_flags()
    assert flags.hardware_flags() & flags.critical_flags()
    assert flags.critical_flags() & flags.flags.GAIN
    assert flags.non_detector_flags() & flags.flags.DEAD
    assert flags.sourceless_flags() & flags.hardware_flags()
    assert flags.sourceless_flags() & flags.flags.DISCARD
    assert not flags.source_flags() & flags.sourceless_flags()

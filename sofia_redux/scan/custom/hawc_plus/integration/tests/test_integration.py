# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.custom.hawc_plus.integration.integration import \
    HawcPlusIntegration


arcsec = units.Unit('arcsec')
second = units.Unit('second')
degree = units.Unit('degree')
um = units.Unit('um')


def test_init():
    integration = HawcPlusIntegration()
    assert not integration.fix_jumps
    assert integration.min_jump_level_frames == 0
    assert integration.fix_subarray is None
    assert integration.drift_dependents is None


def test_scan_astrometry(small_integration):
    assert small_integration.scan_astrometry.epoch.equinox == 'J2000'


def test_apply_configuration(small_integration):
    small_integration.apply_configuration()  # Does nothing


def test_read(no_data_scan, full_hdu):
    hdus = [full_hdu]
    integration = no_data_scan[0]
    integration.frames.data = None
    integration.read(hdus)
    assert isinstance(integration.frames.data, np.ndarray)
    assert integration.frames.data.shape == (10, 3936)


def test_validate(small_integration):
    integration = small_integration.copy()
    integration.configuration.parse_key_value('chopper.shift', '1')
    integration.configuration.parse_key_value('jumpdata', 'foo')
    integration.configuration.parse_key_value('gyrocorrect', 'True')
    integration.configuration.parse_key_value('gyrocorrect.max', '10.0')
    integration.frames.data.fill(1)  # So zero channels are not flagged
    integration.validate()
    assert integration.is_valid


def test_get_table_entry(small_integration):
    integration = small_integration.copy()
    assert integration.get_table_entry('hwp') == 2.25 * degree
    assert integration.get_table_entry('pwv') == 6 * um
    assert integration.get_table_entry('foo') is None


def test_shift_chopper(small_integration):
    integration = small_integration.copy()
    n_frames = integration.size
    position = integration.frames.chopper_position
    position.x = np.arange(n_frames) * arcsec
    position.y = -np.arange(n_frames) * arcsec
    integration.shift_chopper(0)  # Does nothing
    assert integration.frames.valid.all()
    integration.shift_chopper(2)
    assert np.allclose(position.x[:4], [np.nan, np.nan, 0, 1] * arcsec,
                       equal_nan=True)
    assert np.allclose(position.y[:4], [np.nan, np.nan, 0, -1] * arcsec,
                       equal_nan=True)
    assert not np.any(position[-2:].is_nan())
    integration.shift_chopper(-2)
    assert np.allclose(position.x[:4], np.arange(4) * arcsec)
    assert np.allclose(position.y[:4], -np.arange(4) * arcsec)
    assert position[-2:].is_nan().all()


def test_flag_zeroed_channels(small_integration):
    integration = small_integration.copy()
    integration.frames.data.fill(1)
    integration.frames.channels.data.flag.fill(0)
    integration.frames.data[:, 1:3] = 0
    integration.flag_zeroed_channels()
    idx = np.nonzero(integration.channels.data.flag)[0]
    assert np.allclose(idx, [1, 2])
    flag = integration.channel_flagspace.convert_flag('DEAD|DISCARD').value
    assert np.allclose(integration.channels.data.flag[idx], flag)


def test_set_tau(small_integration):
    integration = small_integration.copy()
    integration.set_tau()
    assert np.isclose(integration.zenith_tau, 0.306, atol=1e-3)


def test_print_equivalent_taus(small_integration, capsys):
    integration = small_integration.copy()
    integration.print_equivalent_taus(0.1)
    output = capsys.readouterr()
    assert 'tau(53 um):0.1, tau(LOS):0.115, PWV:-40.5 um' in output.out


def test_check_jumps(small_integration, capsys):
    integration = small_integration.copy()
    frames = integration.frames
    channels = integration.channels.data
    channels.has_jumps.fill(False)
    jumps = frames.jump_counter.copy()
    integration.frames.jump_counter = None
    integration.check_jumps()
    assert 'no jump counter' in capsys.readouterr().err
    frames.valid.fill(False)
    frames.jump_counter = jumps
    frames.jump_counter.fill(0)
    integration.check_jumps()
    assert not channels.has_jumps.any()
    assert 'No valid frames' in capsys.readouterr().err

    frames.valid.fill(True)
    integration.check_jumps()
    assert not channels.has_jumps.any()

    frames.jump_counter[2:] = 2
    integration.check_jumps()
    assert channels.has_jumps.all()


def test_correct_jumps(small_integration):
    integration = small_integration.copy()
    integration.frames.data.fill(0.0)
    integration.frames.valid.fill(True)
    integration.frames.jump_counter[5:] = 2
    n = integration.channels.data.size
    integration.channels.data.jump = np.arange(n)
    integration.correct_jumps()
    assert np.allclose(integration.frames.data[:5], 0)
    assert np.allclose(integration.frames.data[5:], -np.arange(n)[None] * 2)


def test_remove_drifts(small_integration):
    integration = small_integration.copy()
    assert not integration.fix_jumps
    assert integration.fix_subarray is None
    assert integration.dependents is None
    c = integration.configuration
    c.purge('fixjumps')
    c.parse_key_value('fixjumps', 'True')
    c.parse_key_value('fixjumps.r0', 'True')
    c.parse_key_value('fixjumps.r1', 'False')
    c.parse_key_value('fixjumps.t0', 'False')
    c.parse_key_value('fixjumps.t1', 'True')
    integration.remove_drifts()
    assert integration.fix_jumps
    assert isinstance(integration.fix_subarray, np.ndarray)
    assert np.allclose(integration.fix_subarray, [True, False, False, True])
    assert integration.min_jump_level_frames == 10
    assert isinstance(integration.dependents, dict)
    assert 'drifts' in integration.dependents


def test_get_mean_hwp_angle(small_integration):
    integration = small_integration.copy()
    hwp = integration.frames.hwp_angle
    hwp[0] = 0 * degree
    hwp[-1] = 14 * degree
    assert np.isclose(integration.get_mean_hwp_angle(), 7 * degree)


def test_get_full_id(small_integration):
    assert small_integration.get_full_id() == '2016-12-14.UNKNOWN'


def test_check_consistency(small_integration):
    integration = small_integration.copy()
    channels = integration.channels.get_observing_channels().create_group(
        np.arange(5))
    frame_dependents = integration.frames.dependents.copy()
    integration.fix_subarray = np.full(4, True)
    consistent = integration.check_consistency(channels, frame_dependents)
    assert np.allclose(consistent, np.full(5, True))
    integration.frames.jump_counter[5:] = 2
    integration.fix_jumps = True
    channels.has_jumps = np.full(5, True)
    consistent = integration.check_consistency(channels, frame_dependents)
    assert not consistent.any()


def test_get_jump_blank_range(small_integration):
    integration = small_integration.copy()
    c = integration.configuration
    c.purge('fixjumps.blank')
    blank = integration.get_jump_blank_range()
    assert np.allclose(blank, [0, 0])

    c.parse_key_value('fixjumps.blank', '0.02')
    blank = integration.get_jump_blank_range()
    assert np.allclose(blank, [4, 4])

    c.parse_key_value('fixjumps.blank', '0.01,0.02')
    blank = integration.get_jump_blank_range()
    assert np.allclose(blank, [2, 4])

    c.parse_key_value('fixjumps.blank', '0.01,0.02,0.03')
    blank = integration.get_jump_blank_range()
    assert np.allclose(blank, [2, 2])


def test_level_jumps(small_integration):
    integration = small_integration.copy()
    channels = integration.channels.get_observing_channels().create_group(
        np.arange(5))
    frame_dependents = integration.frames.dependents.copy()
    frame_dependents.fill(0)
    integration.frames.jump_counter[5:] = (np.arange(5) + 1)[:, None]
    integration.fix_jumps = True
    integration.fix_subarray = np.full(4, True)
    channels.has_jumps = np.full(5, True)

    no_jumps = integration.level_jumps(channels, frame_dependents)
    assert not no_jumps.any()
    assert np.allclose(frame_dependents, (np.arange(10) < 5).astype(int))


def test_update_inconsistencies(small_integration):
    integration = small_integration.copy()
    channels = integration.channels.get_observing_channels().create_group(
        np.arange(5))
    frame_dependents = integration.frames.dependents.copy()
    frame_dependents.fill(0)
    integration.frames.jump_counter[5:] = 1
    integration.fix_jumps = True
    integration.fix_subarray = np.full(4, True)
    channels.has_jumps = np.full(5, True)
    channels.inconsistencies = np.zeros(5, dtype=int)
    integration.update_inconsistencies(channels, frame_dependents, 2)
    assert np.allclose(channels.inconsistencies, 1)


def test_get_first_frame(small_integration):
    assert small_integration.get_first_frame().fixed_index == 0


def test_get_last_frame(small_integration):
    assert small_integration.get_last_frame().fixed_index == 9

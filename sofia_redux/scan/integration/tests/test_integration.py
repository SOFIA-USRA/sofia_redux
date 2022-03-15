# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import astropy.table
from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode
from sofia_redux.scan.channels.mode.position_response import PositionResponse
from sofia_redux.scan.chopper.chopper import Chopper
from sofia_redux.scan.custom.example.integration.integration \
    import ExampleIntegration
from sofia_redux.scan.custom.example.frames.frames \
    import ExampleFrames
from sofia_redux.scan.integration.integration import Integration
from sofia_redux.scan.integration.dependents.dependents import Dependents
from sofia_redux.scan.signal.correlated_signal import CorrelatedSignal
from sofia_redux.scan.signal.signal import Signal
from sofia_redux.toolkit.utilities.fits import set_log_level


class TestIntegration(object):

    def test_init(self, initialized_scan, scan_file):
        # init okay without a scan
        integ = Integration()
        assert not integ.is_valid
        assert integ.scan is None

        # if scan is provided it must be populated
        with pytest.raises(ValueError) as err:
            Integration(initialized_scan)
        assert 'does not contain FITS HDUL' in str(err)

        # set a scan file to read: initialization succeeds, not valid yet
        s = initialized_scan
        s.hdul = fits.open(scan_file)
        integ = Integration(s)
        assert not integ.is_valid
        # scan is a reference, config is not
        assert integ.scan is s
        assert integ.configuration is not s.configuration

        # set a scan without config enabled: raises error
        s.info.configuration.enabled = False
        with pytest.raises(ValueError) as err:
            Integration(s)
        assert 'has not been configured' in str(err)
        s.hdul.close()

    @pytest.mark.parametrize('prop', ['flagspace', 'channel_flagspace',
                                      'info', 'configuration',
                                      'instrument_name', 'scan_astrometry'])
    def test_blank_properties_none(self, prop):
        integ = Integration()
        assert getattr(integ, prop) is None

        # some have no-op setters, some don't
        try:
            setattr(integ, prop, 'test')
        except AttributeError:
            pass
        assert getattr(integ, prop) is None

    @pytest.mark.parametrize('prop', ['size', 'n_channels'])
    def test_blank_properties_zero(self, prop):
        integ = Integration()
        assert getattr(integ, prop) == 0

    def test_has_option(self):
        integ = Integration()
        assert integ.has_option('test') is False

    def test_clone(self, populated_integration):
        integ = populated_integration
        clone = integ.clone()
        assert clone.scan is integ.scan
        assert clone.channels is not integ.channels
        assert integ.size == 1100
        assert clone.size == 0

    def test_n_channels(self, populated_integration):
        assert populated_integration.n_channels == 121

    def test_get_integration_class(self):
        result = Integration.get_integration_class('example')
        assert result is ExampleIntegration

    def test_scan_astrometry(self, populated_integration):
        integ = populated_integration
        assert integ.scan_astrometry is integ.scan.astrometry

    def test_getitem(self, populated_integration):
        fr = populated_integration[0]
        assert isinstance(fr, ExampleFrames)
        assert fr.size == 1
        fr = populated_integration[1:4]
        assert fr.size == 3

    def test_validate(self, populated_integration):
        integ = populated_integration
        assert not integ.is_valid
        integ.validate()
        assert integ.is_valid

        # no op if repeated
        integ.validate()
        assert integ.is_valid

        # if no valid frames, is not valid
        integ.is_valid = False
        integ.frames.valid[...] = False
        integ.validate()
        assert not integ.is_valid

    def test_validate_shift(self, capsys, populated_integration):
        integ = populated_integration
        m1 = integ.frames[0].mjd
        m2 = integ.frames[101].mjd
        m3 = integ.frames[-1].mjd

        # removes first 10 seconds of data (100 frames)
        integ.configuration.set_option('shift', 10)
        with set_log_level('DEBUG'):
            integ.validate()
        m4 = integ.frames[0].mjd
        m5 = integ.frames[-1].mjd

        assert np.allclose((m4 - m1) * 24 * 3600, 10, atol=0.1)
        assert m4 == m2
        assert m3 == m5

        assert 'Shifting data by 100 frames' in capsys.readouterr().out

    def test_validate_fillgaps(self, capsys, populated_integration):
        populated_integration.configuration.set_option('fillgaps', True)
        integ = populated_integration.copy()

        # no gaps
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Padding' not in capsys.readouterr().out

        # add a 1s gap before the last frame
        integ = populated_integration.copy()
        s1 = 1 / (24 * 3600)
        integ.frames.mjd[-1] += s1
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Padding with 10 empty frames' in capsys.readouterr().out

        # add a 10s gap: too large
        integ = populated_integration.copy()
        integ.frames.mjd[-1] += 10 * s1
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Could not fill gaps' in capsys.readouterr().err

    def test_validate_notch(self, capsys, populated_integration):
        # set without configuring
        integ = populated_integration.copy()
        integ.configuration.set_option('notch', True)
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Notching' not in capsys.readouterr().out

        # set with configuration
        integ = populated_integration.copy()
        integ.configuration.set_option('notch', True)
        integ.configuration.set_option('notch.frequencies', [10])
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Notching 1 bands' in capsys.readouterr().out

    def test_validate_select_frames(self, capsys, populated_integration):
        # set to remove first frame
        integ = populated_integration.copy()
        integ.configuration.set_option('frames', '1:*')
        assert integ.size == 1100
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Removing 1 frames outside range' in capsys.readouterr().out
        assert integ.size == 1099

        # take 800 from the middle
        integ = populated_integration.copy()
        integ.configuration.set_option('frames', '200:999')
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Removing 300 frames outside range' in capsys.readouterr().out
        assert integ.size == 800
        assert integ.is_valid

        # remove them all
        integ = populated_integration.copy()
        integ.configuration.set_option('frames', '1:0')
        with set_log_level('DEBUG'):
            integ.validate()
        capt = capsys.readouterr()
        assert 'Removing 1100 frames outside range' in capt.out
        assert 'No valid frames' in capt.err
        assert integ.size == 0
        assert not integ.is_valid

    def test_validate_avclip(self, capsys, populated_integration):
        integ = populated_integration.copy()

        # vclip auto clips all frames
        integ.configuration.set_option('vclip', 'auto')
        with set_log_level('DEBUG'):
            integ.validate()
        assert not integ.is_valid
        assert 'No valid frames' in capsys.readouterr().err

        # vclip a more appropriate range
        integ = populated_integration.copy()
        integ.configuration.set_option('vclip', '2:140')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        assert np.sum(integ.frames.valid) == 982

        # allow all speeds
        integ = populated_integration.copy()
        integ.configuration.set_option('vclip', '2:1000')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        assert np.sum(integ.frames.valid) == 1100

        # aclip for acceleration max: allow all accelerations
        integ = populated_integration.copy()
        integ.configuration.set_option('vclip', '2:1000')
        integ.configuration.set_option('aclip', 'inf')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        assert np.sum(integ.frames.valid) == 1100
        assert 'Median acceleration' in capsys.readouterr().out

        # aclip for acceleration max: allow about half
        integ = populated_integration.copy()
        integ.configuration.set_option('vclip', '2:1000')
        integ.configuration.set_option('aclip', '16')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        assert np.sum(integ.frames.valid) == 562

        # aclip for acceleration max: allow none
        integ = populated_integration.copy()
        integ.configuration.set_option('vclip', '2:1000')
        integ.configuration.set_option('aclip', '0')
        with set_log_level('DEBUG'):
            integ.validate()
        assert not integ.is_valid
        assert np.sum(integ.frames.valid) == 0

    def test_validate_lab(self, capsys, populated_integration):
        integ = populated_integration.copy()

        # lab will take speed from instrument definition
        integ.configuration.set_option('lab', True)
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        u = units.arcsec / units.s
        assert np.allclose(integ.average_scan_speed[0], 10 * u)
        assert np.allclose(integ.average_scan_speed[1], 0 * (u ** -2))

        # or from configuration
        integ = populated_integration.copy()
        integ.configuration.set_option('lab', True)
        integ.configuration.set_option('lab.scanspeed', 42)
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        assert np.allclose(integ.average_scan_speed[0], 42 * u)
        assert np.allclose(integ.average_scan_speed[1], 0 * (u ** -2))

    def test_validate_filter_kill(self, populated_integration):
        integ = populated_integration.copy()

        # filter kill without specifying bands: no op
        integ.configuration.set_option('filter.kill', True)
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        before = integ.frames.data

        # filter kill with bands: removes a specific fft frequency
        # signal from data
        integ = populated_integration.copy()
        integ.configuration.set_option('filter.kill', True)
        integ.configuration.set_option('filter.kill.bands', '0.1:1.0')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        after = integ.frames.data
        assert not np.allclose(before, after, equal_nan=True)

    def test_validate_range(self, capsys, populated_integration):
        integ = populated_integration.copy()

        # set acceptable data range
        integ.configuration.set_option('range', '0.0:0.1')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        assert 'Flagging out-of-range data. ' \
               '44 channel(s) discarded' in capsys.readouterr().out
        assert integ.n_channels == 121 - 44

        # flag all data
        integ = populated_integration.copy()
        integ.configuration.set_option('range', '0.1:0.5')
        with set_log_level('DEBUG'):
            integ.validate()
        assert not integ.is_valid
        capt = capsys.readouterr()
        assert 'Flagging out-of-range data. ' \
               '121 channel(s) discarded' in capt.out
        assert 'Too few valid channels' in capt.err

        # flag no data
        integ = populated_integration.copy()
        integ.configuration.set_option('range', '-inf:inf')
        integ.configuration.set_option('range.flagfraction', '0.5')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        capt = capsys.readouterr()
        assert 'Flagging out-of-range data. ' \
               '0 channel(s) discarded' in capt.out
        assert integ.n_channels == 121

        # don't check flag fraction: channels are kept
        integ = populated_integration.copy()
        integ.configuration.set_option('range', '0.0:0.1')
        del integ.configuration['range.flagfraction']
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        capt = capsys.readouterr()
        assert 'Flagging out-of-range data' not in capt.out
        assert integ.n_channels == 121

    def test_validate_minlength(self, capsys, populated_integration):
        integ = populated_integration.copy()

        # set acceptable length in frames
        integ.configuration.set_option('subscan.minlength', '10')
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.is_valid
        assert 'Integration is too short' not in capsys.readouterr().err

        # set longer than integration
        integ = populated_integration.copy()
        integ.configuration.set_option('subscan.minlength', '2000')
        with set_log_level('DEBUG'):
            integ.validate()
        assert not integ.is_valid
        assert 'Integration is too short' in capsys.readouterr().err

    def test_validate_tau(self, capsys, populated_integration):
        integ = populated_integration.copy()
        integ.configuration.set_option('tau', 10)
        integ.validate()
        assert integ.is_valid
        assert 'Setting zenith tau to 10.0' in capsys.readouterr().out

        # unconfigured spec for this instrument
        integ = populated_integration.copy()
        integ.configuration.set_option('tau', 'atran')
        with pytest.raises(ValueError) as err:
            integ.validate()
        assert 'does not contain a tau value' in str(err)

    def test_validate_gain(self, populated_integration):
        integ = populated_integration.copy()
        integ.validate()
        before = integ.gain

        # specified scale factor is applied to gain
        integ = populated_integration.copy()
        integ.configuration.set_option('scale', 15)
        integ.validate()
        after = integ.gain
        assert np.allclose(before / 15, after)

        # gain can also be inverted
        integ = populated_integration.copy()
        integ.configuration.set_option('invert', True)
        integ.validate()
        after = integ.gain
        assert np.allclose(before * -1, after)

    def test_validate_slim(self, capsys, populated_integration):
        # set a range to invalidate some channels
        populated_integration.configuration.set_option('range', '0.0:0.1')
        integ = populated_integration.copy()
        assert integ.n_channels == 121

        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Slimmed' in capsys.readouterr().out
        assert integ.n_channels < 121

        # set config to noslim: dead channels not dropped
        integ = populated_integration.copy()
        integ.configuration.set_option('noslim', True)
        with set_log_level('DEBUG'):
            integ.validate()
        assert 'Slimmed' not in capsys.readouterr().out
        assert integ.n_channels == 121

    def test_validate_jackknife(self, capsys, populated_integration):
        integ = populated_integration.copy()
        integ.configuration.set_option('jackknife', False)
        integ.validate()
        assert integ.gain == 1
        assert np.all(integ.frames.sign == 1)

        # jackknife randomly inverts gain for the integration
        integ = populated_integration.copy()
        integ.configuration.set_option('jackknife', True)
        integ.validate()
        assert (integ.gain == -1 or integ.gain == 1)
        assert np.all(integ.frames.sign == 1)

        # jackknife.frames randomly inverts sample gains
        integ = populated_integration.copy()
        integ.configuration.set_option('jackknife', False)
        integ.configuration.set_option('jackknife.frames', True)
        with set_log_level('DEBUG'):
            integ.validate()
        assert integ.gain == 1
        assert not np.all(integ.frames.sign == 1)
        assert np.all((integ.frames.sign == 1) | (integ.frames.sign == -1))
        assert 'JACKKNIFE: Randomly inverted ' \
               'frames' in capsys.readouterr().out

    def test_get_first_last(self, populated_integration):
        integ = populated_integration
        assert np.all(integ.get_first_frame().data == integ.frames.data[0])
        assert np.all(integ.get_first_frame(10).data == integ.frames[10].data)
        assert np.all(integ.get_last_frame().data == integ.frames[-1].data)
        assert np.all(integ.get_last_frame(10).data == integ.frames[9].data)

    def test_select_frames(self, capsys, populated_integration):
        integ = populated_integration.copy()
        del integ.configuration['frames']

        # no op if not configured
        with set_log_level('DEBUG'):
            integ.select_frames()
        assert integ.size == 1100
        test = integ[100].data
        assert 'Removing' not in capsys.readouterr().out

        # select all but the first and last 100
        integ = populated_integration.copy()
        integ.configuration.set_option('frames', '100:-100')
        with set_log_level('DEBUG'):
            integ.select_frames()
        assert integ.size == 900
        assert np.allclose(integ[0].data, test, equal_nan=True)
        assert 'Removing 200 frames' in capsys.readouterr().out

    def test_check_range(self, capsys, populated_integration):
        integ = populated_integration.copy()
        del integ.configuration['range']

        # no op if not configured
        with set_log_level('DEBUG'):
            integ.check_range()
        assert 'Flagging out-of-range' not in capsys.readouterr().out

        # configure to flag bad channels
        integ = populated_integration.copy()
        integ.configuration.set_option('range', '0.0:0.1')
        integ.configuration.set_option('range.flagfraction', '0.05')
        with set_log_level('DEBUG'):
            integ.check_range()
        assert 'Flagging out-of-range data. ' \
               '44 channel(s) discarded' in capsys.readouterr().out

    def test_trim(self, capsys, populated_integration):
        integ = populated_integration.copy()

        # all valid
        assert np.sum(integ.frames.valid) == 1100
        integ.trim()
        assert integ.size == 1100

        # first and last invalid: trim start, then end
        integ = populated_integration.copy()
        integ.frames.valid[0] = False
        integ.frames.valid[-1] = False
        with set_log_level('DEBUG'):
            integ.trim(start=True, end=False)
        assert integ.size == 1099
        assert 'Trimmed to 1099 frames' in capsys.readouterr().out

        integ.frames.valid[0] = False
        with set_log_level('DEBUG'):
            integ.trim(start=False, end=True)
        assert integ.size == 1098
        assert 'Trimmed to 1098 frames' in capsys.readouterr().out

        # all invalid: does not trim
        integ.frames.valid[:] = False
        with set_log_level('DEBUG'):
            integ.trim()
        assert integ.size == 1098
        assert 'Trimmed' not in capsys.readouterr().out

        # mock dependents: should be modified to match trimmed frames
        integ = populated_integration.copy()
        d = Dependents(integ, 'test')
        d.for_frame = np.arange(1100)
        integ.add_dependents(d)
        integ.frames.valid[0:3] = False
        integ.frames.valid[-1] = False
        integ.trim()
        assert integ.dependents['test'].for_frame.size == 1096
        assert np.allclose(integ.dependents['test'].for_frame,
                           np.arange(1100)[3:-1])

    def test_get_pa(self, populated_integration):
        integ = populated_integration

        # simulated data
        pa = integ.get_pa()
        assert np.allclose(pa, -0.43525 * units.radian)

        # set first frame 90 deg
        integ.frames.sin_pa[0] = 1
        integ.frames.cos_pa[0] = 0

        # last frame 0 deg
        integ.frames.sin_pa[-1] = 0
        integ.frames.cos_pa[-1] = 1

        # average of first and last is 45
        pa = integ.get_pa()
        assert np.allclose(pa, 45 * units.deg)

    def test_scale(self, populated_integration):
        integ = populated_integration
        test = integ.frames.data.copy()
        integ.scale(2.0)
        # data is directly scaled
        assert np.allclose(integ.frames.data, test * 2.0)

    def test_frames_for(self, populated_integration):
        integ = populated_integration

        # none or too large time: filter time scale for instrument
        assert integ.frames_for() == 1100
        assert integ.frames_for(integ.filter_time_scale) == 1100
        assert integ.frames_for(1000 * units.s) == 1100

        # smaller time: returns number of frames, using sampling interval
        assert integ.frames_for(10 * units.s) == 100
        assert integ.frames_for(1 * units.s) == 10
        assert integ.frames_for(integ.info.instrument.sampling_interval) == 1

    def test_power2_frames_for(self, populated_integration):
        integ = populated_integration

        # filter time scale for instrument: returns all frames,
        # rounded up to nearest power of 2
        assert integ.power2_frames_for() == 2048

        # smaller time: returns number of frames in interval
        # rounded up to nearest power of 2
        assert integ.power2_frames_for(10 * units.s) == 128
        assert integ.power2_frames_for(1 * units.s) == 16
        assert integ.power2_frames_for(
            integ.info.instrument.sampling_interval) == 1

    def test_filter_frames_for(self, populated_integration):
        # frames available for various filter options
        integ = populated_integration

        # raises error if no spec or value
        with pytest.raises(ValueError):
            integ.filter_frames_for()

        # specify value: 10 frames in 1s, raised to nearest power of 2
        assert integ.filter_frames_for(default_time=1 * units.s) == 16
        assert integ.filter_frames_for(spec=1 * units.s) == 16
        assert integ.filter_frames_for(spec=1) == 16

        # other specs
        assert integ.filter_frames_for(spec='auto') == 128
        integ.configuration.set_option('photometry', True)
        assert integ.filter_frames_for(spec='auto') == 128
        assert integ.filter_frames_for(spec='max') == 2048

    def test_get_positions(self, populated_integration):
        integ = populated_integration
        flags = integ.motion_flagspace.flags

        # no flag
        pos = integ.get_positions(0)
        assert np.all(pos.x == 0 * units.arcsec)
        assert np.all(pos.y == 0 * units.arcsec)

        # telescope motion
        pos = integ.get_positions(flags.TELESCOPE)
        assert np.allclose(np.min(pos.x), 152.5 * units.deg, atol=1)
        assert np.allclose(np.max(pos.x), 152.5 * units.deg, atol=1)
        assert np.allclose(np.min(pos.y), 17.5 * units.deg, atol=1)
        assert np.allclose(np.max(pos.y), 17.5 * units.deg, atol=1)

        # telescope and chopper: no difference for unchopped instrument
        telchp_pos = integ.get_positions(flags.TELESCOPE | flags.CHOPPER)
        assert np.allclose(telchp_pos.x, pos.x)
        assert np.allclose(telchp_pos.y, pos.y)

        # chopper only: all zero
        chp_pos = integ.get_positions(flags.CHOPPER)
        assert np.allclose(chp_pos.x, 0)
        assert np.allclose(chp_pos.y, 0)

        # telescope with project_gls: x changes, y stays same
        telgls_pos = integ.get_positions(flags.TELESCOPE | flags.PROJECT_GLS)
        assert not np.allclose(telgls_pos.x, pos.x)
        assert np.allclose(telgls_pos.y, pos.y)
        assert np.allclose(np.min(telgls_pos.x), 144.5 * units.deg, atol=1)
        assert np.allclose(np.max(telgls_pos.x), 144.5 * units.deg, atol=1)

        # scanning: same with and without chopper
        scan_pos = integ.get_positions(flags.SCANNING)
        scanchp_pos = integ.get_positions(flags.SCANNING | flags.CHOPPER)
        assert np.allclose(np.min(scan_pos.x), -107.5 * units.arcsec, atol=1)
        assert np.allclose(np.max(scan_pos.x), 107.5 * units.arcsec, atol=1)
        assert np.allclose(np.min(scan_pos.y), -117 * units.arcsec, atol=1)
        assert np.allclose(np.max(scan_pos.y), 117 * units.arcsec, atol=1)
        assert np.allclose(scanchp_pos.x, scan_pos.x)
        assert np.allclose(scanchp_pos.y, scan_pos.y)

    def test_get_smooth_positions(self, populated_integration):
        integ = populated_integration
        flags = integ.motion_flagspace.flags
        pos = integ.get_positions(flags.SCANNING)

        # no smooth: same as raw positions
        del integ.configuration['positions.smooth']
        spos = integ.get_smooth_positions(flags.SCANNING)
        assert np.allclose(pos.x, spos.x)
        assert np.allclose(pos.y, spos.y)

        # set smooth time to 1s
        integ.configuration.set_option('positions.smooth', 1)
        spos = integ.get_smooth_positions(flags.SCANNING)

        # some nans in output, at beginning and end
        assert np.sum(~np.isfinite(spos.x)) == 9
        assert np.sum(~np.isfinite(spos.y)) == 9

        # the rest should be close to original values
        idx = np.isfinite(spos.x) & np.isfinite(spos.y)
        assert np.allclose(spos.x[idx], pos.x[idx], atol=11 * units.arcsec)
        assert np.allclose(spos.y[idx], pos.y[idx], atol=11 * units.arcsec)

    def test_get_speed_clip_range(self, populated_integration):
        integ = populated_integration

        del integ.configuration['vclip']
        rng = integ.get_speed_clip_range()
        assert rng.min == -np.inf
        assert rng.max == np.inf

        integ.configuration.set_option('vclip', 'auto')
        rng = integ.get_speed_clip_range()
        assert rng.min == 5 * units.arcsec / units.s
        assert rng.max == 40 * units.arcsec / units.s

        integ.configuration.set_option('vclip', '4:30')
        rng = integ.get_speed_clip_range()
        assert rng.min == 4 * units.arcsec / units.s
        assert rng.max == 30 * units.arcsec / units.s

        # chopped sets range min to zero always
        integ.configuration.set_option('chopped', True)
        rng = integ.get_speed_clip_range()
        assert rng.min == 0 * units.arcsec / units.s
        assert rng.max == 30 * units.arcsec / units.s

    def test_velocity_sigma_clip(self, populated_integration):
        integ = populated_integration.copy()

        assert np.sum(integ.frames.valid) == 1100
        # average speed and weight (1/rms)
        s1, w1 = integ.get_typical_scanning_speed()

        # clip speeds to 5 sigma: all still valid
        integ.velocity_clip(sigma_clip=5.0, strict=True)
        assert np.sum(integ.frames.valid) == 1100

        # clip speeds to 1.5 sigma: about half invalid
        integ = populated_integration.copy()
        integ.velocity_clip(sigma_clip=1.5, strict=True)
        assert np.sum(integ.frames.valid) == 524
        s2, w2 = integ.get_typical_scanning_speed()

        # average speed is close but not the same;
        # rms goes down / weight goes up
        assert not np.allclose(s1, s2)
        assert np.allclose(s1, s2, rtol=0.3)
        assert w2 > w1

        # invalid sigma defaults to 5
        integ = populated_integration.copy()
        integ.velocity_clip(sigma_clip='a', strict=True)
        assert np.sum(integ.frames.valid) == 1100

        # non-strict flags instead of marking invalid
        integ = populated_integration.copy()
        integ.velocity_clip(sigma_clip=1.5, strict=False)
        assert np.sum(integ.frames.valid) == 1100

    def test_vcsv(self, tmpdir, populated_integration):
        # make sure working directory is tmpdir
        integ = populated_integration
        integ.configuration.work_path = str(tmpdir)

        # configure to save velocity csv files
        integ.configuration.set_option('vcsv', True)
        with tmpdir.as_cwd():
            integ.velocity_clip(strict=True, sigma_clip=1.5)
            assert os.path.isfile('used2.0.csv')
            assert os.path.isfile('cleared2.0.csv')
            # again: appends (1) suffix
            integ.velocity_clip(strict=True, sigma_clip=1.5)
            assert os.path.isfile('used2.0(1).csv')
            assert os.path.isfile('cleared2.0(1).csv')

    def test_acceleration_clip(self, populated_integration):
        integ = populated_integration

        # don't specify aclip value: all allowed
        del integ.configuration['aclip']
        integ.acceleration_clip()
        assert np.sum(integ.frames.valid) == 1100

        # specify some value: some allowed
        integ.configuration.set_option('aclip', '16')
        integ.acceleration_clip()
        assert np.sum(integ.frames.valid) == 562

    def test_downsample(self, capsys, populated_integration):
        integ = populated_integration.copy()
        interval = integ.info.instrument.sampling_interval
        mval = np.nanmean(integ.frames.data)

        # unconfigure downsample
        del integ.configuration['downsample']
        with set_log_level('DEBUG'):
            integ.downsample()
        assert 'Downsampling' not in capsys.readouterr().out
        assert integ.size == 1100

        # configure auto: returns 1 by default
        integ = populated_integration.copy()
        integ.configuration.set_option('downsample', 'auto')
        with set_log_level('DEBUG'):
            integ.downsample()
        assert 'Downsampling' not in capsys.readouterr().out
        assert integ.size == 1100

        # configure specific value: halve the frames
        integ = populated_integration.copy()
        integ.configuration.set_option('downsample', 2.0)
        with set_log_level('DEBUG'):
            integ.downsample()
        assert 'Downsampling' in capsys.readouterr().out
        assert integ.size == 548
        assert integ.info.instrument.sampling_interval == interval * 2
        # mean value should be about the same
        assert np.allclose(np.nanmean(integ.frames.data), mval, rtol=.05)

        # configure too big a downsample for the scan: ignored
        integ = populated_integration.copy()
        integ.configuration.set_option('downsample', 1000)
        with set_log_level('DEBUG'):
            integ.downsample()
        assert 'too short to downsample' in capsys.readouterr().err
        assert integ.size == 1100
        assert integ.info.instrument.sampling_interval == interval

    def test_downsample_factor(self, capsys, mocker, populated_integration):
        integ = populated_integration.copy()

        # default
        del integ.configuration['downsample']
        assert integ.get_downsample_factor() == 1

        # specific value: gets clipped and/or rounded
        integ.configuration.set_option('downsample', 3.5)
        assert integ.get_downsample_factor() == 3
        integ.configuration.set_option('downsample', -3)
        assert integ.get_downsample_factor() == 1

        # auto
        integ.configuration.set_option('downsample', 'auto')
        assert integ.get_downsample_factor() == 1

        # set zero speed
        integ.average_scan_speed = (0, np.inf)
        assert integ.get_downsample_factor() == 1
        assert 'No automatic downsampling for ' \
               'zero scan speed' in capsys.readouterr().err

        # set nan speed
        mocker.patch.object(integ, 'calculate_scan_speed_stats',
                            return_value=(np.nan, np.nan))
        integ.average_scan_speed = (np.nan, np.nan)
        assert integ.get_downsample_factor() == 1
        assert 'No automatic downsampling for ' \
               'unknown scanning speed' in capsys.readouterr().err

        # set unknown sampling interval
        integ = populated_integration.copy()
        integ.configuration.set_option('downsample', 'auto')
        integ.average_scan_speed = (1, 1)
        integ.info.instrument.sampling_interval = np.nan * units.s
        assert integ.get_downsample_factor() == 1
        assert 'No automatic downsampling for ' \
               'unknown sampling interval' in capsys.readouterr().err

        # set unknown point size
        integ = populated_integration.copy()
        integ.configuration.set_option('downsample', 'auto')
        mocker.patch.object(integ.scan, 'get_point_size',
                            return_value=np.nan)
        assert integ.get_downsample_factor() == 1
        assert 'No automatic downsampling for ' \
               'unknown point size' in capsys.readouterr().err

        # set negligible scan speed
        integ = populated_integration.copy()
        mocker.patch.object(integ.scan, 'get_point_size',
                            return_value=1e20 * units.arcsec)
        assert integ.get_downsample_factor() == 1
        assert 'No automatic downsampling for ' \
               'negligible scan speed' in capsys.readouterr().err

    def test_fill_nogaps(self, capsys, populated_integration):
        # no gaps: no op
        with set_log_level('DEBUG'):
            populated_integration.fill_gaps()
        assert 'Padding' not in capsys.readouterr().out

    def test_notch_filter(self, capsys, populated_integration):
        populated_integration.configuration.set_option('notch', True)
        populated_integration.configuration.set_option(
            'notch.frequencies', [10])
        integ = populated_integration.copy()

        # set harmonics
        integ.configuration.set_option('notch.harmonics', 4)
        with set_log_level('DEBUG'):
            integ.notch_filter()
        assert 'Notching 3 bands' in capsys.readouterr().out

        # set bands
        integ = populated_integration.copy()
        integ.configuration.set_option('notch.bands', '40:50')
        with set_log_level('DEBUG'):
            integ.notch_filter()
        assert 'Notching 101 bands' in capsys.readouterr().out

    def test_process_notch_filter(self, capsys, populated_integration):
        integ = populated_integration.copy()
        data = integ.frames.data.copy()

        # all invalid in block: no op
        integ.frames.valid[0:20] = False
        integ.process_notch_filter_block(0, 10, 10, [1, 2], True)
        assert np.allclose(integ.frames.data, data)

        # not invalid: removes some frequencies
        integ = populated_integration.copy()
        integ.process_notch_filter_block(0, 10, 10, [1, 2], True)
        assert not np.allclose(integ.frames.data, data)

    def test_offset(self, populated_integration):
        integ = populated_integration.copy()
        data = integ.frames.data.copy()

        # no flags
        integ.offset(0.1)
        assert np.allclose(integ.frames.data, data + 0.1)

        # flag some channels: offset applies only to unflagged
        integ = populated_integration.copy()
        bad_channel_flag = integ.channel_flagspace.flags.DEAD
        integ.channels.data.set_flags(bad_channel_flag,
                                      indices=[0, 1, 2])

        integ.offset(0.1)
        assert np.allclose(integ.frames.data[:, 0:3], data[:, 0:3])
        assert np.allclose(integ.frames.data[:, 3:], data[:, 3:] + 0.1)

    def test_get_position_signal(self, populated_integration):
        integ = populated_integration
        flags = integ.motion_flagspace.flags

        # telescope motion flag
        sig = integ.get_position_signal(flags.TELESCOPE, 'x')
        assert isinstance(sig, Signal)
        assert sig.integration is integ

        # value is x telescope position
        assert np.allclose(sig.value, 152.5 * units.deg, atol=1)

    def test_get_acceleration_signal(self, populated_integration):
        integ = populated_integration

        # x acceleration
        sig = integ.get_acceleration_signal('x')
        assert isinstance(sig, Signal)
        assert sig.integration is integ

        # value ranges negative to positive in daisy pattern
        acc = units.arcsec / units.s ** 2
        assert np.allclose(sig.value.min(), -18.7 * acc, atol=0.1)
        assert np.allclose(sig.value.max(), 18.7 * acc, atol=0.1)

    def test_get_spectra(self, populated_integration):
        integ = populated_integration
        assert np.sum(integ.frames.valid) == 1100

        freq, spec = integ.get_spectra()

        # frequency array
        assert freq.shape == (551,)
        assert np.min(freq) == 0 * units.Hz
        assert np.max(freq) == 5 * units.Hz

        # spectrum is n_freq x n_channel
        assert spec.shape == (551, 121)

        # expect max power for all channels to be at the same frequency
        idx = 44
        assert np.allclose(freq[idx], 0.4 * units.Hz)
        assert np.all(np.argmax(spec, axis=0) == idx)

        # same even if some frames are invalid
        integ.velocity_clip(sigma_clip=1.5, strict=True)
        assert np.sum(integ.frames.valid) == 524
        freq, spec = integ.get_spectra()
        assert np.all(np.argmax(spec, axis=0) == idx)

    def test_write_products(self, tmpdir, capsys, reduced_hawc_scan):
        integ = reduced_hawc_scan.integrations[0]
        integ.configuration.work_path = str(tmpdir)

        # add a signal to write
        integ.add_signal(
            integ.get_acceleration_signal(
                'x', mode=CorrelatedMode(name='test')))

        # invalidate and flag a couple frames
        integ.frames.valid[0] = False
        bad_flag = integ.frames.flagspace.flags.FLAG_WEIGHT
        integ.frames.set_flags(bad_flag, indices=[1, 2, 3])

        # add a flatfield name: will be used first time, deleted,
        # then default will be used
        integ.configuration.set_option('write.flatfield.name',
                                       'test_flat.fits')

        file_id = '2016-12-14_HA_F999-sim-999'
        output = {
            'write.pattern': f'pattern-{file_id}.dat',
            'write.pixeldata': f'pixel-{file_id}.dat',
            'write.flatfield': 'test_flat.fits',
            'write.flatfield:True': f'flat-{file_id}.fits',
            'write.covar:full': f'covar-{file_id}.fits',
            'write.ascii': f'{file_id}-{file_id}.tms',
            'write.signals:test': f'test-{file_id}.tms',
            'write.spectrum:Hamming': f'{file_id}.spec',
            'write.coupling:subarrays': f'{file_id}.subarrays-coupling.dat'}
        with tmpdir.as_cwd():
            for opt, fname in output.items():
                try:
                    opt, opt_val = opt.split(':')
                    integ.configuration.set_option(opt, opt_val)
                except ValueError:
                    integ.configuration.set_option(opt, True)
                integ.write_products()

                # simple check for output presence
                assert os.path.isfile(fname)
                os.remove(fname)

                # invoke an error: should warn and continue
                integ.configuration.work_path = 'bad'
                integ.write_products()
                assert not os.path.isfile(fname)
                assert 'Could not write' in capsys.readouterr().err

                # reset config
                integ.configuration.work_path = str(tmpdir)
                del integ.configuration[opt]

    def test_write_spectra(self, tmpdir, populated_integration):
        integ = populated_integration
        integ.configuration.work_path = str(tmpdir)

        with tmpdir.as_cwd():
            # default window name and size
            integ.write_spectra()

            file_id = 'Simulation.1-1'
            fname = f'{file_id}.spec'
            assert os.path.isfile(fname)

            with open(fname) as fh:
                spec = fh.readlines()

            assert spec[0] == '# SOFSCAN Residual Detector Power Spectra\n'
            assert 'Window Function: Hamming' in spec[11]
            assert 'Window Size: 2200 samples' in spec[12]

            # 551 freq + header
            assert len(spec) == 567

    def test_write_covariances(self, tmpdir, capsys, populated_integration):
        integ = populated_integration
        integ.configuration.work_path = str(tmpdir)

        file_id = 'Simulation.1-1'
        with tmpdir.as_cwd():
            # full covar
            integ.write_covariances()
            fname = f'covar-{file_id}.fits'
            assert os.path.isfile(fname)

            # reduced
            integ.configuration.set_option('write.covar', 'reduced')
            integ.write_covariances()
            fname = f'covar-{file_id}.reduced.fits'
            assert os.path.isfile(fname)

            # missing channel division
            integ.configuration.set_option('write.covar', 'test')
            integ.write_covariances()
            fname = f'covar-{file_id}.test.fits'
            assert not os.path.isfile(fname)
            assert 'Cannot write covariance ' \
                   'for test' in capsys.readouterr().err

            # real channel division
            integ.configuration.set_option('write.covar', 'bias')
            integ.write_covariances()
            fname = f'covar-{file_id}.bias.fits'
            assert os.path.isfile(fname)

    def test_write_covariance_to_file(self, tmpdir, populated_integration):
        integ = populated_integration
        integ.configuration.work_path = str(tmpdir)

        file_id = 'Simulation.1-1'
        covar = np.arange(100).reshape((10, 10))
        with tmpdir.as_cwd():
            fname = f'test-{file_id[:-2]}-{file_id}.fits'

            # no covar
            integ.write_covariance_to_file('test', None)
            assert not os.path.isfile(fname)

            # with covar
            integ.write_covariance_to_file('test', covar)
            assert os.path.isfile(fname)

            # with condense option
            integ.configuration.set_option('write.covar.condensed', True)
            integ.write_covariance_to_file('test', covar)
            assert os.path.isfile(fname)

    def test_condense_covariance(self):
        covar = np.arange(100, dtype=float).reshape((10, 10))

        # no invalid: returns same
        result = Integration.condense_covariance(covar)
        assert result is covar

        # some row/col pairs invalid or zero: are dropped
        covar[0, :] = 0
        covar[:, 0] = 0
        result = Integration.condense_covariance(covar)
        assert np.allclose(result, covar[1:, 1:])

        covar[-1, :] = np.nan
        covar[:, -1] = np.nan
        result = Integration.condense_covariance(covar)
        assert np.allclose(result, covar[1:-1, 1:-1])

    def test_write_coupling_gains(self, tmpdir, capsys, mocker,
                                  populated_integration):
        integ = populated_integration
        integ.configuration.work_path = str(tmpdir)

        with tmpdir.as_cwd():
            # no op if signal not specified
            integ.write_coupling_gains(None)
            assert 'Writing coupling gains' not in capsys.readouterr().out
            integ.write_coupling_gains([])
            assert 'Writing coupling gains' not in capsys.readouterr().out

            # invalid signal: same
            integ.write_coupling_gains(['test'])
            assert 'Writing coupling gains' not in capsys.readouterr().out

            # mock a signal to write
            mode = CorrelatedMode(name='test')
            sig = integ.get_acceleration_signal('x', mode=mode)
            mocker.patch.object(integ, 'get_signal',
                                return_value=sig)
            mocker.patch.object(mode, 'get_gains',
                                return_value=np.ones(22))
            integ.write_coupling_gains(['bias'])
            assert 'Writing coupling gains' in capsys.readouterr().out

    def test_get_crossing_time(self, mocker,
                               populated_integration, reduced_scan):

        # size from instrument
        integ = populated_integration
        assert np.allclose(integ.get_crossing_time(),
                           0.1 * units.s, atol=.01)

        # size from scan source model
        integ = reduced_scan.integrations[0]
        assert np.allclose(integ.get_crossing_time(),
                           0.1 * units.s, atol=.01)

        # duration from chopper (mocked)
        integ.chopper = Chopper()
        integ.chopper.frequency = 1.0
        integ.chopper.positions = 100
        integ.chopper.efficiency = 1.0
        assert integ.chopper.stare_duration == 0.01 * units.s
        assert np.allclose(integ.get_crossing_time(),
                           0.01 * units.s, atol=.01)

        # add modulation time (mocked)
        integ = populated_integration
        mocker.patch.object(integ, 'get_modulation_frequency',
                            return_value=2.0 * units.Hz)
        assert np.allclose(integ.get_crossing_time(),
                           0.6 * units.s, atol=.01)

    def test_get_modulation_frequency(self, populated_integration):
        integ = populated_integration

        # zero if no chopper
        assert integ.get_modulation_frequency(None) == 0 * units.Hz

        # otherwise chopper frequency
        integ.chopper = Chopper()
        integ.chopper.frequency = 1.0
        assert integ.get_modulation_frequency(None) == 1.0 * units.Hz

    def test_get_mjd(self, populated_integration):
        integ = populated_integration

        integ.frames.mjd[0] = 12345
        integ.frames.mjd[-1] = 12347

        # returns midpoint mjd, averaging first and last
        assert integ.get_mjd() == 12346

    def test_get_ascii_header(self, populated_integration):
        integ = populated_integration

        hdr = integ.get_ascii_header()
        assert hdr.count('\n') == 8
        assert 'Integration' not in hdr

        # scan size > 1
        integ.scan.integrations.append(integ.copy())
        hdr = integ.get_ascii_header()
        assert hdr.count('\n') == 9
        assert 'Integration: 1' in hdr

    def test_get_id(self, populated_integration):
        integ = populated_integration

        integ.integration_number = None
        assert integ.get_id() == '1'
        integ.integration_number = 0
        assert integ.get_id() == '1'
        integ.integration_number = 1
        assert integ.get_id() == '2'

    def test_get_standard_id(self, populated_integration):
        integ = populated_integration

        # with scan: full id
        assert integ.get_standard_id() == 'Simulation.1|1'

        integ.scan = None
        assert integ.get_standard_id() == '1'

    def test_get_channel_weights(self, mocker, populated_integration):
        integ = populated_integration

        robust = mocker.patch.object(integ, 'get_robust_channel_weights')
        diff = mocker.patch.object(integ, 'get_differential_channel_weights')
        rms = mocker.patch.object(integ, 'get_rms_channel_weights')

        # method depends on flag, default is rms
        integ.get_channel_weights(None)
        rms.assert_called_once()
        integ.get_channel_weights('robust')
        robust.assert_called_once()
        integ.get_channel_weights('differential')
        diff.assert_called_once()

    def test_get_channel_weights_methods(self, populated_integration):
        integ = populated_integration

        # smoke test for various options: calls numba functions
        # under the hood for weight setting
        integ.get_robust_channel_weights()
        assert integ.comments[-1] == '[W]'

        integ.get_differential_channel_weights()
        assert integ.comments[-1] == 'w'

        integ.get_rms_channel_weights()
        assert integ.comments[-1] == 'W'

    def test_calculate_source_nefd(self, populated_integration):
        integ = populated_integration

        expected = 0.3162277
        integ.calculate_source_nefd()
        assert np.allclose(integ.nefd, expected)

        # divide by sqrt of scan weight
        integ.configuration.set_option('nefd.map')
        integ.scan.weight = 2.0
        integ.calculate_source_nefd()
        assert np.allclose(integ.nefd, expected / np.sqrt(2.0))

        integ.scan.weight = 0
        integ.calculate_source_nefd()
        assert np.isinf(integ.nefd)

    def test_get_time_weights(self, populated_integration):
        integ = populated_integration
        flag = integ.flagspace.flags.FLAG_WEIGHT

        # no bad frames
        integ.get_time_weights()
        assert np.sum(integ.frames.is_flagged(flag)) == 0

        # set noiserange flag so some frames are flagged
        integ.configuration.set_option('weighting.frames.noiserange',
                                       '0.3:1.0')

        integ.get_time_weights()
        assert np.sum(integ.frames.is_flagged(flag)) == 640

        # set flag=False: returns without flagging
        integ.get_time_weights(flag=False)
        assert np.sum(integ.frames.is_flagged(flag)) == 0

        # unset noiserange: same
        del integ.configuration['weighting.frames.noiserange']
        integ.get_time_weights()
        assert np.sum(integ.frames.is_flagged(flag)) == 0

        # unset resolution: uses block size 1, still okay
        del integ.configuration['weighting.frames.resolution']
        integ.get_time_weights()
        assert np.sum(integ.frames.is_flagged(flag)) == 0

        # pass channel group instead: should be same
        group = integ.channels.create_channel_group()
        integ.get_time_weights(channels=group)
        assert np.sum(integ.frames.is_flagged(flag)) == 0

    def test_dejump_frames(self, capsys, populated_integration):
        integ = populated_integration.copy()
        flag = integ.flagspace.flags.TIME_WEIGHTING_FLAGS
        flag2 = integ.flagspace.flags.FLAG_WEIGHT

        with set_log_level('DEBUG'):
            integ.dejump_frames()
        capt = capsys.readouterr()
        assert 'levelled 8 (1049 frames)' in capt.out
        assert 'removed 1 (37 frames)' in capt.out
        assert np.sum(integ.frames.is_flagged(flag)) == 37
        assert np.sum(integ.frames.is_flagged(flag2)) == 0

        # reduce resolution: no more jumps
        integ = populated_integration.copy()
        integ.configuration.set_option('dejump.resolution', 4)
        with set_log_level('DEBUG'):
            integ.dejump_frames()
        capt = capsys.readouterr()
        assert 'levelled 0 (0 frames)' in capt.out
        assert 'removed 0 (0 frames)' in capt.out
        assert np.sum(integ.frames.is_flagged()) == 0

        # unconfigure minlength: uses crossing time
        integ = populated_integration.copy()
        del integ.configuration['dejump.minlength']
        with set_log_level('DEBUG'):
            integ.dejump_frames()
        capt = capsys.readouterr()
        assert 'levelled 9 (1086 frames)' in capt.out
        assert 'removed 0 (0 frames)' in capt.out
        assert np.sum(integ.frames.is_flagged()) == 0

        # set small minlength: sets no minimum frame size
        integ = populated_integration.copy()
        integ.configuration.set_option('dejump.minlength', 0)
        with set_log_level('DEBUG'):
            integ.dejump_frames()
        capt = capsys.readouterr()
        assert 'levelled 0 (0 frames)' in capt.out
        assert 'removed 9 (1086 frames)' in capt.out
        assert np.sum(integ.frames.is_flagged()) == 1086

        # set frame weights
        integ = populated_integration.copy()
        integ.configuration.set_option('weighting.frames', True)
        integ.configuration.set_option('weighting.frames.noiserange',
                                       '0.3:1.0')
        with set_log_level('DEBUG'):
            integ.dejump_frames()
        capt = capsys.readouterr()
        assert 'levelled 8 (1049 frames)' in capt.out
        assert 'removed 1 (37 frames)' in capt.out

        assert np.sum(integ.frames.is_flagged(flag)) == 37
        assert np.sum(integ.frames.is_flagged(flag2)) == 640

        # add a signal to level
        integ = populated_integration.copy()
        sig = integ.get_acceleration_signal(
            'x', mode=CorrelatedMode(name='test'))
        val = sig.value.copy()
        integ.add_signal(sig)
        with set_log_level('DEBUG'):
            integ.dejump_frames()
        assert not np.allclose(sig.value, val, equal_nan=True)

    def test_get_mean_level(self, populated_integration):
        integ = populated_integration

        # median values and weights by channel
        values, weights = integ.get_mean_level()
        assert values.size == 121
        assert weights.size == 121

        # all valid, similar values
        assert np.allclose(values, .014, atol=.05)
        assert np.allclose(weights, 1100)

    def test_level(self, populated_integration):
        integ = populated_integration
        data = integ.frames.data.copy()

        mval = np.full(121, 10.0)
        wval = np.full(121, 1)
        result = integ.level(channel_means=mval, channel_weights=wval)
        assert result.size == 121
        assert np.all(result)

        # passed means are directly subtracted
        assert np.allclose(integ.frames.data, data - mval[None])

    def test_get_time_stream(self, populated_integration):
        integ = populated_integration

        # all valid, unweighted: just return data copy
        data = integ.get_time_stream()
        assert np.allclose(data, integ.frames.data)

        # invalid frames marked with NaN
        integ.frames.valid[:3] = False
        data = integ.get_time_stream()
        assert np.all(np.isnan(data[:3]))
        assert np.allclose(data[3:], integ.frames.data[3:])

        # weighted: fill value is zero, weights are returned
        data, weights = integ.get_time_stream(weighted=True)
        assert np.allclose(data[:3], 0)
        assert np.allclose(data[3:], integ.frames.data[3:])
        assert np.allclose(weights[:3], 0)
        assert np.allclose(weights[3:], 1)

    @pytest.mark.parametrize('method', ['neighbors', 'absolute',
                                        'gradual', 'multires',
                                        'features'])
    def test_despike_methods(self, populated_integration, method):
        integ = populated_integration
        integ.configuration.unlock('despike.method')
        integ.configuration.set_option('despike.method', method)
        integ.configuration.set_option('despike.blocks', True)
        integ.despike()
        # none of the methods have any effect with default params
        assert np.sum(integ.frames.is_flagged()) == 0

    def test_despike_neighbouring(self, capsys, populated_integration):
        integ = populated_integration.copy()

        with set_log_level('DEBUG'):
            # flag some individual frame spikes
            integ.despike_neighbouring(0.1, 1)
            assert '4002 live channel frames (3.01%) ' \
                   'flagged as spikes' in capsys.readouterr().out

            # make delta too large
            integ = populated_integration.copy()
            integ.despike_neighbouring(0.1, 1101)
            assert 'delta (1101) too large' in capsys.readouterr().err

    def test_flag_spiky_frames(self, capsys, populated_integration):
        integ = populated_integration
        flag = integ.frames.flagspace.flags.FLAG_SPIKY

        # flag some spikes
        integ.despike_neighbouring(0.1, 1)

        # with spike limit not passed, flags none
        integ.flag_spiky_frames()
        assert np.sum(integ.frames.is_flagged(flag)) == 0

        # with minimal limit passed, flags some
        integ.flag_spiky_frames(frame_spikes=1)
        assert np.sum(integ.frames.is_flagged(flag)) == 176

    def test_flag_spiky_channels(self, capsys, populated_integration):
        integ = populated_integration
        flag = integ.channels.data.flagspace.flags.SPIKY

        # flag some spikes
        integ.despike_neighbouring(0.1, 1)

        # with count limit not passed, flags none
        integ.flag_spiky_channels()
        assert np.sum(integ.channels.data.is_flagged(flag)) == 0

        # with minimal limit passed, flags all
        integ.flag_spiky_channels(flag_fraction=0, flag_count=1)
        assert np.sum(integ.channels.data.is_flagged(flag)) == 121

    def test_detector_stage(self, capsys, populated_integration):
        integ = populated_integration

        # before detector stage, readout is no-op
        with set_log_level('DEBUG'):
            integ.readout_stage()
        assert 'Unstaging detector' not in capsys.readouterr().out
        assert not integ.is_detector_stage

        with set_log_level('DEBUG'):
            integ.detector_stage()
        assert 'Staging detector' in capsys.readouterr().out
        assert integ.is_detector_stage

        # second time is no-op
        with set_log_level('DEBUG'):
            integ.detector_stage()
        assert 'Staging detector' not in capsys.readouterr().out
        assert integ.is_detector_stage

        # after detector stage, readout unstages
        with set_log_level('DEBUG'):
            integ.readout_stage()
        assert 'Unstaging detector' in capsys.readouterr().out
        assert not integ.is_detector_stage

    def test_get_signal(self, populated_integration):
        integ = populated_integration
        mode = CorrelatedMode(name='test')

        # missing mode/signal
        result = integ.get_signal(mode)
        assert result is None

        # add a signal
        sig = integ.get_acceleration_signal('x', mode=mode)
        integ.add_signal(sig)

        result = integ.get_signal(mode)
        assert result is sig

        # add response mode: signal is derived
        mode = PositionResponse(name='position')
        mode.set_direction('x')
        mode.set_type('Scanning')
        result = integ.get_signal(mode)
        assert isinstance(result, Signal)

    def test_get_coupling_gains(self, populated_integration):
        integ = populated_integration

        mode = CorrelatedMode(name='test')
        mode.set_channel_group(integ.channels.create_channel_group())
        sig = integ.get_acceleration_signal('x', mode=mode)

        # all 1 for all channels, since all are included
        result = integ.get_coupling_gains(sig)
        assert result.size == integ.channels.size
        assert np.allclose(result, 1)

    def test_shift_frames(self, capsys, populated_integration):
        integ = populated_integration

        # no op for non finite value
        with set_log_level('DEBUG'):
            integ.shift_frames(np.inf)
        assert 'Shifting data' not in capsys.readouterr().out

        # frames if not quantity
        with set_log_level('DEBUG'):
            integ.shift_frames(10.2)
        assert 'Shifting data by 10 frames' in capsys.readouterr().out

        # calculate frames if quantity
        with set_log_level('DEBUG'):
            integ.shift_frames(10.2 * units.s)
        assert 'Shifting data by 102 frames' in capsys.readouterr().out

    def test_get_table_entry(self, populated_integration):
        integ = populated_integration

        # gain, nefd, zenith_tau from attributes
        assert np.allclose(integ.get_table_entry('scale'), np.ones(121))

        assert np.isnan(integ.get_table_entry('NEFD'))
        expected = 0.3162277
        integ.calculate_source_nefd()
        assert np.allclose(integ.get_table_entry('NEFD'), expected)

        assert np.allclose(integ.get_table_entry('zenithtau'), 0)

        # tau.spec from get_tau, via config
        integ.configuration.set_option('tau.model.a', 1.0)
        integ.configuration.set_option('tau.model.b', 0.0)
        integ.configuration.set_option('tau.example.a', 1.0)
        integ.configuration.set_option('tau.example.b', 0.0)
        assert np.allclose(integ.get_table_entry('tau.model'), 0)

        # speeds from avg scan speed
        assert np.isnan(integ.get_table_entry('scanspeed'))
        assert np.isnan(integ.get_table_entry('rmsspeed'))

        # hipass filter time scale
        assert np.isinf(integ.get_table_entry('hipass'))

        # chopper info
        assert integ.get_table_entry('chopfreq') is None
        integ.chopper = Chopper()
        integ.chopper.frequency = 1.0
        assert integ.get_table_entry('chopfreq') == 1.0 * units.Hz

        # missing value
        assert integ.get_table_entry('test') is None

    def test_setup_filters(self, capsys, populated_integration):
        integ = populated_integration

        # set filter ordering to include 2 real and 1 not
        integ.configuration.set_option('filter.ordering',
                                       ['motion', 'kill', 'test'])
        integ.setup_filters()
        assert len(integ.filter.filters) == 2
        assert 'No filter for test' in capsys.readouterr().err

    def test_remove_dc_offsets(self, capsys, populated_integration):
        integ = populated_integration
        capsys.readouterr()

        # no op if level not configured
        del integ.configuration['level']
        with set_log_level('DEBUG'):
            integ.remove_dc_offsets()
        assert 'Removing DC offsets' not in capsys.readouterr().out
        data = integ.frames.data.copy()

        # configure
        integ.configuration.set_option('level', True)
        with set_log_level('DEBUG'):
            integ.remove_dc_offsets()
        assert 'Removing DC offsets' in capsys.readouterr().out
        assert not np.allclose(integ.frames.data, data)

    def test_remove_drifts(self, capsys, mocker, populated_integration):
        integ = populated_integration

        with set_log_level('DEBUG'):
            ok = integ.remove_drifts()
        capt = capsys.readouterr()
        assert 'Removing channel drifts' in capt.out
        assert 'Total drift inconsistencies = 0' in capt.out
        assert ok

        # mock an inconsistency
        mocker.patch('sofia_redux.scan.integration.'
                     'integration_numba_functions.'
                     'apply_drifts_to_channel_data',
                     return_value=(1, 1))
        with set_log_level('DEBUG'):
            ok = integ.remove_drifts()
        capt = capsys.readouterr()
        assert 'Removing channel drifts' in capt.out
        assert 'Total drift inconsistencies = 1' in capt.out
        assert ok

    def test_set_tau(self, populated_integration):
        integ = populated_integration

        # error if nothing specified
        with pytest.raises(ValueError) as err:
            integ.set_tau()
        assert 'Configuration does not contain a ' \
               'tau specification' in str(err)

        # set value directly
        integ.set_tau(value=6.0)
        assert integ.zenith_tau == 6.0
        integ.set_tau(spec=7.0)
        assert integ.zenith_tau == 7.0
        integ.configuration.set_option('tau', 8.0)
        integ.set_tau()
        assert integ.zenith_tau == 8.0

        # configure spec
        integ.configuration.set_option('tau.model', 1.0)
        integ.configuration.set_option('tau.model.a', 1.0)
        integ.configuration.set_option('tau.model.b', 0.0)
        integ.configuration.set_option('tau.example.a', 1.0)
        integ.configuration.set_option('tau.example.b', 0.0)
        integ.set_tau(spec='model')
        assert integ.zenith_tau == 1.0

        del integ.configuration['tau.model']
        with pytest.raises(ValueError) as err:
            integ.set_tau(spec='model')
        assert 'does not contain a tau value' in str(err)

    def test_set_tau_value(self, populated_integration):
        integ = populated_integration

        # ground based
        integ.set_tau_value(6.0)
        assert integ.zenith_tau == 6.0

        # not ground based: sets transmission instead
        integ.info.astrometry.ground_based = False
        integ.set_tau_value(6.0)
        assert np.allclose(integ.frames.transmission, np.exp(-6))

    def test_get_tau_coefficients(self, populated_integration):
        integ = populated_integration

        with pytest.raises(ValueError) as err:
            integ.get_tau_coefficients('test')
        assert 'Tau tau.test has no scaling' in str(err)

        integ.configuration.set_option('tau.test.a', 1.0)
        with pytest.raises(ValueError) as err:
            integ.get_tau_coefficients('test')
        assert 'Tau tau.test has no offset' in str(err)

        integ.configuration.set_option('tau.test.b', 0.0)
        a, b = integ.get_tau_coefficients('test')
        assert a == 1.0
        assert b == 0.0

    def test_set_scaling(self, capsys, populated_integration):
        integ = populated_integration

        with set_log_level('DEBUG'):
            integ.set_scaling(np.nan)
        assert 'Applying scaling' not in capsys.readouterr().out
        assert integ.gain == 1

        with set_log_level('DEBUG'):
            integ.set_scaling(2.0)
        assert 'Applying scaling' in capsys.readouterr().out
        assert integ.gain == 0.5

    def test_get_default_scaling_factor(self, populated_integration):
        integ = populated_integration
        config = integ.configuration
        config.parse_key_value('scale.value', '1.5')
        config.parse_key_value('scale.grid', '2.0')
        config.parse_key_value('grid', '4.0')
        assert integ.get_default_scaling_factor() == 6
        config.parse_key_value('grid', '2.0')
        assert integ.get_default_scaling_factor() == 1.5
        del config['scale.grid']
        assert integ.get_default_scaling_factor() == 1.5
        del config['scale']
        assert integ.get_default_scaling_factor() == 1

    def test_reindex_channels(self, populated_integration):
        integ = populated_integration

        # add a signal to reindex
        mode = CorrelatedMode(name='test')
        mode.set_channel_group(integ.channels.create_channel_group())
        sig = integ.get_acceleration_signal('x', mode=mode)
        integ.add_signal(sig)

        integ.reindex_channels()
        assert mode.channel_group.data is integ.channels.data

    def test_bootstrap_weights(self, capsys, populated_integration):
        integ = populated_integration

        del integ.configuration['pixeldata']

        # no op
        integ.configuration.set_option('weighting', False)
        with set_log_level('DEBUG'):
            integ.bootstrap_weights()
        assert 'Bootstrapping' not in capsys.readouterr().out

        integ.configuration.set_option('weighting', True)
        integ.configuration.set_option('uniform', True)
        with set_log_level('DEBUG'):
            integ.bootstrap_weights()
        assert 'Bootstrapping' not in capsys.readouterr().out

        integ.configuration.set_option('uniform', False)
        with set_log_level('DEBUG'):
            integ.bootstrap_weights()
        assert 'Bootstrapping' in capsys.readouterr().out

    def test_decorrelate(self, capsys, populated_integration):
        populated_integration.validate()
        integ = populated_integration.copy()

        # missing modality
        assert integ.decorrelate('test') is False
        assert integ.update_gains('test') is False

        # existing modality
        integ = populated_integration.copy()
        with set_log_level('DEBUG'):
            assert integ.decorrelate('sky') is True
        assert 'robust' not in capsys.readouterr().out

        integ = populated_integration.copy()
        with set_log_level('DEBUG'):
            assert integ.decorrelate('sky', robust=True) is True
        assert 'robust' in capsys.readouterr().out

        # turn trigger off
        integ = populated_integration.copy()
        integ.configuration.set_option('correlated.sky.trigger', '0')
        assert integ.decorrelate('sky') is False
        assert integ.update_gains('sky') is False

        # set frame resolution
        integ = populated_integration.copy()
        integ.configuration.set_option('correlated.sky.resolution', 2)
        with set_log_level('DEBUG'):
            assert integ.decorrelate('sky') is True
        assert 'resolution = 32' in capsys.readouterr().out

        # no modalities
        integ = populated_integration.copy()
        integ.channels.modalities = None
        assert integ.decorrelate('sky') is False
        assert integ.update_gains('sky') is False

    def test_update_gains(self, mocker, populated_integration):
        integ = populated_integration.copy()

        # stops short if not gains
        integ.configuration.set_option('gains', False)
        assert integ.update_gains('sky') is True

        # if robust is None, gets from config
        integ = populated_integration.copy()
        assert integ.update_gains('sky', robust=None) is True

        # update all gains
        mocker.patch.object(integ.channels.modalities['sky'],
                            'update_all_gains', return_value=True)
        assert integ.update_gains('sky') is True

    def test_merge(self, populated_integration):
        integ1 = populated_integration.copy()
        assert integ1.size == 1100
        integ2 = populated_integration.copy()
        integ1.merge(integ2)
        assert integ1.size == 2200

    def test_get_thread_count(self, populated_integration):
        integ = populated_integration
        assert integ.get_thread_count() is None
        integ.parallelism = 4
        assert integ.get_thread_count() == 4

    def test_perform(self, mocker, populated_integration):
        integ = populated_integration
        integ.validate()
        integ.comments = None

        # smoke test all task branches

        expected = ['J', 'tW', '0:45', ' ']
        assert integ.perform('dejump') is True
        assert integ.comments == expected

        expected.extend(['O', ' ', ' '])
        assert integ.perform('offsets') is True
        assert integ.comments == expected

        expected.extend(['D(512)', ' ', ' '])
        assert integ.perform('drifts') is True
        assert integ.comments == expected

        mocker.patch.object(integ, 'remove_drifts', return_value=False)
        assert integ.perform('drifts') is False
        assert integ.comments == expected

        expected.extend(['Cs', ' '])
        assert integ.perform('correlated.sky') is True
        assert integ.comments == expected

        # unrecognized correlation
        assert integ.perform('correlated.test') is False
        assert integ.comments == expected

        expected.extend(['W', '9.93e-06 beam / Jy', '115', ' '])
        assert integ.perform('weighting') is True
        assert integ.comments == expected

        expected.extend(['tW', '(128)', ' '])
        assert integ.perform('weighting.frames') is True
        assert integ.comments == expected

        expected.extend(['dN', ' '])
        assert integ.perform('despike') is True
        assert integ.comments == expected

        # filter is no op for default config
        assert integ.perform('filter') is False
        assert integ.comments == expected

        integ.filter = None
        assert integ.perform('filter') is False
        assert integ.comments == expected

        # unknown task just returns False
        assert integ.perform('test') is False
        assert integ.comments == expected

    def test_get_fits_data(self, populated_integration):
        integ = populated_integration
        integ.setup_filters()

        result = integ.get_fits_data()
        assert isinstance(result, astropy.table.Table)
        assert len(result) == 1
        assert len(result.columns) == 7

        updated = integ.add_details(result)
        assert len(updated) == 121
        assert len(updated.columns) == 14

        # empty table for add_details
        details = integ.add_details()
        assert isinstance(details, astropy.table.Table)
        assert len(details) == 121
        assert len(details.columns) == 7

    def test_jackknife(self, mocker, capsys, populated_integration):
        integ = populated_integration

        # mock random to always return < 0.5
        mocker.patch('sofia_redux.scan.integration.integration.'
                     'np.random.random', return_value=np.full(1100, 0.1))

        # no config: nothing inverted
        integ.jackknife()
        assert integ.gain == 1
        assert np.sum(integ.frames.sign < 0) == 0

        # invert gain and all frames
        integ.configuration.set_option('jackknife', True)
        integ.configuration.set_option('jackknife.frames', True)
        integ.jackknife()
        assert integ.gain == -1
        assert np.sum(integ.frames.sign < 0) == 1100

    def test_get_floats(self, populated_integration):
        integ = populated_integration

        # empty array, nearest power of 2 above size (1100)
        arr = integ.get_floats()
        assert arr.shape == (2048,)

    def test_get_correlated_signal_class(self):
        cls = Integration.get_correlated_signal_class()
        assert cls is CorrelatedSignal

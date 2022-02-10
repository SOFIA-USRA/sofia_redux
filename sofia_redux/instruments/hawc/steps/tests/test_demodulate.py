# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data


class TestDemodulate(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)

        # unprepared data fails
        step = StepDemodulate()
        with pytest.raises(ValueError):
            step(inp, boxfilter=0)

        # prepared data passes
        out = step(prep)
        assert isinstance(out, DataFits)

    def test_read_beam(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepDemodulate()
        step.datain = df
        step.runstart(df, {})

        expected = (19.0, 'E')

        # test defaults
        result = step.read_beam()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_beam()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_beam()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

    def test_readdata(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        braw = StepDemodulate()

        # raw data fails
        braw.datain = inp
        with pytest.raises(ValueError):
            braw.readdata()
        assert "Column 'R array' not found"

        # prepared data succeeds
        prep = StepPrepare()(inp)
        braw.datain = prep
        braw.readdata()
        shape = prep.table['R array'][0].shape
        assert braw.praw['nrow'] == shape[0]
        assert braw.praw['ncol'] == shape[1]

        # test makedemod after data is loaded
        braw.makedemod(braw.praw['nsamp'])
        assert braw.pdemod['nrow'] == shape[0]
        assert braw.pdemod['ncol'] == shape[1]

    def test_tracking(self, tmpdir, capsys):
        # test data
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        step = StepDemodulate()

        # run with beam tolerance tracking
        # boxfilter is 0 for faster testing
        out = step(prep, track_tol='beam', boxfilter=0)
        capt = capsys.readouterr()
        assert 'Removing bad samples, tracking issues - ' \
               'using beam size' in capt.out
        assert 'GOOD' in out.getheadval('TRCKSTAT')

        # add some nan values to trackerr3/4
        nantrack = prep.copy()
        nantrack.table['TrackErrAoi3'][0:4] = np.nan
        nantrack.table['TrackErrAoi4'][0:4] = np.nan
        out = step(nantrack, track_tol='beam', boxfilter=0)
        capt = capsys.readouterr()
        assert 'TrackErrAoi3 signal contains NaNs' in capt.err
        assert 'TrackErrAoi4 signal contains NaNs' in capt.err
        assert 'GOOD' in out.getheadval('TRCKSTAT')

        # remove a column entirely
        notrack = prep.copy()
        notrack.tabledelcol('TrackErrAoi3')
        step = StepDemodulate()
        out = step(notrack, track_tol='beam', boxfilter=0)
        capt = capsys.readouterr()
        assert 'TrackErrAoi3 and 4 tables not found' in capt.err
        assert 'NONE' in out.getheadval('TRCKSTAT')

        # make too much data bad
        badtrack = prep.copy()
        badtrack.table['TrackErrAoi3'] = 100.0
        step = StepDemodulate()
        out = step(badtrack, track_tol='beam', boxfilter=0)
        capt = capsys.readouterr()
        assert 'Tracking status is BAD' in capt.err
        assert 'BAD' in out.getheadval('TRCKSTAT')

        # run with centroidexp
        out = step(prep, track_tol='centroidexp', boxfilter=0)
        capt = capsys.readouterr()
        assert 'Removing bad samples, tracking issues - ' \
               'using centroidexp' in capt.out
        assert 'GOOD' in out.getheadval('TRCKSTAT')

        # remove the column
        notrack = prep.copy()
        notrack.tabledelcol('CentroidExpMsec')
        step = StepDemodulate()
        out = step(notrack, track_tol='centroidexp', boxfilter=0)
        capt = capsys.readouterr()
        assert 'CentroidExpMsec not found' in capt.err
        assert 'NONE' in out.getheadval('TRCKSTAT')

        # make too much data bad
        badtrack = prep.copy()
        badtrack.table['CentroidExpMsec'] = 1.0
        step = StepDemodulate()
        out = step(badtrack, track_tol='centroidexp', boxfilter=0)
        capt = capsys.readouterr()
        assert 'Tracking status is BAD' in capt.err
        assert 'BAD' in out.getheadval('TRCKSTAT')

        # run with a specific tolerance
        out = step(prep, track_tol=2.0, boxfilter=0)
        capt = capsys.readouterr()
        assert 'Removing bad samples, tracking issues - ' \
               'using track_tol' in capt.out
        assert 'GOOD' in out.getheadval('TRCKSTAT')

        # try to run with a bad value
        with pytest.raises(ValueError):
            step(prep, track_tol='badval', boxfilter=0)
        capt = capsys.readouterr()
        assert 'track_tol value is undefined' in capt.err

        # make the tolerance too low
        out = step(badtrack, track_tol=1e-3, boxfilter=0)
        capt = capsys.readouterr()
        assert 'Tracking status is BAD' in capt.err
        assert 'BAD' in out.getheadval('TRCKSTAT')

    def test_track_extra(self, tmpdir, capsys):
        # test data
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        step = StepDemodulate()

        # run with track_extra to remove a couple more samples
        out = step(prep, track_tol='beam', track_extra=[1, 1], boxfilter=0)
        capt = capsys.readouterr()
        assert 'Number of good samples after EXTRA removal' in capt.out
        assert 'GOOD' in out.getheadval('TRCKSTAT')

        # run again but remove too many
        out = step(prep, track_tol='beam', track_extra=[50, 50], boxfilter=0)
        capt = capsys.readouterr()
        assert 'Number of good samples after EXTRA removal' in capt.out
        assert 'BAD' in out.getheadval('TRCKSTAT')

    def test_run_errors(self, capsys):
        # minimal test data
        # run will fail with attribute error -- no table
        df = DataFits()
        df.imageset(np.zeros((10, 10)))
        df.header['SMPLFREQ'] = 203.25
        df.header['CHPFREQ'] = 10.2
        df.header['CHPONFPA'] = False
        df.header['NODDING'] = True
        df.header['NHWP'] = 1
        df.config = None

        step = StepDemodulate()
        step.datain = df
        step.runstart(df, {})

        # missing user frequency in parameters
        with pytest.raises(AttributeError):
            step.run()
        capt = capsys.readouterr()
        assert 'Default user frequency: 10.2' in capt.out

        # invalid chop frequency
        df.header['CHPFREQ'] = -9999
        with pytest.raises(ValueError):
            step.run()
        capt = capsys.readouterr()
        assert 'Invalid chop frequency: -9999' in capt.err

        # if mode is abs, use user frequency
        step.runstart(df, {'l0method': 'ABS'})
        with pytest.raises(AttributeError):
            step.run()
        capt = capsys.readouterr()
        assert 'Invalid chop frequency' in capt.out
        assert 'Using user frequency' in capt.out

    def test_checkhwp(self, tmpdir, capsys):
        # fewer frames needed for this check
        hdul = pol_raw_data(nframe=20)

        # set an unexpected NHWP
        hdul[0].header['NHWP'] = 2

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        step = StepDemodulate()

        # with checkhwp, just warns
        out = step(prep, checkhwp=True, boxfilter=0)
        capt = capsys.readouterr()
        assert 'Expected 2 HWP angles; found 3' in capt.err
        assert out.getheadval('NHWP') == 2

        # without, it will set the value
        out = step(prep, checkhwp=False, boxfilter=0)
        capt = capsys.readouterr()
        assert 'Assigning NHWP keyword with the actual ' \
               'number of HWP angles (3)' in capt.err
        assert out.getheadval('NHWP') == 3

    def test_phase(self, tmpdir, capsys):
        hdul = pol_raw_data(nframe=20)
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        step = StepDemodulate()

        # run with bad phasefile
        out1 = step(prep, phasefile='bad.fits')
        capt = capsys.readouterr()
        assert 'Could not set phasefile <bad.fits>' in capt.err
        assert 'Applying a phase correction of <0.0>' in capt.out
        assert 'Doing Chop Phase Correction' in capt.out
        assert not np.allclose(out1.table['Phase Corr'], 0)

        # run with no chopphase requested
        out2 = step(prep, phasefile=0.0, chopphase=False)
        capt = capsys.readouterr()
        assert 'Doing Chop Phase Correction' not in capt.out
        assert np.allclose(out2.table['Phase Corr'], 0)

    def test_tags(self):
        # check chop/nod/phase tags with minimal data
        nframe = 5
        chopsamp = 1
        chopstate = np.full(nframe, 1)
        nodstate = np.full(nframe, 1)
        hwpstate = np.full(nframe, 1)

        def reset():
            nstep = StepDemodulate()
            nstep.praw = {'Nod Index': np.full(nframe, -1),
                          'HWP Index': np.full(nframe, 0)}
            return nstep

        step = reset()
        nchop, choptag, phasetag = step.make_chop_phase_tags(
            chopstate, chopsamp, hwpstate, nodstate)
        # 4 chops per nod
        assert nchop == 4
        assert np.all(choptag == np.arange(5))
        assert np.all(phasetag == 0)
        # nod index all zero except last value
        assert np.all(step.praw['Nod Index'][:-1] == 0)
        assert step.praw['Nod Index'][-1] == -1
        # hwp index unmodified
        assert np.all(step.praw['HWP Index'][-1] == 0)

        # zero in chopstate => first chop is moved later
        step = reset()
        chopstate[0] = 0
        nchop, choptag, phasetag = step.make_chop_phase_tags(
            chopstate, chopsamp, hwpstate, nodstate)
        # 1 less chop
        assert nchop == 3
        assert np.all(choptag == np.array([-1, 0, 1, 2, 3]))
        assert np.all(phasetag == np.array([-1, 0, 0, 0, 0]))
        assert np.all(step.praw['Nod Index'] == np.array([0, 0, 0, 0, -1]))

        # zero in nod state instead --
        # chop unaffected, first nod value is -1
        step = reset()
        chopstate[0] = 1
        nodstate[0] = 0
        nchop, choptag, phasetag = step.make_chop_phase_tags(
            chopstate, chopsamp, hwpstate, nodstate)
        assert nchop == 4
        assert np.all(choptag == np.arange(5))
        assert np.all(phasetag == 0)
        assert np.all(step.praw['Nod Index'] == np.array([-1, 0, 0, 0, -1]))

        # add an hwp transition
        step = reset()
        nodstate[2] = 0
        hwpstate[2] = 0
        hwpstate[-1] = 0
        nchop, choptag, phasetag = step.make_chop_phase_tags(
            chopstate, chopsamp, hwpstate, nodstate)
        assert nchop == 4
        assert np.all(choptag == np.arange(5))
        assert np.all(phasetag == 0)
        assert np.all(step.praw['Nod Index'] == np.array([-1, 0, -1, 0, -1]))

        # add off-nods
        step = reset()
        chopstate = np.array([0, 1, 1, 1, 1])
        nodstate = np.array([1, 1, 0, -1, -1])
        hwpstate = np.array([1, 1, 1, 1, 1])
        nchop, choptag, phasetag = step.make_chop_phase_tags(
            chopstate, chopsamp, hwpstate, nodstate)
        assert nchop == 3
        assert np.all(choptag == np.array([-1, 0, 1, 2, 3]))
        assert np.all(phasetag == np.array([-1, 0, 0, 0, 0]))
        assert np.all(step.praw['Nod Index'] == np.array([0, 0, -1, 1, -1]))

        # larger chop sample => incomplete chop
        step = reset()
        step.praw['Nod Index'] = np.full(9, -1)
        chopstate = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])
        nodstate = np.array([1, 1, 1, 0, -1, -1, -1, 0, 1])
        hwpstate = np.full(9, 1)
        nchop, choptag, phasetag = step.make_chop_phase_tags(
            chopstate, 3, hwpstate, nodstate)
        log.info(nchop)
        log.info(choptag)
        log.info(phasetag)
        log.info(step.praw['Nod Index'])
        assert nchop == 2
        assert np.all(choptag == np.array([-1, 0, 0, 0, 1, 1, 1, -1, -1]))
        assert np.all(phasetag == np.array([-1, 0, 1, 2, 0, 1, 2, -1, -1]))
        assert np.all(step.praw['Nod Index']
                      == np.array([0, 0, 0, -1, 1, 1, 1, -1, -1]))

    def test_zero_chop(self, tmpdir, capsys, mocker):
        hdul = pol_raw_data(nframe=20)
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        step = StepDemodulate()

        # mock tag function to return zero chops
        mocker.patch.object(StepDemodulate, 'make_chop_phase_tags',
                            return_value=(0, np.array([]), np.array([])))
        with pytest.raises(ValueError):
            step(prep)
        capt = capsys.readouterr()
        assert 'Invalid number of chops (0)' in capt.err

    def test_l0method(self, tmpdir, capsys):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        inp = DataFits(ffile)
        prep = StepPrepare()(inp)

        # default - real
        step = StepDemodulate()
        out1 = step(prep.copy(), l0method='RE')

        # absolute
        step = StepDemodulate()
        out2 = step(prep.copy(), l0method='ABS')

        # imaginary
        step = StepDemodulate()
        out3 = step(prep.copy(), l0method='IM')

        # invalid
        step = StepDemodulate()
        out4 = step(prep.copy(), l0method='BAD')
        capt = capsys.readouterr()
        assert 'unknown l0method=BAD' in capt.err

        # invalid uses real method
        assert np.allclose(out1.table['R Array'], out4.table['R Array'])

        # abs value should be higher
        abs_sum = np.nansum(out2.table['R Array'])
        assert np.nansum(out1.table['R Array']) < abs_sum
        assert np.nansum(out3.table['R Array']) < abs_sum

    def test_keywords(self, tmpdir):
        hdul = pol_raw_data(nframe=20)
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)

        # default
        step = StepDemodulate()
        out1 = step(prep.copy())
        assert 'ASSC_AOR' in out1.header
        assert 'ASSC_MSN' in out1.header

        # chop on FPA - double exposure time
        prep.setheadval('CHPONFPA', True)
        # remove aor_id, missn-id -- assc values no longer there
        del prep.header['AOR_ID']
        del prep.header['MISSN-ID']
        step = StepDemodulate()
        out2 = step(prep.copy())
        assert np.allclose(out2.getheadval('EXPTIME'),
                           2 * out1.getheadval('EXPTIME'))
        assert 'ASSC_AOR' not in out2.header
        assert 'ASSC_MSN' not in out2.header

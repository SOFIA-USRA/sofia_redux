# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments import hawc
from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepscanmap import StepScanMap
from sofia_redux.instruments.hawc.steps.stepscanmap import Reduction as SMR
from sofia_redux.instruments.hawc.steps.stepscanmappol import StepScanMapPol
from sofia_redux.instruments.hawc.steps.stepscanmappol import Reduction as SMPR
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, scan_raw_data


@pytest.fixture(scope='function')
def test_options():
    # options for faster test reductions
    return 'rounds=1'


@pytest.mark.timeout(0)
class TestScanMap(DRPTestCase):

    def test_mimo(self, tmpdir, test_options):
        hdul = scan_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            step = StepScanMap()

            out = step([df], options=test_options)
            assert isinstance(out, list)
            assert len(out) == 1
            assert isinstance(out[0], DataFits)

    def test_bad_file(self, tmpdir, capsys, test_options):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)

        # also test badly formatted options: they should be ignored
        opt = test_options + ' q'

        with tmpdir.as_cwd():
            step = StepScanMap()
            with pytest.raises(ValueError):
                step([df], options=opt)
            capt = capsys.readouterr()
            assert 'No scans to reduce' in capt.err

    def test_scan_errors(self, tmpdir, capsys, test_options):
        # if scanmap produces too many output files in scan
        # mode, it will raise an error. This can happen
        # if INSTCFG is set incorrectly for scanpol data,
        # for example.

        # make scan pol data
        angle = [5.0, 50.0]
        inp = []
        for i in range(2):
            hdul = scan_raw_data()
            hdul[2].data['hwpCounts'] = angle[i] * 4

            ffile = str(tmpdir.join('test{}.fits'.format(i)))
            hdul.writeto(ffile, overwrite=True)
            hdul.close()
            inp.append(DataFits(ffile))

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            # run scanmap (not scanmappol)
            step = StepScanMap()
            with pytest.raises(ValueError):
                step(inp, options=test_options)
            capt = capsys.readouterr()
            assert 'Unexpected output' in capt.err
            assert 'Check INSTCFG' in capt.err

    def test_scanpol(self, tmpdir, capsys, test_options):
        # make scan pol data
        angle = [5.0, 50.0, 27.0, 72.0]
        inp = []
        exp = []
        for i in range(4):
            hdul = scan_raw_data()
            hdul[2].data['hwpCounts'] = angle[i] * 4

            ffile = str(tmpdir.join('test{}.fits'.format(i)))
            hdul.writeto(ffile, overwrite=True)
            hdul.close()
            exp.append(hdul[0].header['EXPTIME'])

            inp.append(DataFits(ffile))

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            step = StepScanMapPol()
            out = step(inp, options=test_options)
            names = []
            for i in range(4):
                names.append('DATA R HWP%d' % i)
                names.append('DATA T HWP%d' % i)
                names.append('ERROR R HWP%d' % i)
                names.append('ERROR T HWP%d' % i)
                names.append('EXPOSURE R HWP%d' % i)
                names.append('EXPOSURE T HWP%d' % i)
            for name in names:
                assert name in out[0].imgnames

            # exposure time in primary header should sum over input
            assert out[0].getheadval('EXPTIME') == np.sum(exp)

            # check that if too few angles are passed, they
            # are dropped from the reduction
            out = step([inp[0]], options=test_options)
            assert len(out) == 0
            capt = capsys.readouterr()
            assert 'Dropping files from reduction' in capt.err

    def test_scanpol_errors(self, capsys):
        # make minimal test cases, as if from SCANMAP
        angle = [5.0, 50.0, 27.0, 72.0]
        dataout = []
        for i in range(4):
            df = DataFits()
            df.imageset(np.zeros((10, 10)), imagename='PRIMARY IMAGE')
            df.imageset(np.zeros((10, 10)), imagename='NOISE')
            df.imageset(np.zeros((10, 10)), imagename='EXPOSURE')

            df.setheadval('TELVPA', 270.0)
            df.setheadval('HWPINIT', angle[i])
            df.setheadval('CDELT2', 1.0)
            df.setheadval('EXPTIME', 100.0)

            df.setheadval('SUBARRAY', 'R0')
            dataout.append(df)

            tdf = df.copy()
            tdf.setheadval('SUBARRAY', 'T0')
            dataout.append(tdf)

        # minimal data input with standard SOFIA keys for merging
        df = DataFits()
        df.setheadval('OBS_ID', 'UNKNOWN')
        df.setheadval('TELVPA', 270.0)

        step = StepScanMapPol()
        step.datain = [df]

        # working set
        basehead = df.header
        assembled = step.assemble_scanpol(dataout, basehead)
        assert isinstance(assembled, DataFits)
        # angle is 180 off from tel_vpa
        assert int(assembled.getheadval('VPOS_ANG')) == 90

        # missing R0
        bad = dataout[1:]
        with pytest.raises(ValueError):
            step.assemble_scanpol(bad, basehead)
        capt = capsys.readouterr()
        assert 'Missing R0' in capt.err

        # missing T0
        bad = [dataout[0]] + dataout[2:]
        with pytest.raises(ValueError):
            step.assemble_scanpol(bad, basehead)
        capt = capsys.readouterr()
        assert 'Missing T0' in capt.err

        # too many R0s
        bad = dataout.copy()
        bad[1] = dataout[1].copy()
        bad[1].setheadval('SUBARRAY', 'R0')
        with pytest.raises(ValueError):
            step.assemble_scanpol(bad, basehead)
        capt = capsys.readouterr()
        assert 'Too many subarray=R0' in capt.err

        # wrong number of angles
        bad = dataout.copy()
        bad.append(dataout[0].copy())
        bad[-1].setheadval('HWPINIT', 81.0)
        bad.append(dataout[1].copy())
        bad[-1].setheadval('HWPINIT', 81.0)
        with pytest.raises(ValueError):
            step.assemble_scanpol(bad, basehead)
        capt = capsys.readouterr()
        assert 'Must be exactly 4 HWP angles' in capt.err

        # missing keys
        bad = dataout.copy()
        bad[-1].delheadval('HWPINIT')
        with pytest.raises(ValueError):
            step.assemble_scanpol(bad, basehead)
        capt = capsys.readouterr()
        assert 'missing required keywords' in capt.err

    def test_smap_options(self, tmpdir, mocker, capsys):
        df = DataFits()
        df.imageset(np.zeros((10, 10)), imagename='PRIMARY IMAGE')
        df.imageset(np.zeros((10, 10)), imagename='NOISE')
        df.imageset(np.zeros((10, 10)), imagename='EXPOSURE')

        df.setheadval('TELVPA', 270.0)
        df.setheadval('BUNIT', 'counts')

        fname = str(tmpdir.join('testdata.fits'))
        df.filename = fname
        df.save()

        # mock scanmap to return faster
        mocker.patch.object(SMR, 'run', return_value=None)

        step = StepScanMap()

        # default options --
        # no output saved, so will fail
        with pytest.raises(ValueError):
            step([df])
        capt = capsys.readouterr()
        assert 'No output created' in capt.err
        # options are not in scanmap command string
        assert 'deep=True' not in capt.out
        assert 'faint=True' not in capt.out
        assert 'extended=True' not in capt.out
        # df.filename is in command
        assert os.path.basename(fname) in capt.out

        # deep option
        with pytest.raises(ValueError):
            step([df], deep=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'deep=True' in capt.out

        # faint option
        with pytest.raises(ValueError):
            step([df], faint=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'faint=True' in capt.out

        # extended option
        with pytest.raises(ValueError):
            step([df], extended=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'extended=True' in capt.out

        # if df.filename is not on disk, it will check for
        # df.rawname
        df.rawname = df.filename
        df.filename = 'badfile.fits'
        with pytest.raises(ValueError):
            step([df])
        step.arglist = {}
        capt = capsys.readouterr()
        assert os.path.basename(fname) in capt.out
        assert 'badfile.fits' not in capt.out

        # if noout is set, it won't fail
        step([df, df], noout=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert os.path.basename(fname) in capt.out
        assert len(step.dataout) == 2

    def test_merge_head(self, capsys):
        hdul = scan_raw_data()
        full_header = hdul[0].header
        scanmap_header = fits.Header()

        step = StepScanMap()
        df = DataFits()
        df.header = full_header
        step.dataout = df.copy()
        step.datain = [df, df, df]

        # add a comment from the scanmap header: not added to output
        scanmap_header['COMMENT'] = 'Test comment'
        step.merge_scan_hdr(step.dataout, scanmap_header, step.datain)
        assert 'COMMENT' not in step.dataout.header

        # assc_aor and assc_msn should be set from aor_id and missn_id
        assert step.dataout.header['ASSC_AOR'] == df.header['AOR_ID']
        assert step.dataout.header['ASSC_MSN'] == df.header['MISSN-ID']

        # attempt to add a bad value from the scanmap header
        class MockCard(object):
            def __init__(self):
                self.keyword = 'BADKEY'
                self.value = np.nan
                self.comment = 'Bad value'
                self.image = 'BADKEY'

        class MockHeader(object):
            def __init__(self, cards):
                self.cards = cards

        bad_header = MockHeader([MockCard()])
        step.merge_scan_hdr(step.dataout, bad_header, step.datain)
        capt = capsys.readouterr()
        assert 'Unable to add FITS keyword BADKEY' in capt.err

        # merge headers without AOR_ID and MISSN-ID
        del df.header['AOR_ID']
        del df.header['MISSN-ID']
        del df.header['ASSC_AOR']
        del df.header['ASSC_MSN']
        step.dataout = df.copy()
        step.merge_scan_hdr(step.dataout, scanmap_header, step.datain)
        assert 'ASSC_AOR' not in df.header
        assert 'ASSC_MSN' not in df.header

    def test_scanmap_frame_clip(self, tmpdir, capsys):
        """Test frame clipping options."""
        # make some data to test
        nf = 80
        hdul = scan_raw_data(nframe=nf)
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        inp = [df]

        with tmpdir.as_cwd():
            step = StepScanMap()

            # run with all frames used, turning off some other
            # clipping options
            options = 'downsample=1 blacklist=vclip,fillgaps ' \
                      'shift=0 chopper.shift=0 rounds=1'
            step(inp, use_frames='', options=options)
            capt = capsys.readouterr()
            assert f'Reading {nf} frames' in capt.out
            assert f'{nf} valid frames' in capt.out

            # run with a specific range, with negative values on end
            step(inp, use_frames='5:-5', options=options)
            capt = capsys.readouterr()
            assert f'{nf - 10} valid frames' in capt.out
            assert f'Removing {10} frames outside range'

            # clip negative values on end
            step(inp, use_frames='*:-5', options=options)
            capt = capsys.readouterr()
            assert f'{nf - 5} valid frames' in capt.out
            assert f'Removing {5} frames outside range'

            # clip from beginning
            step(inp, use_frames='10:*', options=options)
            capt = capsys.readouterr()
            assert f'{nf - 10} valid frames' in capt.out
            assert f'Removing {10} frames outside range'

            # try to use too many -- uses all
            step(inp, use_frames=f'0:{nf + 10}', options=options)
            capt = capsys.readouterr()
            assert 'out of bounds' in capt.err
            assert f'{nf} valid frames' in capt.out

            # pass a bad value -- uses all
            step(inp, use_frames='BADVAL', options=options)
            capt = capsys.readouterr()
            assert 'Bad use_frames' in capt.err
            assert f'{nf} valid frames' in capt.out

    def test_scanmappol_frame_clip(self, tmpdir, capsys):
        """Test frame clipping options, in scanpol mode."""
        # make some data to test
        angle = [5.0, 50.0, 27.0, 72.0]
        inp = []
        nf = 80
        for i in range(4):
            hdul = scan_raw_data(nframe=nf)
            hdul[2].data['hwpCounts'] = angle[i] * 4
            ffile = str(tmpdir.join(f'test{i}.fits'))
            hdul.writeto(ffile, overwrite=True)
            hdul.close()
            inp.append(DataFits(ffile))

        with tmpdir.as_cwd():
            step = StepScanMapPol()

            # run with all frames used, turning off some other
            # clipping options
            options = 'downsample=1 blacklist=vclip,fillgaps ' \
                      'shift=0 chopper.shift=0 rounds=1'
            step(inp, use_frames='', options=options)
            capt = capsys.readouterr()
            assert f'Reading {nf} frames' in capt.out
            assert f'{nf} valid frames' in capt.out

            # clip some from beginning and end
            step(inp, use_frames='5:-5', options=options)
            capt = capsys.readouterr()
            assert f'{nf - 10} valid frames' in capt.out
            assert f'Removing {10} frames outside range'

    def test_scanpol_group(self, tmpdir, capsys, test_options):
        """Test scanpol grouping by SCRIPTID"""

        # make some scanpol data to test:
        # this should make two groups of 4 angles each
        inp = []
        fn = 0
        for scriptid in ['1', '2']:
            angle = [5.0, 50.0, 27.0, 72.0]
            for i in range(4):
                hdul = scan_raw_data()
                hdul[2].data['hwpCounts'] = angle[i] * 4
                hdul[0].header['SCRIPTID'] = scriptid
                ffile = str(tmpdir.join('test{}_{}.fits'.format(scriptid, fn)))
                hdul.writeto(ffile, overwrite=True)
                hdul.close()
                df = DataFits(ffile)
                df.config['data']['filenum'] = r'.*_(\d+).*\.fits'
                inp.append(df)
                fn += 1

        # also test badly formatted options: they should be ignored
        opt = test_options + ' q'

        # move to tmpdir -- scanmap writes temp files
        # to current directory
        with tmpdir.as_cwd():
            # run the scanmappol step
            step = StepScanMapPol()
            out = step(inp, options=opt)

            # check that log records two group runs
            capt = capsys.readouterr()
            assert 'Group 1/2: SCRIPTID = 1' in capt.out
            assert 'Group 2/2: SCRIPTID = 2' in capt.out

            # check that two output files were generated,
            # one for each scriptid
            assert len(out) == 2
            assert 'test1' in out[0].filename
            assert 'test2' in out[1].filename

            # check that file numbers were assigned correctly
            assert '0-3' in out[0].filename
            assert '4-7' in out[1].filename

    def test_scanpol_group_options(self, tmpdir, capsys, mocker):
        # make some minimal input data
        df = DataFits()
        df.imageset(np.zeros((10, 10)), imagename='PRIMARY IMAGE')
        df.imageset(np.zeros((10, 10)), imagename='NOISE')
        df.imageset(np.zeros((10, 10)), imagename='EXPOSURE')
        df.setheadval('TELVPA', 270.0)
        df.setheadval('BUNIT', 'counts')
        fname = str(tmpdir.join('testdata.fits'))
        df.filename = fname
        df.save()

        # mock scanmap to return faster
        mocker.patch.object(SMPR, 'run', return_value=None)

        # run the scanmappol step
        step = StepScanMapPol()
        step([df])

        # no scriptid => group is 'UNKNOWN'
        capt = capsys.readouterr()
        assert 'Group 1/1: SCRIPTID = UNKNOWN' in capt.out

        # input with all different SCRIPTIDs: will warn and run
        # all together, allowing SCANMAP to group by angle
        df1 = df.copy()
        df1.setheadval('SCRIPTID', '1')
        df2 = df.copy()
        df2.setheadval('SCRIPTID', '2')
        step([df1, df2])
        capt = capsys.readouterr()
        assert 'No matching SCRIPTIDs. Running all data together' in capt.err

        # input with disparate VPA: will warn, but continue
        df2.setheadval('TELVPA', 280.)
        step([df1, df2])
        capt = capsys.readouterr()
        assert 'VPA is outside tolerance' in capt.err

        # check that top-level scanmap parameters are passed

        # in previous run: default is off for all options
        assert 'deep=True' not in capt.out
        assert 'faint=True' not in capt.out
        assert 'extended=True' not in capt.out

        # add deep option
        step([df], deep=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'deep=True' in capt.out

        # add faint option
        step([df], faint=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'faint=True' in capt.out

        # add extended option
        step([df], extended=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'extended=True' in capt.out

        # if df.filename is not on disk, it will check for
        # df.rawname
        df.rawname = df.filename
        df.filename = 'badfile.fits'
        step([df])
        step.arglist = {}
        capt = capsys.readouterr()
        assert os.path.basename(fname) in capt.out
        assert 'badfile.fits' not in capt.out

        # if rawname also doesn't exist, it will raise an error
        df.rawname = 'badfile.fits'
        with pytest.raises(ValueError):
            step([df])
        capt = capsys.readouterr()
        assert 'File badfile.fits not found' in capt.err

        # mock an unexpected error in assembly: should be logged
        df.filename = fname
        mocker.patch.object(SMPR, 'run', return_value=[df.to_hdulist()])
        mocker.patch.object(step, 'assemble_scanpol',
                            side_effect=ValueError('test unexpected'))
        out = step([df], options=test_options)
        assert len(out) == 0
        capt = capsys.readouterr()
        assert 'test unexpected' in capt.err

    def test_scanpol_wcs(self):
        # make minimal test cases, as if from SCANMAP
        angle = [5.0, 27.0, 50.0, 72.0]
        dates = ['2019-02-20T04:00:00.000',
                 '2019-02-20T03:00:00.000',
                 '2019-02-20T02:00:00.000',
                 '2019-02-20T01:00:00.000']
        dataout = []
        for i in range(4):
            df = DataFits()
            df.imageset(np.zeros((10, 10)), imagename='PRIMARY IMAGE')
            df.imageset(np.zeros((10, 10)), imagename='NOISE')
            df.imageset(np.zeros((10, 10)), imagename='EXPOSURE')

            df.setheadval('TELVPA', 270.0)
            df.setheadval('HWPINIT', angle[i])
            df.setheadval('CDELT2', 1.0)
            df.setheadval('DATE-OBS', dates[i])
            df.setheadval('EXPTIME', 100.0)

            # add a different WCS value for each file

            # for R
            df.setheadval('SUBARRAY', 'R0')
            df.setheadval('CRPIX1', i)
            df.setheadval('CRPIX2', i)
            df.setheadval('CRVAL1', i)
            df.setheadval('CRVAL2', i)
            dataout.append(df)

            # for T
            tdf = df.copy()
            tdf.setheadval('SUBARRAY', 'T0')
            tdf.setheadval('CRPIX1', i + 10)
            tdf.setheadval('CRPIX2', i + 10)
            tdf.setheadval('CRVAL1', i + 10)
            tdf.setheadval('CRVAL2', i + 10)
            dataout.append(tdf)

        # minimal data input with standard SOFIA keys for merging
        df = DataFits()
        df.setheadval('OBS_ID', 'UNKNOWN')
        df.setheadval('TELVPA', 270.0)
        df.setheadval('CRPIX1', 42)
        df.setheadval('CRPIX2', 42)
        df.setheadval('CRVAL1', 42)
        df.setheadval('CRVAL2', 42)

        step = StepScanMapPol()
        step.datain = [df]

        # assemble data into CRH file
        basehead = df.header
        assembled = step.assemble_scanpol(dataout, basehead)
        assert isinstance(assembled, DataFits)

        # check WCS for each extension
        # there are 6 extensions per angle:
        # R, T, R error, T error, R exposure, T exposure
        assert len(assembled.imgheads) == 24
        hwp_idx = -1
        for i, imhead in enumerate(assembled.imgheads):
            if i % 6 == 0:
                hwp_idx += 1
            sub = imhead['SUBARRAY']
            if sub == 'R0':
                assert np.allclose(imhead['CRPIX1'], hwp_idx)
                assert np.allclose(imhead['CRPIX2'], hwp_idx)
                assert np.allclose(imhead['CRVAL1'], hwp_idx)
                assert np.allclose(imhead['CRVAL2'], hwp_idx)
            else:
                assert np.allclose(imhead['CRPIX1'], hwp_idx + 10)
                assert np.allclose(imhead['CRPIX2'], hwp_idx + 10)
                assert np.allclose(imhead['CRVAL1'], hwp_idx + 10)
                assert np.allclose(imhead['CRVAL2'], hwp_idx + 10)

    def test_sibs_offsets(self, tmpdir):
        """Test SCANMAP offset calculations."""
        # test on real data containing a point source
        data_path = os.path.join(os.path.dirname(hawc.__file__),
                                 'tests', 'data', 'uranusD.fits')
        # skip test if file not found
        if not os.path.isfile(data_path):
            return

        ffile = os.path.join(tmpdir, 'uranusD.fits')
        shutil.copyfile(data_path, ffile)

        df = DataFits(ffile)
        inp = [df]

        with tmpdir.as_cwd():
            step = StepScanMap()

            out = step(inp)[0]
            tabhead = out.tabheads[0]
            for key in ['SIBS_DX', 'SIBS_DY', 'SIBS_DE', 'SIBS_DXE']:
                assert key in tabhead

            # magnitude of dx/y and de/xe should match
            dxy = np.sqrt(tabhead['SIBS_DX']**2 + tabhead['SIBS_DY']**2)
            dexe = np.sqrt(tabhead['SIBS_DE']**2 + tabhead['SIBS_DXE']**2)

            pixscal = out.header['PIXSCAL']
            assert np.allclose(dxy * pixscal, dexe)

    def test_outpath_space(self, tmpdir):
        """Test that SCANMAP commands handle spaces in directory names."""
        cwd = os.getcwd()
        dir_name = tmpdir.join('test directory')
        os.mkdir(dir_name)
        os.chdir(dir_name)

        hdul = scan_raw_data()
        ffile = str(dir_name.join('test file.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)

        step = StepScanMap()
        out = step([df])
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], DataFits)

        # make scan pol data
        angle = [5.0, 50.0, 27.0, 72.0]
        inp = []
        for i in range(4):
            hdul = scan_raw_data()
            hdul[2].data['hwpCounts'] = angle[i] * 4

            ffile = str(tmpdir.join('test {}.fits'.format(i)))
            hdul.writeto(ffile, overwrite=True)
            hdul.close()
            inp.append(DataFits(ffile))

        step = StepScanMapPol()
        out = step(inp)
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], DataFits)

        # return to starting directory
        os.chdir(cwd)

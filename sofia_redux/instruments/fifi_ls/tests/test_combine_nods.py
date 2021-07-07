# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_csb_files
from sofia_redux.instruments.fifi_ls.combine_nods \
    import (combine_nods, _mjd, _read_exthdrs,
            classify_files, combine_extensions)


class TestCombineNods(FIFITestCase):

    def make_nodstyles(self):
        # default files are NMC
        files = get_csb_files()

        # modify files to be C2NC2
        asym = []
        c2nc2 = []
        sym = []
        for fn in files:
            hdul = fits.open(fn)
            sym.append(hdul)

            asym_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
            asym_hdul[0].header['NODSTYLE'] = 'ASYMMETRIC'
            asym_hdul[0].header['FILENAME'] += '.asym.fits'
            asym.append(asym_hdul)

            c2nc2_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
            c2nc2_hdul[0].header['NODSTYLE'] = 'C2NC2'
            c2nc2_hdul[0].header['FILENAME'] += '.c2nc2.fits'
            c2nc2.append(c2nc2_hdul)

        return sym, asym, c2nc2

    def test_nowrite(self, tmpdir):
        files = get_csb_files()
        df = combine_nods(files, offbeam=False, outdir=str(tmpdir),
                          write=False)
        noda = df[df['nodbeam'] == 'A']
        for _, row in noda.iterrows():
            assert row['chdul'] is not None
            assert not os.path.isfile(row['outfile'])

    def test_write(self, tmpdir):
        files = get_csb_files()
        df = combine_nods(
            files, offbeam=False, outdir=str(tmpdir), write=True)
        noda = df[df['nodbeam'] == 'A']
        failure = False
        for _, row in noda.iterrows():
            if row['chdul'] is None:
                failure = True
                print("HDUL not created")
            if not os.path.isfile(row['outfile']):
                failure = True
                print("file %s not found" % row['outfile'])
        assert not failure

    def test_offbeam(self):
        files = get_csb_files()
        df_a = combine_nods(files, offbeam=False, outdir=None, write=False)
        df_b = combine_nods(files, offbeam=True, outdir=None, write=False)
        assert (df_a[df_b['nodbeam'] == 'B']['nodbeam'] == 'A').all()

    def test_mjd(self):
        dateobs = '2016-03-01T10:38:39'
        expected = 57448.443506944444
        assert np.allclose(_mjd(dateobs), expected)

        dateobs = 'BADVAL'
        assert _mjd(dateobs) == 0

    def test_read_exthdrs(self):
        test_file = get_csb_files()[0]
        hdul = fits.open(test_file)

        # first extension only -- returns empty array
        result = _read_exthdrs([hdul[0]], 'INDPOS')
        assert len(result) == 0

        # otherwise returns array of values from all extensions
        result = _read_exthdrs(hdul, 'INDPOS')
        assert np.allclose(result, [463923, 464433, 464943, 465453])

    def test_classify_nodstyles(self, capsys):
        # default files are NMC
        files = get_csb_files()
        df = classify_files(files)
        assert (df['nodstyle'] == 'NMC').all()
        assert not df['asymmetric'].any()

        sym, asym, c2nc2 = self.make_nodstyles()

        df = classify_files(sym)
        assert (df['nodstyle'] == 'NMC').all()
        assert not df['asymmetric'].any()

        df = classify_files(asym)
        assert (df['nodstyle'] == 'ASYMMETRIC').all()
        assert df['asymmetric'].all()

        df = classify_files(c2nc2)
        assert (df['nodstyle'] == 'C2NC2').all()
        assert df['asymmetric'].all()

        # mixed set: should warn but still classify okay
        df = classify_files(sym + asym + c2nc2)
        assert df['asymmetric'].sum() == len(asym) + len(c2nc2)
        capt = capsys.readouterr()
        assert 'Mismatched NODSTYLE' in capt.err

        # test exptime=0 for asym B beams, not for others
        asym_b = df[df['asymmetric'] & (df['nodbeam'] == 'B')]
        others = df[~(df['asymmetric'] & (df['nodbeam'] == 'B'))]

        # Bs
        exptimes = [hdul[0].header['EXPTIME'] for hdul in asym_b['hdul']]
        nexps = [hdul[0].header['NEXP'] for hdul in asym_b['hdul']]
        assert np.allclose(exptimes, 0)
        assert np.allclose(nexps, 0)

        # others
        exptimes = [hdul[0].header['EXPTIME'] for hdul in others['hdul']]
        nexps = [hdul[0].header['NEXP'] for hdul in others['hdul']]
        assert not np.any(np.isclose(exptimes, 0))
        assert not np.any(np.isclose(nexps, 0))

    def test_classify_errors(self, capsys):
        # bad filenames only
        result = classify_files(['badfile1.fits', 'badfile2.fits'])
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid HDUList' in capt.err

        # mixed good and bad: continues with good
        files = get_csb_files()
        result = classify_files(['badfile1.fits', 'badfile2.fits'] + files)
        assert len(result) == len(files)
        capt = capsys.readouterr()
        assert 'Invalid HDUList' in capt.err

        # good files, but some bad dates -- keeps good
        inp = []
        for i, f in enumerate(files):
            hdul = fits.open(f)
            if i % 2 == 0:
                hdul[0].header['DATE-OBS'] = 'BADVAL'
            inp.append(hdul)
        result = classify_files(inp)
        assert len(result) == len(files) // 2
        capt = capsys.readouterr()
        assert 'DATE-OBS in header is BADVAL' in capt.err

    def test_combine_nodstyles(self, capsys):
        # make 2 A, 2B in each of sym and asym styles
        sym, asym, _ = self.make_nodstyles()

        # sym
        # expect that file 1 matches with 2, 3 matches with 4
        sym_df = classify_files(sym)

        result = combine_extensions(sym_df)
        result_a = result[result['nodbeam'] == 'A']
        assert not result_a['chdul'].isna().any()
        capt = capsys.readouterr()
        assert 'Adding B {} to ' \
               'A {}'.format(sym[1][0].header['FILENAME'],
                             sym[0][0].header['FILENAME']) in capt.out
        assert 'Adding B {} to ' \
               'A {}'.format(sym[3][0].header['FILENAME'],
                             sym[2][0].header['FILENAME']) in capt.out

        # asym ABA
        # expect that file 1 and 3 match with 2
        asym_df = classify_files(asym[:3])
        result = combine_extensions(asym_df)
        result_a = result[result['nodbeam'] == 'A']
        assert not result_a['chdul'].isna().any()
        capt = capsys.readouterr()
        assert 'Subbing B {} from ' \
               'A {}'.format(asym[1][0].header['FILENAME'],
                             asym[0][0].header['FILENAME']) in capt.out
        assert 'Subbing B {} from ' \
               'A {}'.format(asym[1][0].header['FILENAME'],
                             asym[2][0].header['FILENAME']) in capt.out

        # pass only As: no combination done
        a_df = classify_files([sym[0], sym[2]])
        result = combine_extensions(a_df)
        assert result['chdul'].isna().all()
        capt = capsys.readouterr()
        assert 'No B nods found' in capt.err

        # pass only Bs: no combination done
        b_df = classify_files([sym[1], sym[3]])
        result = combine_extensions(b_df)
        assert result['chdul'].isna().all()
        capt = capsys.readouterr()
        assert 'No A nods found' in capt.err

        # pass an A without a matching B
        aba_df = classify_files(sym[:3])
        result = combine_extensions(aba_df)
        assert result['chdul'].isna().sum() == 2
        capt = capsys.readouterr()
        assert 'No matching B found for A ' \
               '{}'.format(sym[2][0].header['FILENAME']) in capt.out

    def test_combine_ext_failure(self, capsys):
        # test bad method argument
        with pytest.raises(ValueError) as err:
            combine_extensions([1, 2, 3], b_nod_method='wrong')
        assert 'Bad b_nod_method' in str(err)

    def test_combine_nods_failure(self, capsys, mocker):
        # bad files
        result = combine_nods(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input" in capt.err

        # single file, bad outdir
        files = get_csb_files()
        result = combine_nods(files[0], write=False, outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert "Output directory badval does not exist" in capt.err

        # problem in classification
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.combine_nods.classify_files',
            return_value=None)
        result = combine_nods(files[0], write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem in file classification' in capt.err

    def test_combined_header(self):
        sym, asym, _ = self.make_nodstyles()

        # test that for asym headers, the A is the basehead,
        # even if the B is earlier
        a_file, b_file = asym[0], asym[1]
        a_file[0].header['DATE-OBS'] = '2019-01-01T02:00:00'
        b_file[0].header['DATE-OBS'] = '2019-01-01T01:00:00'
        result = combine_nods([b_file, a_file], write=False)
        abeams = result[result['nodbeam'] == 'A']
        chdul = abeams['chdul'][0]

        assert chdul[0].header['EXPTIME'] > 0
        assert chdul[0].header['NEXP'] > 0
        assert chdul[0].header['DATE-OBS'] == '2019-01-01T02:00:00'
        assert chdul[0].header['DLAM_MAP'] == a_file[0].header['DLAM_MAP']
        assert chdul[0].header['DBET_MAP'] == a_file[0].header['DBET_MAP']

        # check bunit
        for ext in chdul[1:]:
            assert ext.header['BUNIT'] == 'adu/s'

    def test_scanpos(self):
        _, _, c2nc2 = self.make_nodstyles()

        # if no scanpos data present, no errors, nothing propagated
        result = combine_nods(c2nc2, write=False)
        for hdul in result[result['nodbeam'] == 'A']['chdul']:
            assert isinstance(hdul, fits.HDUList)
            for i in range(hdul[0].header['NGRATING']):
                assert f'SCANPOS_G{i}' not in hdul

        # if scanpos data present, it should be passed forward
        # unmodified
        updated = []
        for hdul in c2nc2:
            for i in range(hdul[0].header['NGRATING']):
                hdul.append(fits.ImageHDU(np.arange(i + 10, dtype=float),
                                          name=f'SCANPOS_G{i}'))
            updated.append(hdul)
        result = combine_nods(updated, write=False)
        for hdul in result[result['nodbeam'] == 'A']['chdul']:
            assert isinstance(hdul, fits.HDUList)
            for i in range(hdul[0].header['NGRATING']):
                orig_hdul = result[result['nodbeam'] == 'A']['hdul'][0]
                assert f'SCANPOS_G{i}' in hdul
                assert isinstance(orig_hdul, fits.HDUList)
                assert np.allclose(hdul[f'SCANPOS_G{i}'].data,
                                   orig_hdul[f'SCANPOS_G{i}'].data)

    def test_average_two(self, capsys):
        _, _, c2nc2_2files = self.make_nodstyles()

        # add a couple more files to combine
        c2nc2 = [c2nc2_2files[0], c2nc2_2files[1],
                 fits.HDUList([hdu.copy() for hdu in c2nc2_2files[0]]),
                 fits.HDUList([hdu.copy() for hdu in c2nc2_2files[1]])]
        hdr = c2nc2[2][0].header
        hdr['FILENAME'] = \
            c2nc2[0][0].header['FILENAME'].replace('_001', '_003')
        hdr['ASSC_OBS'] = 'R003'
        hdr['DATE-OBS'] = '2016-03-01T10:39:39'
        hdr = c2nc2[3][0].header
        hdr['FILENAME'] = \
            c2nc2[1][0].header['FILENAME'].replace('_002', '_004')
        hdr['ASSC_OBS'] = 'R004'
        hdr['DATE-OBS'] = '2016-03-01T10:39:51'

        # default: take only nearest nod
        default = combine_nods(c2nc2, write=False, b_nod_method='nearest')
        capt = capsys.readouterr()
        assert 'Subbing B {} from ' \
               'A {}'.format(c2nc2[1][0].header['FILENAME'],
                             c2nc2[0][0].header['FILENAME']) in capt.out
        assert 'Subbing B {} from ' \
               'A {}'.format(c2nc2[3][0].header['FILENAME'],
                             c2nc2[2][0].header['FILENAME']) in capt.out
        hdul = default[default['nodbeam'] == 'A']['chdul'][0]
        assert hdul[0].header['ASSC_OBS'] == 'R001,R002'
        hdul = default[default['nodbeam'] == 'A']['chdul'][1]
        assert hdul[0].header['ASSC_OBS'] == 'R003,R004'

        # average 2: first A will take only nearest B nod, since
        # there's no before B nod.  second A will average before
        # and after B.
        result = combine_nods(c2nc2, write=False, b_nod_method='average')
        capt = capsys.readouterr()
        assert 'Subbing B {} from ' \
               'A {}'.format(c2nc2[1][0].header['FILENAME'],
                             c2nc2[0][0].header['FILENAME']) in capt.out
        assert 'Averaging B {} and {} and subbing from ' \
               'A {}'.format(c2nc2[1][0].header['FILENAME'],
                             c2nc2[3][0].header['FILENAME'],
                             c2nc2[2][0].header['FILENAME']) in capt.out
        hdul1 = result[result['nodbeam'] == 'A']['chdul'][0]
        assert hdul1[0].header['ASSC_OBS'] == 'R001,R002'
        hdul2 = result[result['nodbeam'] == 'A']['chdul'][1]
        assert hdul2[0].header['ASSC_OBS'] == 'R002,R003,R004'

        # error estimate on averaged one should be lower, flux should be same
        assert np.allclose(hdul2['FLUX_G0'].data,
                           hdul1['FLUX_G0'].data, equal_nan=True)
        assert np.nansum(hdul2['STDDEV_G0'].data) \
            < np.nansum(hdul1['STDDEV_G0'].data)

        # change the data values, to verify averaging
        c2nc2[1]['FLUX_G0'].data[:] = 10.0
        c2nc2[2]['FLUX_G0'].data[:] = 40.0
        c2nc2[3]['FLUX_G0'].data[:] = 30.0
        result = combine_nods(c2nc2, write=False, b_nod_method='nearest')
        hdul = result[result['nodbeam'] == 'A']['chdul'][1]
        assert np.allclose(hdul['FLUX_G0'].data, 10)

        c2nc2[2]['FLUX_G0'].data[:] = 40.0
        result = combine_nods(c2nc2, write=False, b_nod_method='average')
        hdul = result[result['nodbeam'] == 'A']['chdul'][1]
        assert np.allclose(hdul['FLUX_G0'].data, 20)

    def test_interpolate_two(self, capsys):
        _, _, c2nc2_2files = self.make_nodstyles()

        # add a couple more files to combine
        c2nc2 = [c2nc2_2files[0], c2nc2_2files[1],
                 fits.HDUList([hdu.copy() for hdu in c2nc2_2files[0]]),
                 fits.HDUList([hdu.copy() for hdu in c2nc2_2files[1]])]
        hdr = c2nc2[2][0].header
        hdr['FILENAME'] = \
            c2nc2[0][0].header['FILENAME'].replace('_001', '_003')
        hdr['ASSC_OBS'] = 'R003'
        hdr['DATE-OBS'] = '2016-03-01T10:39:39'
        hdr = c2nc2[3][0].header
        hdr['FILENAME'] = \
            c2nc2[1][0].header['FILENAME'].replace('_002', '_004')
        hdr['ASSC_OBS'] = 'R004'
        hdr['DATE-OBS'] = '2016-03-01T10:39:51'
        hdr['START'] += 2

        # time comes from start, fifistrt, alpha keys
        atime = 1456857532.28
        btime1 = 1456857531.0
        btime2 = 1456857533.0

        # interpolate: first A will take only nearest B nod, since
        # there's no before B nod.  second A will interpolate before
        # and after B.
        result = combine_nods(c2nc2, write=False, b_nod_method='interpolate')
        capt = capsys.readouterr()
        assert 'Subbing B {} from ' \
               'A {}'.format(c2nc2[1][0].header['FILENAME'],
                             c2nc2[0][0].header['FILENAME']) in capt.out
        assert 'Interpolating B {} at {} and {} at {} to A time {} ' \
               'and subbing from ' \
               'A {}'.format(c2nc2[1][0].header['FILENAME'], btime1,
                             c2nc2[3][0].header['FILENAME'], btime2, atime,
                             c2nc2[2][0].header['FILENAME']) in capt.out
        hdul1 = result[result['nodbeam'] == 'A']['chdul'][0]
        assert hdul1[0].header['ASSC_OBS'] == 'R001,R002'
        hdul2 = result[result['nodbeam'] == 'A']['chdul'][1]
        assert hdul2[0].header['ASSC_OBS'] == 'R002,R003,R004'

        # error estimate on interpolated one should be close to
        # nearest, flux should be same
        assert np.allclose(hdul2['FLUX_G0'].data,
                           hdul1['FLUX_G0'].data, equal_nan=True)
        assert np.allclose(np.nansum(hdul2['STDDEV_G0'].data),
                           np.nansum(hdul1['STDDEV_G0'].data),
                           rtol=0.1)

        # change the data values, to verify interpolation
        c2nc2[1]['FLUX_G0'].data[:] = 10.0
        c2nc2[2]['FLUX_G0'].data[:] = 40.0
        c2nc2[3]['FLUX_G0'].data[:] = 30.0
        result = combine_nods(c2nc2, write=False, b_nod_method='nearest')
        hdul = result[result['nodbeam'] == 'A']['chdul'][1]
        assert np.allclose(hdul['FLUX_G0'].data, 10)

        # expected B value is point on the line between B values, at a time
        interp_val = (atime - btime1) * (30. - 10) / (btime2 - btime1) + 10.
        c2nc2[2]['FLUX_G0'].data[:] = 40.0
        result = combine_nods(c2nc2, write=False, b_nod_method='interpolate')
        hdul = result[result['nodbeam'] == 'A']['chdul'][1]
        idx = ~np.isnan(hdul['FLUX_G0'].data)
        assert np.allclose(hdul['FLUX_G0'].data[idx], 40 - interp_val)

        # modify A data in first grating to make it OTF style
        # (3D cube, with RAMPSTRT, RAMPEND for times)
        c2nc2[2]['FLUX_G0'].data = np.full((10, 16, 25), 40.0)
        # set start and end times to B times
        c2nc2[2]['FLUX_G0'].header['RAMPSTRT'] = btime1
        c2nc2[2]['FLUX_G0'].header['RAMPEND'] = btime2
        incr = (btime2 - btime1) / 9

        result = combine_nods(c2nc2, write=False, b_nod_method='interpolate')
        hdul = result[result['nodbeam'] == 'A']['chdul'][1]
        # check interpolated value at each frame
        for i in range(10):
            atime = btime1 + i * incr
            interp_val = \
                (atime - btime1) * (30. - 10) / (btime2 - btime1) + 10.
            data = hdul['FLUX_G0'].data[i]
            idx = ~np.isnan(data)
            assert np.allclose(data[idx], 40 - interp_val)

        # check error if time keys missing and interpolate on
        del c2nc2[2][0].header['START']
        result = combine_nods(c2nc2, write=False, b_nod_method='nearest')
        assert result is not None
        with pytest.raises(ValueError) as err:
            combine_nods(c2nc2, write=False, b_nod_method='interpolate')
        assert 'Missing START, FIFISTRT, ALPHA, or EXPTIME' in str(err)

        # check error if OTF-style data and offbeam selected
        with pytest.raises(ValueError) as err:
            combine_nods(c2nc2, write=False, offbeam=True,
                         b_nod_method='nearest')
        assert 'Offbeam option is not available for OTF mode' in str(err)

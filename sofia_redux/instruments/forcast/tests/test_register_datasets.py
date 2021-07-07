# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.forcast.register_datasets \
    import (wcs_shift, get_shifts, expand_array,
            shift_set, shift_datasets,
            resize_datasets, register_datasets)
from sofia_redux.instruments.forcast.setpar import setpar
from sofia_redux.instruments.forcast.tests.resources \
    import npc_testdata, nmc_testdata
from sofia_redux.toolkit.image import adjust


def create_datasets(n=5, maxoff=10, wcs=False, centroid=False,
                    normmap=False, specstyle=False):
    test = nmc_testdata() if centroid else npc_testdata()
    d0 = test['data'].copy()
    h0 = test['header'].copy()
    h0['CRVAL1'] = 0.0
    h0['CRVAL2'] = 0.0
    if specstyle:
        h0['CRPIX1A'] = 0.0
        h0['CRPIX2A'] = 0.0
        h0['CRPIX3A'] = 0.0
        h0['CRVAL1A'] = 0.0
        h0['CRVAL2A'] = 0.0
        h0['CRVAL3A'] = 0.0

    rand = np.random.RandomState(42)
    v0 = test['data'].copy() * 0.1
    if normmap:
        datasets = [(d0, h0, v0, d0 * 0 + 1)]
    else:
        datasets = [(d0, h0, v0)]
    shifts = [[0, 0.]]
    if n > 1:
        for i in range(n - 1):
            dx, dy = rand.normal(0, maxoff, 2)
            shifts.append([dx, dy])
            if not wcs:
                dnew = adjust.shift(d0.copy(), (-dy, -dx),
                                    order=3, missing=0)
                vnew = adjust.shift(v0.copy(), (-dy, -dx),
                                    order=0, missing=0)
                if not normmap:
                    datasets.append((dnew, h0.copy(), vnew))
                else:
                    datasets.append((dnew, h0.copy(), vnew, dnew * 0 + 1))
            else:
                hnew = h0.copy()
                hnew['CRVAL1'] += dx
                hnew['CRVAL2'] += dy

                if specstyle:
                    hnew['CRVAL2A'] += dy
                    hnew['CRVAL3A'] += dx

                if not normmap:
                    datasets.append((d0.copy(), hnew, v0.copy()))
                else:
                    datasets.append((d0.copy(), hnew, v0.copy(), d0 * 0 + 1))
    return datasets, shifts


class TestRegisterDatasets(object):

    def test_wcs_shift(self):
        datasets, shifts = create_datasets(wcs=True)
        refheader = datasets[0][1]
        for s, dset in zip(shifts[1:], datasets[1:]):
            offset = wcs_shift(refheader, dset[1],
                               [dset[1]['CRPIX1'], dset[1]['CRPIX2']])
            assert np.allclose(offset, s)

    def test_get_crpix_shifts(self):
        # for WCS shift, should return all zeros
        setpar('CORCOADD', 'WCS')
        datasets, shifts = create_datasets(wcs=True)
        pixshifts = get_shifts(datasets, do_wcs_shift=True)
        assert np.allclose(pixshifts, 0)

        # for centroid shift: returns shifts minus wcs shift
        setpar('CORCOADD', 'CENTROID')
        datasets, shifts = create_datasets(centroid=True)
        pixshifts = get_shifts(datasets, do_wcs_shift=True)
        refheader = datasets[0][1]
        for ps, s, dset in zip(pixshifts[1:], shifts[1:], datasets[1:]):
            offset = wcs_shift(refheader, dset[1],
                               [refheader['CRPIX1'], refheader['CRPIX2']])
            assert np.allclose(-1 * (ps - offset), s, atol=0.001)

    def test_3d_wcs(self):
        datasets, shifts = create_datasets(wcs=True, specstyle=True)

        # wcs shift: returns 3D for 3D WCS
        refheader = datasets[0][1]
        for s, dset in zip(shifts[1:], datasets[1:]):
            offset = wcs_shift(refheader, dset[1],
                               [dset[1]['CRPIX1A'], dset[1]['CRPIX2A'],
                                dset[1]['CRPIX3A']],
                               wcskey='A')[0]
            assert np.allclose(offset[2], s[0])
            assert np.allclose(offset[1], s[1])

        # get_shifts: returns 2D for 3D WCS
        setpar('CORCOADD', 'WCS')
        pixshifts = get_shifts(datasets, do_wcs_shift=False, wcskey='A')
        assert np.allclose(pixshifts, shifts)

        # get_shifts with WCS: should return 0
        setpar('CORCOADD', 'WCS')
        pixshifts = get_shifts(datasets, do_wcs_shift=True, wcskey='A')
        assert np.allclose(pixshifts, 0)

    def test_apply_wcs_shift(self):
        datasets, shifts = create_datasets(wcs=True)
        setpar('CORCOADD', 'WCS')
        s = get_shifts(datasets)
        for s0, s1 in zip(shifts, s):
            assert np.allclose(s0, s1)

    def test_header(self):
        datasets, shifts = create_datasets()
        setpar('CORCOADD', 'HEADER')
        s0 = get_shifts(datasets)
        for s in s0[1:]:
            assert np.allclose(s, s0[0])

    def test_override(self):
        datasets, shifts = create_datasets()
        setpar('CORCOADD', 'OVERRIDE')
        user_shifts = []
        for i in range(len(shifts)):
            user_shifts.append([float(i + 1), float(i + 1)])
        s = get_shifts(datasets, user_shifts=user_shifts)
        for spos, sneg in zip(user_shifts, s):
            assert spos[0] == -sneg[0]
            assert spos[1] == -sneg[1]

    def test_expand_array(self):
        array1 = np.full((5, 5), 1)
        array2 = expand_array(array1, (10, 9), missing=2)
        assert np.allclose(array2[:5, :5], 1)
        assert array2.shape == (10, 9)
        assert np.allclose(array2[5:, :], 2)
        assert np.allclose(array2[:, 5:], 2)

        # bad array
        assert expand_array(10, (10, 9)) is None

    def test_shift_set(self):
        dataset, shifts = create_datasets(2)
        oset = dataset[0]
        dataset = dataset[1]
        normmap = dataset[0] * 0 + 1
        dataset = [dataset[0], dataset[1], dataset[2], normmap]
        d1 = dataset[0] - oset[0]
        assert not np.allclose(d1[~np.isnan(d1)], 0, atol=0.1)
        shifted = shift_set(dataset, shifts[1])

        # test data
        d2 = shifted[0] - oset[0]
        assert np.allclose(d2[~np.isnan(d2)], 0, atol=0.1)
        assert np.isnan(d2).sum() < (~np.isnan(d2)).sum()

        # test variance
        d3 = shifted[2] - oset[2]
        assert np.allclose(d3[~np.isnan(d3)], 0, atol=0.1)

        # test normmap
        assert np.isnan(shifted[3]).any()

        # check bad variance
        dataset[2] = 10
        shifted = shift_set(dataset, shifts[1])
        assert shifted[2] is None

    def test_shift_datasets(self):
        dataset, shifts = create_datasets(2)
        o1data = dataset[0][0]
        o2data = dataset[1][0]
        xmax = o1data.shape[1]
        ymax = o1data.shape[0]

        # single set
        result = shift_datasets([dataset[0]],
                                np.array([[10, 10]]), 0)
        s1data = result[0][0]
        assert np.allclose(s1data[10:ymax, 10:xmax],
                           o1data[:-10, :-10])

        # two sets
        result = shift_datasets(dataset,
                                np.array([[10, 10], [20, 20]]), 0)
        s1data = result[0][0]
        s2data = result[1][0]
        assert np.allclose(s1data[10:ymax, 10:xmax],
                           o1data[:-10, :-10])
        assert np.allclose(s2data[20:ymax, 20:xmax],
                           o2data[:-20, :-20])

    def test_resize_datasets(self):
        datasets, shifts = create_datasets(normmap=True)
        current_size = datasets[0][0].shape

        # clip datasets
        clipped = []
        for i, ds in enumerate(datasets):
            clipped.append([ds[0][i * 10:, i * 10:], ds[1],
                            ds[2][i * 10:, i * 10:],
                            ds[3][i * 10:, i * 10:]])

        result = resize_datasets(clipped)
        for i, ds in enumerate(result):
            if i == 0:
                assert ds[0].shape == clipped[i][0].shape
            else:
                assert ds[0].shape != clipped[i][0].shape
            assert ds[0].shape == current_size
            assert ds[2].shape == current_size
            assert ds[3].shape == current_size

    def test_register_datasets(self):
        setpar('corcoadd', 'CENTROID')
        datasets, shifts = create_datasets(centroid=True, normmap=True)
        result = register_datasets(datasets)
        set0 = result[0]
        for dset in result[1:]:
            diff_d = set0[0] - dset[0]
            mask = ~np.isnan(diff_d)
            assert np.allclose(diff_d[mask], 0, atol=0.1)

            # different mask for variance -- uses order 0 shifting
            diff_v = set0[2] - dset[2]
            mask = ~np.isnan(diff_v)
            assert np.allclose(diff_v[mask], 0, atol=0.1)

            # Test that the normalization map used order 0 shifting
            assert np.nanmin(dset[3]), 1
            assert np.nanmax(dset[3]), 1

    def test_regset_errors(self, capsys, mocker):
        dataset, shifts = create_datasets(2)
        user_shifts = []
        for i in range(len(shifts)):
            user_shifts.append([float(i + 1), float(i + 1)])

        # datasets not a list
        assert register_datasets(1) is None
        capt = capsys.readouterr()
        assert 'Invalid dataset' in capt.err

        # bad user shifts
        setpar('CORCOADD', 'OVERRIDE')
        # the right way
        assert register_datasets(dataset,
                                 user_shifts=user_shifts) is not None
        # the wrong way
        assert register_datasets(dataset,
                                 user_shifts=1) is None
        assert register_datasets(dataset,
                                 user_shifts=[1, 2, 3]) is None
        setpar('CORCOADD', 'HEADER')

        # the following are bad datasets
        # should be: (data, header, var, nmap), nmap optional

        goodset = dataset[0]

        # too few items
        badset = (goodset[0])
        dataset[0] = badset
        assert register_datasets(dataset) is None
        capt = capsys.readouterr()
        assert 'invalid elements' in capt.err

        # too many
        badset = (goodset[0], goodset[1],
                  goodset[2], goodset[2], goodset[2])
        dataset[0] = badset
        assert register_datasets(dataset) is None
        capt = capsys.readouterr()
        assert 'invalid elements' in capt.err

        # bad data - not an array
        badset = (1, goodset[1], goodset[2])
        dataset[0] = badset
        assert register_datasets(dataset) is None
        capt = capsys.readouterr()
        assert 'invalid image' in capt.err

        # bad header
        badset = (goodset[0], 1, goodset[2])
        dataset[0] = badset
        assert register_datasets(dataset) is None
        capt = capsys.readouterr()
        assert 'invalid header' in capt.err

        # bad variance
        badset = (goodset[0], goodset[1], np.array(10))
        dataset[0] = badset
        assert register_datasets(dataset) is None
        capt = capsys.readouterr()
        assert 'invalid variance' in capt.err

        # bad normmap
        badset = (goodset[0], goodset[1], goodset[2], np.array(10))
        dataset[0] = badset
        assert register_datasets(dataset) is None
        capt = capsys.readouterr()
        assert 'invalid normmap' in capt.err

        # mock get_shifts for failure
        dataset[0] = goodset
        assert register_datasets(dataset) is not None
        mocker.patch(
            'sofia_redux.instruments.forcast.register_datasets.get_shifts',
            return_value=[None, None])
        assert register_datasets(dataset) is None
        capt = capsys.readouterr()
        assert 'failed to register' in capt.err

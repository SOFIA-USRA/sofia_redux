# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.instruments.flitecam.split_input import split_input


class TestSplitInput(object):

    def test_success(self):
        infiles = []
        for i in range(6):
            hdul = fits.HDUList(fits.PrimaryHDU())
            if i % 3 == 0:
                hdul[0].header['OBSTYPE'] = 'OBJECT'
            elif i % 2 == 0:
                hdul[0].header['OBSTYPE'] = 'SKY'
            else:
                hdul[0].header['OBSTYPE'] = 'STANDARD'
            infiles.append(hdul)

        manifest = split_input(infiles)
        assert len(manifest['object']) == 2
        assert len(manifest['sky']) == 2
        assert len(manifest['standard']) == 2

        assert manifest['object'][0] == infiles[0]
        assert manifest['object'][1] == infiles[3]
        assert manifest['sky'][0] == infiles[2]
        assert manifest['sky'][1] == infiles[4]
        assert manifest['standard'][0] == infiles[1]
        assert manifest['standard'][1] == infiles[5]

    def test_bad_key(self):
        hdul = fits.HDUList(fits.PrimaryHDU())

        # missing key gets classed as object
        manifest = split_input([hdul])
        assert len(manifest['object']) == 1
        assert len(manifest['sky']) == 0
        assert len(manifest['standard']) == 0
        assert manifest['object'][0] == hdul

        # same for unrecognized
        hdul[0].header['OBSTYPE'] = 'BAD'
        manifest = split_input([hdul])
        assert len(manifest['object']) == 1
        assert len(manifest['sky']) == 0
        assert len(manifest['standard']) == 0
        assert manifest['object'][0] == hdul

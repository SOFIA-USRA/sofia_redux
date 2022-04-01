# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.scan.info.telescope import TelescopeInfo


class TestTelescopeInfo(object):
    def test_init(self):
        info = TelescopeInfo()
        assert not info.is_tracking
        assert info.telescope is None
        assert info.log_id == 'tel'
        assert info.get_telescope_name() == 'UNKNOWN'

    def test_edit_image_header(self):
        info = TelescopeInfo()
        header = fits.Header()

        info.edit_image_header(header)
        assert header['TELESCOP'] == 'UNKNOWN'
        assert len(header) == 1

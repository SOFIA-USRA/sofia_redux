# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.instruments.forcast.getpar import getpar
import sofia_redux.instruments.forcast.configuration as dripconfig


class TestGetpar(object):

    def test_configuration_loaded(self):
        """Check the configuration was loaded"""
        dripconfig.load()
        assert 'doinhdch' in dripconfig.configuration

    def test_basic_functionality(self):
        """Read value from configuration, also store it in the header"""
        dripconfig.load()
        header = fits.header.Header()
        value = getpar(header, 'doinhdch')
        assert value.lower() in ['true', 'false']
        assert 'doinhdch' in header
        assert header['doinhdch'] == value

    def test_no_update(self):
        """Don't update the header when retrieving a value"""
        dripconfig.load()
        header = fits.header.Header()
        getpar(header, 'doinhdch', update_header=False)
        assert 'doinhdch' not in header

    def test_comments(self):
        """Check we can add a comment to the header"""
        dripconfig.load()
        header = fits.header.Header()
        getpar(header, 'doinhdch', comment='adding a test comment')
        assert 'doinhdch' in header
        assert header.comments['doinhdch'] == 'adding a test comment'

    def test_writename(self):
        """Check we can add a keyword with a different name"""
        dripconfig.load()
        header = fits.header.Header()
        value = getpar(header, 'doinhdch', writename='WRITENAM')
        assert 'WRITENAM' in header
        assert header['WRITENAM'] == value

    def test_detchan(self):
        """Check detchan parses integer values and updates the header

        Ensure the comment is left untouched
        """
        dripconfig.load()
        header = fits.header.Header()
        header['DETCHAN'] = 'SW', 'A test comment'
        value = getpar(header, 'DETCHAN')
        assert value == 'SW'
        assert header['DETCHAN'] == 'SW'
        assert header.comments['DETCHAN'] == 'A test comment'

    def test_errors(self, capsys):
        header = fits.header.Header()

        # test bad parname
        value = getpar(header, ['BADVAL'], default='test')
        capt = capsys.readouterr()
        assert 'invalid parname' in capt.err
        assert value == 'test'

        # test bad header
        value = getpar(['BADVAL'], 'TESTKEY', default='test')
        capt = capsys.readouterr()
        assert 'invalid header' in capt.err
        assert value == 'test'

        # test bad dtype
        header['TESTKEY'] = 'test string value'
        value = getpar(header, 'TESTKEY', dtype=int, default=1)
        capt = capsys.readouterr()
        assert 'could not convert' in capt.err.lower()
        assert value is None

    def test_no_dripconf(self):
        dripconfig.load()
        dripconfig.configuration['TESTVAL'] = 'from config'
        header = fits.header.Header()
        header['TESTVAL'] = 'from header'
        value = getpar(header, 'TESTVAL', dripconf=False)
        assert value == 'from header'
        value = getpar(header, 'TESTVAL', dripconf=True)
        assert value == 'from config'

    def test_listval(self):
        header = fits.header.Header()
        dripconfig.load()

        listval = [1, 2, 3]
        testval = '[1,2,3]'
        dripconfig.configuration['TESTVAL'] = listval
        value = getpar(header, 'TESTVAL')
        assert value == testval

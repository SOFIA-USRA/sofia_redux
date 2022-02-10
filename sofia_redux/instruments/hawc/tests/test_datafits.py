# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data, pol_bgs_data


class TestDataFits(DRPTestCase):
    def test_load(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        # load first header only
        df = DataFits()
        df.loadhead(ffile)
        assert df.filename == ffile
        assert df.loaded is False
        assert isinstance(df.imgheads, list)
        assert len(df.imgheads) == 1
        assert df.imgheads[0]['EXTNAME'].upper() == 'PRIMARY'
        assert len(df.tabnames) == 0

        # load all headers
        df.loadhead(ffile, dataname='all')
        assert isinstance(df.imgheads, list)
        assert len(df.imgheads) == 3

        # load the timestream header
        df.loadhead(ffile, dataname='TIMESTREAM')
        assert isinstance(df.imgheads, list)
        assert len(df.imgheads) == 1
        assert df.imgheads[0]['EXTNAME'].upper() == 'TIMESTREAM'

        # load data
        df.load()
        assert len(df.tabnames) == 1
        assert df.loaded is True

        # load from hdul should be identical
        df2 = DataFits()
        df2.loadhead(hdul=hdul)
        assert fits.HeaderDiff(df2.header, df.header).identical
        df2.loadhead(hdul=hdul, dataname='all')
        for i in range(len(df.imgheads)):
            assert fits.HeaderDiff(df2.imgheads[i], df.imgheads[i]).identical
        df2.load(hdul=hdul)
        assert fits.TableDataDiff(df2.table, df.table).identical

    def test_load_errors(self, capsys, tmpdir, mocker):
        # non existent file
        badfile = 'badfile.fits'
        with pytest.raises(IOError):
            DataFits(badfile)
        capt = capsys.readouterr()
        assert 'No such file' in capt.err

        # file with image extensions, no extnames, no date
        imhdul = pol_bgs_data()
        for hdu in imhdul:
            del hdu.header['EXTNAME']
            try:
                del hdu.header['DATE-OBS']
            except KeyError:
                pass
        ffile = str(tmpdir.join('test.fits'))
        imhdul.writeto(ffile, overwrite=True)

        # load
        df = DataFits()
        df.load(ffile)
        assert df.imgnames[0] == 'PRIMARY IMAGE'
        assert df.imgnames[1] == 'SECONDARY IMAGE 1'
        assert len(df.config_files) == 1

        # mock getheadval to raise KeyError for NAXIS1
        def mock_get_hval(obj, key, **kwargs):
            if key == 'NAXIS1':
                raise KeyError('no NAXIS1')
            else:
                return obj.header[key]
        mocker.patch.object(DataFits, 'getheadval', mock_get_hval)

        df = DataFits()
        df.load(ffile)
        capt = capsys.readouterr()
        assert 'missing naxis' in capt.err

        # file with primary image, 2 tables, no extnames
        tabhdul = pol_raw_data()
        hdul = fits.HDUList([imhdul[0], tabhdul[2], tabhdul[2]])
        for hdu in hdul:
            try:
                del hdu.header['EXTNAME']
            except KeyError:
                pass
        hdul.writeto(ffile, overwrite=True)

        df = DataFits()
        df.load(ffile)
        assert len(df.tabnames) == 2
        assert df.tabnames[0] == 'PRIMARY TABLE'
        assert df.tabnames[1] == 'SECONDARY TABLE 1'

        # table with None data
        hdul[1].data = None
        hdul.writeto(ffile, overwrite=True)
        df = DataFits()
        df.load(ffile)
        capt = capsys.readouterr()
        assert 'Table in HDU number 1 has no data' in capt.err

        # table with bad data
        def mock_data():
            raise AttributeError('bad data')
        mocker.patch.object(fits.BinTableHDU, 'data', mock_data)
        df = DataFits()
        df.load(ffile)
        capt = capsys.readouterr()
        assert 'Problem loading table' in capt.err

    def test_getattr(self, tmpdir, capsys):
        # empty datafits
        df = DataFits()
        assert df.data is None
        assert df.image is None
        assert df.table is None
        assert df.header is None
        capt = capsys.readouterr()
        assert 'has no data' in capt.err
        assert 'has no image data' in capt.err
        assert 'has no table data' in capt.err
        assert 'has no header data' in capt.err

        # load a table-only file
        tbhdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        tbhdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        capsys.readouterr()

        # table data there
        assert df.table is df.tabdata[0]
        assert df.data is df.tabdata[0]

        # image data not there
        assert df.image is None
        capt = capsys.readouterr()
        assert 'has no image data' in capt.err

        # header is primary
        assert df.header is df.imgheads[0]

        # load a image-only file
        imhdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        imhdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        capsys.readouterr()

        # image data there
        assert df.image is df.imgdata[0]
        assert df.data is df.imgdata[0]

        # table data not there
        assert df.table is None
        capt = capsys.readouterr()
        assert 'has no table data' in capt.err

        # header is primary
        assert df.header is df.imgheads[0]

        # get filenamebegin from parent getattr
        assert df.filenamebegin == os.path.splitext(ffile)[0] + '.'

    def test_setattr(self):
        # table and image data
        tbhdul = pol_raw_data()
        imhdul = pol_bgs_data()
        tab = tbhdul[2].data
        img = imhdul[0].data
        hdr = imhdul[0].header

        # test on empty datafits
        df = DataFits()
        assert len(df.tabdata) == 0
        assert len(df.imgdata) == 0
        assert not df.loaded

        # set a table to data
        df.data = tab
        assert len(df.tabdata) == 1
        assert len(df.imgdata) == 0
        assert df.loaded

        # set an image to data
        df = DataFits()
        df.data = img
        assert len(df.tabdata) == 0
        assert len(df.imgdata) == 1
        assert df.loaded

        # set an image to primary image
        df = DataFits()
        df.image = img
        assert len(df.tabdata) == 0
        assert len(df.imgdata) == 1
        assert df.imgnames[0] == 'PRIMARY IMAGE'
        assert isinstance(df.imgheads[0], fits.Header)
        assert len(df.imgheads[0]) == 0
        assert df.loaded

        # reset the image; header should be unaffected
        df.header['TESTKEY'] = 'TESTVAL'
        df.image = img * 10.0
        assert len(df.imgdata) == 1
        assert len(df.imgheads[0]) == 1
        assert df.header['TESTKEY'] == 'TESTVAL'

        # set a table to primary table
        df = DataFits()
        df.table = tab
        assert len(df.tabdata) == 1
        assert len(df.imgdata) == 0
        assert df.tabnames[0] == 'PRIMARY TABLE'
        assert isinstance(df.tabheads[0], fits.Header)
        assert len(df.tabheads[0]) == 0
        assert df.loaded

        # reset the table; header should be unaffected
        # note -- table can't be primary HDU, so df.header is still None
        df.tabheads[0]['TESTKEY'] = 'TESTVAL'
        df.table = tab.copy()
        assert len(df.tabdata) == 1
        assert len(df.tabheads[0]) == 1
        assert df.tabheads[0]['TESTKEY'] == 'TESTVAL'

        # set a header to header
        df = DataFits()
        df.header = hdr
        assert len(df.tabdata) == 0
        assert len(df.imgdata) == 1
        assert df.imgdata[0] is None
        assert df.imgheads[0] is hdr
        assert not df.loaded

        # update header -- data is not touched
        hdr2 = tbhdul[0].header
        df.header = hdr2
        assert df.imgheads[0] is hdr2
        assert df.imgdata[0] is None

    def test_loadhead_errors(self, tmpdir, capsys):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        # missing file entirely
        df = DataFits()
        with pytest.raises(ValueError):
            df.loadhead()
        capt = capsys.readouterr()
        assert 'FITS read error' in capt.err

        # bad file name
        df = DataFits()
        with pytest.raises(IOError):
            df.loadhead('badfile.fits')
        capt = capsys.readouterr()
        assert 'FITS read error' in capt.err

        # good file, bad extension name
        df = DataFits()
        with pytest.raises(ValueError):
            df.loadhead(ffile, dataname='BADVAL')
        capt = capsys.readouterr()
        assert 'HDU with EXTNAME=BADVAL not found' in capt.err

        # good extension name
        df.loadhead(ffile, dataname='Configuration')
        assert df.header == hdul[1].header

        # missing extension name
        df = DataFits()
        del hdul[1].header['EXTNAME']
        hdul.writeto(ffile, overwrite=True)
        df.loadhead(ffile, dataname=1)
        assert df.imgnames[0] == 'PRIMARY HEADER'

    def test_hdulist(self, tmpdir):
        # test a file with image extensions
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        new_hdul = df.to_hdulist()
        new_keys = ['PIPEVERS', 'FILENAME', 'DATE', 'EXTNAME']
        for i, hdu in enumerate(new_hdul):
            hd = fits.diff.HeaderDiff(hdu.header, hdul[i].header,
                                      ignore_keywords=new_keys)
            assert len(hd.diff_keywords) == 0
            assert len(hd.diff_keyword_values) == 0

            assert np.allclose(hdu.data, hdul[i].data, equal_nan=True)

        for key in new_keys:
            assert key in new_hdul[0].header

        # test a file with only a table
        hdul = pol_raw_data()
        hdul = fits.HDUList([hdul[0], hdul[2]])
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        new_hdul = df.to_hdulist()
        assert len(new_hdul) == 2
        new_keys = ['PIPEVERS', 'FILENAME', 'DATE', 'EXTNAME',
                    'NAXIS', 'NAXIS1', 'BITPIX']
        for i, hdu in enumerate(new_hdul):
            hd = fits.diff.HeaderDiff(hdu.header, hdul[i].header,
                                      ignore_keywords=new_keys)
            assert len(hd.diff_keywords) == 0
            assert len(hd.diff_keyword_values) == 0
            if isinstance(hdu, fits.BinTableHDU):
                dd = fits.diff.TableDataDiff(hdu.data, hdul[i].data)
                assert dd.identical

        # same file but don't save the table
        new_hdul = df.to_hdulist(save_tables=False)
        assert len(new_hdul) == 1
        assert not isinstance(new_hdul[0], fits.BinTableHDU)

        # test a file with no primary HDU -- a simple one will be added
        df = DataFits()
        df.tabnames = ['Table 1']
        df.tabdata = [hdul[1].data]
        df.tabheads = [fits.Header()]
        new_hdul = df.to_hdulist()
        assert len(new_hdul) == 2
        assert isinstance(new_hdul[0], fits.PrimaryHDU)

    def test_header_list(self, tmpdir):
        # data with images and tables
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        hlist = df.to_header_list()
        assert len(hlist) == len(hdul)

        new_keys = ['PIPEVERS', 'FILENAME', 'DATE']
        for i, hdr in enumerate(hlist):
            hd = fits.diff.HeaderDiff(hdr, hdul[i].header,
                                      ignore_keywords=new_keys)
            assert len(hd.diff_keywords) == 0
            assert len(hd.diff_keyword_values) == 0

    def test_save(self, tmpdir):
        # data with images
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        new_name = str(tmpdir.join('test1.fits'))
        df.filename = new_name
        df.save()
        assert os.path.isfile(new_name)

        new_name = str(tmpdir.join('test2.fits'))
        df.save(new_name)
        assert os.path.isfile(new_name)

        # bad filename type
        with pytest.raises(TypeError):
            df.save(20)

        # bad file directory
        with pytest.raises(IOError):
            df.save('/bad/path/for/data.fits')

    def test_copy(self):
        df = DataFits()
        df.image = np.arange(10)
        df.table = np.array([10, 10], dtype=[('x', int)])

        new_df = df.copy()

        # names match
        assert new_df.imgnames == df.imgnames
        assert new_df.tabnames == df.tabnames
        assert new_df.filename == df.filename
        assert new_df.rawname == df.rawname
        assert new_df.loaded == df.loaded

        # data matches, but is not the same object
        assert new_df.imgdata[0] is not df.imgdata[0]
        assert np.all(new_df.imgdata[0] == df.imgdata[0])
        assert new_df.tabdata[0] is not df.tabdata[0]
        assert np.all(new_df.tabdata[0] == df.tabdata[0])

        # same for headers
        assert new_df.imgheads[0] is not df.imgheads[0]
        assert np.all(new_df.imgheads[0] == df.imgheads[0])
        assert new_df.tabheads[0] is not df.tabheads[0]
        assert np.all(new_df.tabheads[0] == df.tabheads[0])

        # table only
        df = DataFits()
        df.image = None
        df.table = np.array([10, 10], dtype=[('x', int)])

        new_df = df.copy()
        assert new_df.imgdata[0] is None
        assert new_df.tabdata[0] is not df.tabdata[0]
        assert np.all(new_df.tabdata[0] == df.tabdata[0])

    def test_mergehead(self):
        # make a config for test value merging
        combo = ['FIRST', 'LAST', 'MIN', 'MAX', 'SUM',
                 'OR', 'AND', 'CONCATENATE', 'DEFAULT']
        keys = ['TEST{}'.format(i + 1) for i in range(len(combo))]
        config = {'headmerge': dict(zip(keys, combo))}

        # set up a couple test headers
        val1 = list(range(len(combo)))
        hdr1 = fits.Header(dict(zip(keys, val1)))
        hdr1['COMMENT'] = 'comment 1'
        hdr1['HISTORY'] = 'history 1'
        h1_copy = hdr1.copy()

        val2 = [v + 10 for v in val1]
        hdr2 = fits.Header(dict(zip(keys, val2)))
        hdr2['COMMENT'] = 'comment 2'
        hdr2['HISTORY'] = 'history 2'
        h2_copy = hdr2.copy()

        # make the datafits, load config and header
        df1 = DataFits()
        df1.config = config
        df1.header = hdr1
        df2 = DataFits()
        df2.config = config
        df2.header = hdr2

        # merge all keys
        df1.mergehead(df2)
        new_hdr = df1.header

        # check values
        # first
        assert new_hdr['TEST1'] == val1[0]
        # last
        assert new_hdr['TEST2'] == val2[1]
        # min
        assert new_hdr['TEST3'] == val1[2]
        # max
        assert new_hdr['TEST4'] == val2[3]
        # sum
        assert new_hdr['TEST5'] == val1[4] + val2[4]
        # or
        assert new_hdr['TEST6'] == val1[5] | val2[5]
        # and
        assert new_hdr['TEST7'] == val1[6] & val2[6]
        # concatenate
        assert new_hdr['TEST8'] == ','.join(sorted([str(val1[7]),
                                                    str(val2[7])]))
        # default
        assert new_hdr['TEST9'] == -9999

        # all history in new header; only original comments
        assert 'comment 1' in str(new_hdr['COMMENT'])
        assert 'comment 2' not in str(new_hdr['COMMENT'])
        assert 'history 1' in str(new_hdr['HISTORY'])
        assert 'history 2' in str(new_hdr['HISTORY'])

        # test concatenation with multiple values, some overlapping
        df1.header['TEST8'] = 'q,a,r,c'
        df2.header['TEST8'] = 'b,d,c'
        df1.mergehead(df2)
        assert df1.header['TEST8'] == 'a,b,c,d,q,r'

        # test all default values
        df1.header['TEST9'] = 10
        df1.mergehead(df2)
        assert type(df1.header['TEST9']) is int
        assert df1.header['TEST9'] == -9999

        df1.header['TEST9'] = 10.0
        df1.mergehead(df2)
        assert type(df1.header['TEST9']) is float
        assert df1.header['TEST9'] == -9999.0

        df1.header['TEST9'] = 'ten'
        df1.mergehead(df2)
        assert type(df1.header['TEST9']) is str
        assert df1.header['TEST9'] == 'UNKNOWN'

        # test copyhead
        df1.header = h1_copy.copy()
        df2.header = h2_copy.copy()
        df1.copyhead(df2)

        # all keys match h2
        for key in keys:
            assert df1.header[key] == h2_copy[key]

        # all history and comments present
        assert 'comment 1' in str(df1.header['COMMENT'])
        assert 'comment 2' in str(df1.header['COMMENT'])
        assert 'history 1' in str(df1.header['HISTORY'])
        assert 'history 2' in str(df1.header['HISTORY'])

        # copy to new extname
        # also do without history, comment in h2, and add a new
        # key to h2
        df1 = DataFits()
        df2 = DataFits()
        h1 = h1_copy.copy()
        h2 = h2_copy.copy()
        h2['NEWKEY'] = 'TESTVAL'
        del h2['HISTORY']
        del h2['COMMENT']
        df1.imageset(np.zeros(10), imagename='PRIMARY',
                     imageheader=h1)
        df1.imageset(np.zeros(10), imagename='NEWNAME',
                     imageheader=h1)
        df2.imageset(np.zeros(10), imagename='PRIMARY',
                     imageheader=h2)
        df2.imageset(np.zeros(10), imagename='NEWNAME',
                     imageheader=h2)
        df1.copyhead(df2, name='NEWNAME')

        for key in keys:
            assert df1.header[key] == h1_copy[key]
            assert df1.getheader('NEWNAME')[key] == h2_copy[key]

        assert df1.getheader('NEWNAME')['NEWKEY'] == h2['NEWKEY']

    def test_copydata(self):
        df1 = DataFits()
        df2 = DataFits()
        h1 = fits.Header({'TEST1': 'VAL1'})
        h2 = fits.Header({'TEST2': 'VAL2'})
        df1.imageset(np.arange(10), imagename='PRIMARY',
                     imageheader=h1)
        df1.imageset(np.arange(10) + 10, imagename='IMGNAME',
                     imageheader=h1)
        df1.tableset(np.array([10, 20], dtype=[('x', int)]),
                     tablename='TABNAME',
                     tableheader=h1)

        df2.imageset(np.arange(10) + 20, imagename='PRIMARY',
                     imageheader=h2)
        df2.imageset(np.arange(10) + 30, imagename='OTHERNAME',
                     imageheader=h2)
        df2.tableset(np.array([20, 30], dtype=[('x', int)]),
                     tablename='TABNAME',
                     tableheader=h1)
        df2.tableset(np.array([40, 50], dtype=[('x', int)]),
                     tablename='IMGNAME',
                     tableheader=h1)

        # copy image from df2 to df1: overwrite
        df1.copydata(df2, 'PRIMARY')
        assert df1.getheader('PRIMARY') == df2.getheader('PRIMARY')
        assert np.all(df1.imageget('PRIMARY') == df2.imageget('PRIMARY'))
        assert 'IMGNAME' in df1.imgnames
        assert 'OTHERNAME' not in df1.imgnames

        # copy image: add new
        df1.copydata(df2, 'OTHERNAME')
        assert df1.getheader('OTHERNAME') == df2.getheader('OTHERNAME')
        assert np.all(df1.imageget('OTHERNAME') == df2.imageget('OTHERNAME'))
        assert 'IMGNAME' in df1.imgnames

        # copy table: overwrite
        df1.copydata(df2, 'TABNAME')
        assert np.all(df1.tableget('TABNAME') == df2.tableget('TABNAME'))

        # try to copy missing name
        with pytest.raises(ValueError):
            df1.copydata(df2, 'BADNAME')

        # copy a table to a former image name
        df1.copydata(df2, 'IMGNAME')
        assert np.all(df1.tableget('IMGNAME') == df2.tableget('IMGNAME'))
        assert 'IMGNAME' not in df1.imgnames

        # copy an image to a former table name
        df2.tabledel('TABNAME')
        df2.imageset(np.arange(10) + 30, imagename='TABNAME')
        df1.copydata(df2, 'TABNAME')
        assert np.all(df1.imageget('TABNAME') == df2.imageget('TABNAME'))
        assert 'TABNAME' not in df1.tabnames

    def test_index(self):
        df = DataFits()
        df.imageset(np.arange(10), imagename='IMG1')
        df.imageset(np.arange(10), imagename='IMG2')
        df.tableset(np.array([10, 20], dtype=[('x', int)]),
                    tablename='TAB1')
        df.tableset(np.array([10, 20], dtype=[('x', int)]),
                    tablename='TAB2')

        # test image index
        assert df.imageindex() == 0
        assert df.imageindex('IMG1') == 0
        assert df.imageindex('IMG2') == 1
        with pytest.raises(ValueError):
            df.imageindex('TAB1')

        # test table index
        assert df.tableindex() == 0
        assert df.tableindex('TAB1') == 0
        assert df.tableindex('TAB2') == 1
        with pytest.raises(ValueError):
            df.tableindex('IMG1')

        # table index with no tables
        df.tabledel('TAB1')
        df.tabledel('TAB2')
        with pytest.raises(RuntimeError):
            df.tableindex('TAB1')

    def test_imageset(self):
        data = np.arange(10)
        header = fits.Header({'TEST1': 'VAL1'})

        # insert primary, no name
        df = DataFits()
        df.imageset(data)
        assert df.imgnames[0] == 'PRIMARY'
        assert np.all(df.imgdata[0] == data)

        # reset same extension to new data
        df.imageset(data + 10)
        assert np.all(df.imgdata[0] == data + 10)

        # insert primary, with name
        df = DataFits()
        df.header = header
        df.imageset(data, imagename='TEST')
        assert df.imgnames[0] == 'TEST'
        assert np.all(df.imgdata[0] == data)

    def test_imagedel(self):
        data = np.arange(10)
        df = DataFits()
        df.imageset(data)
        df.imageset(data, imagename='TEST1')
        df.imageset(data, imagename='TEST2')

        assert len(df.imgnames) == 3
        assert len(df.imgdata) == 3
        assert len(df.imgheads) == 3

        # delete non-primary
        df.imagedel('TEST1')
        assert len(df.imgnames) == 2
        assert len(df.imgdata) == 2
        assert len(df.imgheads) == 2
        assert df.imgnames[0] == 'PRIMARY'

        # delete primary
        df.imagedel('PRIMARY')
        assert len(df.imgnames) == 1
        assert len(df.imgdata) == 1
        assert len(df.imgheads) == 1
        assert df.imgnames[0] == 'TEST2'

    def test_tableset(self):
        data = np.array([10, 10], dtype=[('x', int)])
        data2 = np.array([20, 20], dtype=[('x', int)])
        header = fits.Header({'TEST1': 'VAL1'})

        # insert primary, no name
        df = DataFits()
        df.tableset(data)
        assert df.tabnames[0] == 'PRIMARY TABLE'
        assert np.all(df.tabdata[0] == data)

        # reset same extension to new data
        df.tableset(data2)
        assert np.all(df.tabdata[0] == data2)

        # insert primary, with name
        df = DataFits()
        df.header = header
        df.tableset(data, tablename='TEST')
        assert df.tabnames[0] == 'TEST'
        assert np.all(df.tabdata[0] == data)

    def test_tabledel(self):
        data = np.array([10, 10], dtype=[('x', int)])
        df = DataFits()
        df.tableset(data)
        df.tableset(data, tablename='TEST1')
        df.tableset(data, tablename='TEST2')

        assert len(df.tabnames) == 3
        assert len(df.tabdata) == 3
        assert len(df.tabheads) == 3

        # delete non-primary
        df.tabledel('TEST1')
        assert len(df.tabnames) == 2
        assert len(df.tabdata) == 2
        assert len(df.tabheads) == 2
        assert df.tabnames[0] == 'PRIMARY TABLE'

        # delete primary
        df.tabledel('PRIMARY TABLE')
        assert len(df.tabnames) == 1
        assert len(df.tabdata) == 1
        assert len(df.tabheads) == 1
        assert df.tabnames[0] == 'TEST2'

    def test_tablecol(self):
        data = np.array([10, 10], dtype=[('x', int)])
        df = DataFits()
        df.tableset(data)

        # add a column
        df.tableaddcol('y', [20, 20])
        assert np.all(df.table['y'] == 20)

        # add an array
        df.tableaddcol('z', [[30, 30], [30, 30]])
        assert np.all(df.table['z'] == 30)
        assert df.table['z'].shape == (2, 2)

        # add a new column when table does not yet exist
        df = DataFits()
        df.tableaddcol('x', [10, 10])
        assert np.all(df.table['x'] == 10)
        assert df.tabnames[0] == 'TABLE'

        # specify the table name
        df.tableaddcol('y', [20, 20], tablename='Table')
        assert np.all(df.tableget('TABLE')['y'] == 20)

        # try to add column with too many rows
        with pytest.raises(ValueError):
            df.tableaddcol('z', [30, 30, 30])

        # delete a column
        df.tabledelcol('y')
        assert 'y' not in df.tableget('TABLE').dtype.names

        # try to delete an invalid column
        with pytest.raises(ValueError):
            df.tabledelcol('z')

        # try to pass a bad data type for the column name
        with pytest.raises(ValueError):
            df.tabledelcol({'name': 'z'})

        # delete all columns -- table is deleted
        assert df.tabnames[0] == 'TABLE'
        df.tabledelcol('x')
        assert len(df.tabnames) == 0
        assert len(df.tabdata) == 0
        assert len(df.tabheads) == 0

        # set a fits record rather than a numpy record
        df = DataFits()
        data = pol_raw_data()[2].data
        df.tableset(data)
        assert 'RA' in df.table.names
        df.tabledelcol('RA')
        assert 'RA' not in df.table.names

    def test_tablerow(self):
        data1 = np.array([10, 10], dtype=[('x', int)])
        data2 = np.array([(10, 20), (10, 20)],
                         dtype=[('x', int), ('y', int)])

        # add a row - 1 column
        df = DataFits()
        df.tableset(data1)
        df.tableaddrow([20])

        # add a row - 2 columns
        df = DataFits()
        df.tableset(data2)
        assert len(df.table) == 2
        with pytest.raises(ValueError):
            df.tableaddrow([10])
        with pytest.raises(ValueError):
            df.tableaddrow([10, 'a'])
        df.tableaddrow([10, 20])
        assert len(df.table) == 3
        assert np.all(df.table['x'] == 10)
        assert np.all(df.table['y'] == 20)

        # delete a row
        df.tabledelrow(-1)
        assert len(df.table) == 2

        # try to delete a bad index
        with pytest.raises(ValueError):
            df.tabledelrow(2)

        # delete all rows
        df.tabledelrow(-1, tablename='PRIMARY TABLE')
        assert len(df.table) == 1
        assert len(df.tabnames) == 1
        df.tabledelrow(-1)
        assert len(df.tabnames) == 0
        assert len(df.tabdata) == 0
        assert len(df.tabheads) == 0

    def test_mergerows(self, capsys):
        data1 = np.array([(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)],
                         dtype=[('x', int), ('y', int)])
        data2 = np.array([('a', 20), ('b', 30), ('c', 40)],
                         dtype=[('p', str), ('q', int)])
        data3 = np.array([(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)],
                         dtype=[('x', int), ('Samples', int)])
        data4 = np.array([(10, -1), (20, -1), (30, -1), (40, -1), (50, -1)],
                         dtype=[('x', int), ('Samples', int)])

        df = DataFits()

        # try bad row type
        with pytest.raises(AttributeError):
            df.tablemergerows([1, 2, 3])

        # unconfigured -- returns first, with warning
        inrows = data1[1:3]
        outrow = df.tablemergerows(inrows)
        assert outrow == inrows[0]
        capt = capsys.readouterr()
        assert 'Missing table merge entry' in capt.err

        # try config with mismatched function - returns first, with warning
        config = {'table': {'p': 'avg'}}
        df.config = config
        outrow = df.tablemergerows(data2)
        assert outrow == data2[0]
        capt = capsys.readouterr()
        assert 'Error in avg( p ) - returning first' in capt.err

        # try config with unknown function -- returns first with warning
        config = {'table': {'p': 'badval'}}
        df.config = config
        outrow = df.tablemergerows(data2)
        assert outrow == data2[0]
        capt = capsys.readouterr()
        assert 'Unknown operation' in capt.err

        # test all operations
        df.config = {'table': {'x': 'first', 'y': 'last'}}
        outrow = df.tablemergerows(data1)
        assert outrow['x'] == data1['x'][0]
        assert outrow['y'] == data1['y'][-1]

        df.config = {'table': {'x': 'min', 'y': 'max'}}
        outrow = df.tablemergerows(data1)
        assert outrow['x'] == 10
        assert outrow['y'] == 60

        df.config = {'table': {'x': 'med', 'y': 'avg'}}
        outrow = df.tablemergerows(data1)
        assert outrow['x'] == 30.0
        assert outrow['y'] == 40.0

        # weighted average with good samples
        df.config = {'table': {'x': 'wtavg', 'samples': 'sum'}}
        outrow = df.tablemergerows(data3)
        assert outrow['x'] == 35
        assert outrow['Samples'] == 200

        # weighted average with bad samples
        df.config = {'table': {'x': 'wtavg', 'samples': 'sum'}}
        outrow = df.tablemergerows(data4)
        assert outrow['x'] == 0.0
        assert outrow['Samples'] == -5

    def test_mergetables(self, capsys):
        # numpy record -- not supported
        data1 = np.array([(10, 20), (20, 30), (30, 40)],
                         dtype=[('x', int), ('y', int)])
        # FITS table
        data2 = pol_raw_data()[2].data

        # mismatched FITS table
        names = data2.names[0:5:2]
        new_hdu = fits.BinTableHDU.from_columns(data2.columns[0:5:2])
        data3 = new_hdu.data

        # matching, larger values
        data4 = data3.copy()
        for name in names:
            data4[name] += 10

        df = DataFits()

        # try to merge numpy tables
        with pytest.raises(ValueError):
            df.tablemergetables([data1])
        capt = capsys.readouterr()
        assert 'only available for FITS tables' in capt.err

        # try to merge a single FITS table -- returns first
        outhdu = df.tablemergetables([data2])
        assert np.all(outhdu.data == data2)

        # try to merge mismatched tables
        with pytest.raises(ValueError):
            df.tablemergetables([data2, data3])
        capt = capsys.readouterr()
        assert 'columns differ' in capt.err

        # merge matched tables, unconfigured
        df.config = {}
        outhdu = df.tablemergetables([data2, data2])
        assert np.all(outhdu.data == data2)
        capt = capsys.readouterr()
        assert 'Missing table merge entry' in capt.err

        # try unknown operation -- returns first
        df.config = {'table': {}}
        for name in names:
            df.config['table'][name.lower()] = 'badval'
        outhdu = df.tablemergetables([data3, data4])
        assert np.all(outhdu.data == data3)
        capt = capsys.readouterr()
        assert 'Unknown operation' in capt.err

        # test all functions

        # first
        for name in names:
            df.config['table'][name.lower()] = 'first'
        outhdu = df.tablemergetables([data3, data4])
        assert np.all(outhdu.data == data3)
        # last
        for name in names:
            df.config['table'][name.lower()] = 'last'
        outhdu = df.tablemergetables([data3, data4])
        assert np.all(outhdu.data == data4)
        # min
        for name in names:
            df.config['table'][name.lower()] = 'min'
        outhdu = df.tablemergetables([data3, data4])
        assert np.all(outhdu.data == data3)
        # max
        for name in names:
            df.config['table'][name.lower()] = 'max'
        outhdu = df.tablemergetables([data3, data4])
        assert np.all(outhdu.data == data4)
        # med
        val = []
        for name in names:
            df.config['table'][name.lower()] = 'med'
            val.append(np.median([data3[name], data4[name]], axis=0))
        outhdu = df.tablemergetables([data3, data4])
        for i, name in enumerate(names):
            assert np.all(outhdu.data[name] == np.array(val[i]))
        # avg
        val = []
        for name in names:
            df.config['table'][name.lower()] = 'avg'
            val.append(np.mean([data3[name], data4[name]], axis=0))
        outhdu = df.tablemergetables([data3, data4])
        for i, name in enumerate(names):
            assert np.all(outhdu.data[name] == np.array(val[i]))
        # sum
        val = []
        for name in names:
            df.config['table'][name.lower()] = 'sum'
            val.append(np.sum([data3[name], data4[name]], axis=0))
        outhdu = df.tablemergetables([data3, data4])
        for i, name in enumerate(names):
            assert np.all(outhdu.data[name] == np.array(val[i]))

        # wtavg, no samples
        capsys.readouterr()
        for name in names:
            df.config['table'][name.lower()] = 'wtavg'
        outhdu = df.tablemergetables([data3, data4])
        assert np.all(outhdu.data == data3)
        capt = capsys.readouterr()
        assert 'Error in wtavg' in capt.err
        assert 'returning first' in capt.err

        # wtavg, with samples
        data3.columns[1].name = 'Samples'
        data4.columns[1].name = 'Samples'
        data3['Samples'] += 10
        names[1] = 'Samples'
        val = []
        for i, name in enumerate(names):
            if i == 2:
                df.config['table'][name.lower()] = 'wtavg'
                v1 = np.sum([data3[name] * data3['Samples'],
                             data4[name] * data4['Samples']], axis=0)
                v2 = np.sum([data3['Samples'],
                             data4['Samples']], axis=0)
                val.append(v1 / v2)
            else:
                df.config['table'][name.lower()] = 'sum'
                val.append(np.sum([data3[name], data4[name]], axis=0))
        outhdu = df.tablemergetables([data3, data4])
        for i, name in enumerate(names):
            assert np.allclose(outhdu.data[name], np.array(val[i]))

    def test_getheader_error(self, capsys):
        # usual functionality is exercised in other tests;
        # just check error/corner cases here
        df = DataFits()

        # no header, no data name
        hdr = df.getheader()
        assert hdr is None

        # bad data name
        with pytest.raises(ValueError):
            df.getheader('badval')
        capt = capsys.readouterr()
        assert 'Invalid data name' in capt.err

    def test_setheader(self, capsys):
        df = DataFits()
        df.imageset(np.arange(100).reshape(10, 10),
                    imagename='IMG1')
        df.imageset(np.arange(100).reshape(10, 10),
                    imagename='IMG2')
        df.tableset(np.array([10, 10], dtype=[('x', int)]),
                    tablename='TAB1')

        hdr1 = fits.Header({'TESTKEY': 'TESTVAL1'})
        hdr2 = fits.Header({'TESTKEY': 'TESTVAL2'})

        # no dataname -- set to primary
        df.setheader(hdr1)
        assert df.header['TESTKEY'] == 'TESTVAL1'
        assert df.getheader('IMG1')['TESTKEY'] == 'TESTVAL1'
        assert 'TESTKEY' not in df.getheader('IMG2')

        # with img dataname
        df.setheader(hdr2, dataname='IMG2')
        assert df.header['TESTKEY'] == 'TESTVAL1'
        assert df.getheader('IMG1')['TESTKEY'] == 'TESTVAL1'
        assert df.getheader('IMG2')['TESTKEY'] == 'TESTVAL2'

        # with table dataname
        df.setheader(hdr2, dataname='TAB1')
        assert df.header['TESTKEY'] == 'TESTVAL1'
        assert df.getheader('TAB1')['TESTKEY'] == 'TESTVAL2'

        # with invalid data name
        with pytest.raises(ValueError):
            df.setheader(hdr1, dataname='badval')
        capt = capsys.readouterr()
        assert 'Invalid data name' in capt.err

    def test_getheadval(self, capsys):
        df = DataFits()
        hdr1 = fits.Header({'TESTKEY': 'TESTVAL1', 'ALTKEY': 'ALTVAL'})
        hdr2 = fits.Header({'TESTKEY': 'TESTVAL2'})
        df.imageset(np.arange(100).reshape(10, 10),
                    imagename='IMG1', imageheader=hdr1)
        df.imageset(np.arange(100).reshape(10, 10),
                    imagename='IMG2', imageheader=hdr2)

        # get from primary
        assert df.getheadval('TESTKEY') == 'TESTVAL1'

        # get from dataname
        assert df.getheadval('TESTKEY', dataname='IMG2') == 'TESTVAL2'

        # get from config: alternate key
        df.config = {'header': {'TESTKEY': 'ALTKEY'}}
        assert df.getheadval('TESTKEY') == 'ALTVAL'
        with pytest.raises(KeyError):
            df.getheadval('TESTKEY', dataname='IMG2')
        capt = capsys.readouterr()
        assert 'Missing ALTKEY keyword in header IMG2' in capt.err

        # get from config: optional alternate - used only if first
        # key not present in header
        df.config = {'header': {'TESTKEY': '? ALTKEY',
                                'TESTKEY2': '? ALTKEY'}}
        assert df.getheadval('TESTKEY') == 'TESTVAL1'
        assert df.getheadval('TESTKEY2') == 'ALTVAL'

        # get from config: direct value
        # -- returned and set in header
        df.config = {'header': {'TESTKEY': "'CONFVAL' / Test Key"}}
        assert df.getheadval('TESTKEY') == 'CONFVAL'
        assert df.header['TESTKEY'] == 'CONFVAL'
        assert df.getheadval('TESTKEY', dataname='IMG2') == 'CONFVAL'
        assert df.getheader('IMG2')['TESTKEY'] == 'CONFVAL'

        # bad config
        df.config = None
        df.getheadval('TESTKEY')
        capt = capsys.readouterr()
        assert 'Missing Configuration' in capt.err

        # undefined value for keyword
        df.header['UNDEFKEY'] = fits.Undefined()
        assert df.getheadval('UNDEFKEY') == ''
        capt = capsys.readouterr()
        assert 'Missing value for key = UNDEFKEY' in capt.err

    def test_setheadval(self, capsys):
        # set with no imageheader
        df = DataFits()
        df.setheadval('TESTKEY', 'NOHEAD')
        assert df.header['TESTKEY'] == 'NOHEAD'
        assert df.imgnames[0] == 'PRIMARY'
        assert df.imgdata[0] is None

        # set with dataname not present
        df = DataFits()
        with pytest.raises(ValueError):
            df.setheadval('TESTKEY', 'NOHEAD', dataname='BADVAL')
        capt = capsys.readouterr()
        assert 'Invalid data name' in capt.err

        # set an image and a table
        df = DataFits()
        hdr1 = fits.Header({'TESTKEY': 'TESTVAL1', 'ALTKEY': 'ALTVAL'})
        df.imageset(np.arange(100).reshape(10, 10),
                    imagename='IMG1', imageheader=hdr1)
        df.tableset(np.array([10, 20], dtype=[('x', int)]),
                    tablename='TAB1')

        df.setheadval('TESTKEY', 'IMGVAL', dataname='IMG1')
        assert df.getheader('IMG1')['TESTKEY'] == 'IMGVAL'

        df.setheadval('TESTKEY', 'TABVAL', dataname='TAB1')
        assert df.getheader('TAB1')['TESTKEY'] == 'TABVAL'

    def test_delheadval(self, capsys):
        df = DataFits()
        hdr1 = fits.Header({'TESTKEY': 'TESTVAL1',
                            'ALTKEY': 'ALTVAL',
                            'EXTRAKEY': 'EXTRAVAL'})
        hdr2 = fits.Header({'TESTKEY': 'TESTVAL2'})
        df.imageset(np.arange(100).reshape(10, 10),
                    imagename='IMG1', imageheader=hdr1)
        df.imageset(np.arange(100).reshape(10, 10),
                    imagename='IMG2', imageheader=hdr2)

        # delete one key
        df.delheadval('EXTRAKEY')
        assert 'EXTRAKEY' not in df.getheader('IMG1')
        df.delheadval('TESTKEY', dataname='IMG2')
        assert 'TESTKEY' in df.getheader('IMG1')
        assert 'TESTKEY' not in df.getheader('IMG2')

        # delete a couple keys, may or may not be present
        df.delheadval(['TESTKEY', 'EXTRAKEY'])
        assert 'TESTKEY' not in df.getheader('IMG1')
        assert 'EXTRAKEY' not in df.getheader('IMG1')

        # try to pass some other data type
        with pytest.raises(ValueError):
            df.delheadval({'keys': ['TESTKEY', 'ALTKEY']})
        capt = capsys.readouterr()
        assert 'Invalid key' in capt.err

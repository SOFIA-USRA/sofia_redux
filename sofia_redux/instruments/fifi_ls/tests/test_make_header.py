# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.time import Time
from astropy.io import fits
import numpy as np
import pandas
import pytest

from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, test_files
from sofia_redux.instruments.fifi_ls import make_header as u


@pytest.fixture()
def dummy_table():
    def make_dummy_table(key='TESTKEY', required=True, default='UNKNOWN',
                         dtype=str, combine='first', dmin=np.nan, dmax=np.nan,
                         enum=None, comment='testing key', value='foo'):
        columns = ['required', 'default', 'type', 'combine',
                   'min', 'max', 'enum', 'comment', 'value']
        index = [key]
        if enum is None:
            enum = []
        table = pandas.DataFrame(columns=columns, index=index)
        table = table.to_dict('index')
        table[key]['required'] = required
        table[key]['default'] = default
        table[key]['type'] = dtype
        table[key]['combine'] = combine
        table[key]['min'] = dmin
        table[key]['max'] = dmax
        table[key]['enum'] = enum
        table[key]['comment'] = comment
        table[key]['value'] = value

        return table

    return make_dummy_table


class TestMakeHeader(FIFITestCase):

    def test_order_single(self):
        h = fits.Header()
        h['TESTKEY'] = 'foo'
        headers = [h]
        result = u.order_headers(headers)
        assert len(result) == 2
        assert isinstance(result[0], fits.Header)
        assert len(result[1]) == 1
        assert result[0]['TESTKEY'] == 'foo'

    def test_order_multiple(self):
        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h['DATE-OBS'] = '2018-01-01T00:00:%02i' % (59 - idx)
            h['IDX'] = idx
        basehead, oheaders = u.order_headers(headers)
        assert basehead['IDX'] == (len(headers) - 1)
        assert oheaders[0]['IDX'] == (len(headers) - 1)
        assert oheaders[len(headers) - 1]['IDX'] == 0

    def test_order_nods(self):
        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h['DATE-OBS'] = '2018-01-01T00:00:%02i' % (59 - idx)
            h['IDX'] = idx
            h['NODSTYLE'] = 'ASYMMETRIC'
            h['NODBEAM'] = 'B' if (idx % 2) == 0 else 'A'
        basehead, oheaders = u.order_headers(headers)

        # basehead is earliest A nod
        assert basehead['IDX'] == 3

        # sorted headers are by date-obs
        assert oheaders[0]['IDX'] == 4
        assert oheaders[4]['IDX'] == 0

    def test_get_keyword_values(self, mocker):
        headers = [fits.Header()]
        table = u.get_keyword_values(headers[0], headers)
        assert isinstance(table, dict)
        expected_columns = ['required', 'default', 'type', 'combine',
                            'min', 'max', 'enum', 'comment', 'value']

        for row in table.values():
            for column in expected_columns:
                assert column in row

        with pytest.raises(ValueError) as err:
            u.get_keyword_values(headers[0], headers,
                                 default_file='__does_not_exist__')
        assert "could not create requirements table" in str(err.value).lower()

        with pytest.raises(ValueError) as err:
            u.get_keyword_values(headers[0], headers,
                                 comment_file='__does_not_exist__')
        assert "could not create requirements table" in str(err.value).lower()

        # mock create table to return None: will also raise error
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.make_header.'
            'create_requirements_table',
            return_value=None)
        with pytest.raises(ValueError) as err:
            u.get_keyword_values(headers[0], headers,
                                 comment_file='__does_not_exist__')
        assert "could not create requirements table" in str(err.value).lower()

    def test_get_keyword_table(self):
        with pytest.raises(ValueError):
            u.get_keyword_table(filename='__does_not_exist__')
        table = u.get_keyword_table()
        columns = ['required', 'default', 'type', 'combine',
                   'min', 'max', 'enum']
        for column in columns:
            assert column in table

        row = table[
            table.apply(lambda x: len(x.enum) != 0, axis=1)].iloc[0]
        assert isinstance(row.enum, list)
        assert row.type(row.enum[0]) == row.enum[0]

        assert table.apply(
            lambda x: isinstance(x.required, bool), axis=1).all()
        assert table.apply(lambda x: isinstance(x['min'], float), axis=1).all()
        assert table.apply(lambda x: isinstance(x['max'], float), axis=1).all()
        allowed_combines = ['first', 'last', 'default', 'and', 'or',
                            'concatenate', 'mean', 'sum']
        assert table.apply(
            lambda x: x['combine'] in allowed_combines, axis=1).all()

    def test_get_keyword_comments_table(self):
        with pytest.raises(ValueError):
            u.get_keyword_comments_table(filename='__does_not_exist__')
        table = u.get_keyword_comments_table()
        assert table.apply(
            lambda row: isinstance(row.comment, str), axis=1).all()

    def test_clear_values(self):
        # set some values from a header
        filename = test_files()[0]
        hdul = fits.open(filename)
        header = hdul[0].header

        table = u.get_keyword_values(header, [header])
        values = np.array([r['value'] for r in table.values()])
        assert not np.all(np.equal(values, None))

        # clear them
        u.clear_values(table)
        values = np.array([r['value'] for r in table.values()])
        assert np.all(np.equal(values, None))

        # if table is none, just return
        u.clear_values(None)

    def test_value_from_header(self):
        key = 'TESTKEY'
        row = {'key': key, 'type': str, 'default': 'UNKNOWN',
               'combine': 'first'}

        # check for default return
        header = fits.Header()
        assert u.value_from_header(header, row) is None
        assert u.value_from_header(header, row, default='test') == 'test'

    # Checks for value_from_headers
    def test_and(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': bool, 'default': False, 'combine': 'and'}

        headers = [fits.Header() for _ in range(5)]
        for h in headers:
            h[key] = True

        assert u.aggregate_key_value(headers[0], headers, row)
        headers[-1][key] = False
        assert not u.aggregate_key_value(headers[0], headers, row)

    def test_concatenate(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': str, 'default': 'UNKNOWN',
               'combine': 'concatenate'}
        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h[key] = str(idx)

        assert u.aggregate_key_value(headers[0], headers, row) == '0,1,2,3,4'

        headers[-1][key] = '1'
        assert u.aggregate_key_value(headers[0], headers, row) == '0,1,2,3'

    def test_default(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': str, 'default': 'foo',
               'combine': 'default'}

        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h[key] = str(idx)
        assert u.aggregate_key_value(headers[0], headers, row) == 'foo'

    def test_first(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': str, 'default': 'UNKNOWN',
               'combine': 'first'}
        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h[key] = str(idx)
        assert u.aggregate_key_value(headers[0],
                                     headers, row) == headers[0][key]

        # should be same for unknown combine value
        row['combine'] = 'unknown'
        assert u.aggregate_key_value(headers[0],
                                     headers, row) == headers[0][key]

    def test_last(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': str, 'default': 'UNKNOWN',
               'combine': 'last'}

        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h[key] = str(idx)

        assert u.aggregate_key_value(headers[0],
                                     headers, row) == headers[-1][key]

    def test_mean(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': int, 'default': -9999,
               'combine': 'mean'}

        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h[key] = idx
        expected = int(np.mean(np.arange(len(headers))))
        assert u.aggregate_key_value(headers[0], headers, row) == expected

    def test_or(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': bool, 'default': False,
               'combine': 'or'}

        headers = [fits.Header() for _ in range(5)]
        for h in headers:
            h[key] = False
        assert not u.aggregate_key_value(headers[0], headers, row)

        headers[-1][key] = True
        assert u.aggregate_key_value(headers[0], headers, row)

    def test_sum(self):

        key = 'TESTKEY'
        row = {'key': key, 'type': float, 'default': -9999.,
               'combine': 'sum'}

        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h[key] = idx
        expected = int(np.sum(np.arange(len(headers))))
        assert np.isclose(u.aggregate_key_value(headers[0],
                                                headers, row), expected)

        # check bad sum: should assign default
        # (convert to dict because it's not easy to put a
        # bad value in a fits header)
        dict_headers = []
        for h in headers:
            h = dict(h)
            h[key] = np.inf
            dict_headers.append(h)
        expected = -9999
        assert np.isclose(u.aggregate_key_value(dict_headers[0],
                                                dict_headers, row), expected)

    def test_check_row(self, dummy_table):

        # Check bad date
        key = 'DATE-OBS'
        table = dummy_table(key=key, value='1970-01-01T00:00:00')
        assert not u.check_key(table, key)

        # Check skips unrequired
        table[key]['required'] = False
        assert u.check_key(table, key)

        # Check good date
        table[key]['required'] = True
        table[key]['value'] = '1970-01-02T00:00:00'
        assert u.check_key(table, key)

        # Check null value
        table[key]['value'] = None
        assert not u.check_key(table, key)

        # Check missing definition
        table = dummy_table(value=None)
        assert u.check_key(table, key)

        # Check wrong type
        key = 'TESTKEY'
        table = dummy_table(key=key, dtype=u.robust_bool, value=False)
        assert u.check_key(table, key)
        table[key]['value'] = 1
        assert not u.check_key(table, key)

        # Check enum
        row = table[key]
        row['type'] = int
        row['enum'] = [1, 2, 3]
        row['value'] = 4
        assert not u.check_key(table, key)
        row['value'] = 3
        assert u.check_key(table, key)

        # Check min/max
        table = dummy_table(key=key, value=3.0, dmin=2.0,
                            dmax=4.0, dtype=float)
        assert u.check_key(table, key)
        row = table[key]
        row['value'] = 0.0
        assert not u.check_key(table, key)

        # remove min requirement
        row['min'] = np.nan
        assert u.check_key(table, key)

        # check max requirement
        row['value'] = 5.0
        assert not u.check_key(table, key)

        # remove max requirement
        row['max'] = np.nan
        assert u.check_key(table, key)

    def test_update_single(self):
        header = fits.Header()
        header['AOR_ID'] = 'test_aor '
        header['MISSN-ID'] = 'test_mid '
        header['OBS_ID'] = 'test_obs '

        headers = [header]
        basehead, headers = u.order_headers(headers)

        table = u.get_keyword_values(header, headers)
        u.update_basehead(basehead, table, headers)

        assert basehead['ASSC_AOR'] == 'TEST_AOR'
        assert basehead['ASSC_MSN'] == 'TEST_MID'
        assert basehead['OBS_ID'] == 'P_TEST_OBS'
        assert basehead['PROCSTAT'] == 'LEVEL_2'
        assert abs(Time.now() - Time(basehead['DATE'])).to('s').value < 10
        assert basehead['FILENUM'] == 'UNKNOWN'

    def test_update_filenum(self):
        header = fits.Header()
        header['FILENAME'] = '12345_a_file'
        basehead, headers = u.order_headers([header])
        table = u.get_keyword_values(header, [header])
        u.update_basehead(basehead, table, [header])
        assert basehead['FILENUM'] == '12345'

        header = fits.Header()
        header['OBS_ID'] = 'something_B11_words_R22'
        basehead, headers = u.order_headers([header])
        table = u.get_keyword_values(header, [header])
        u.update_basehead(basehead, table, [header])
        assert basehead['FILENUM'] == '22'

        n = 5
        headers = [fits.Header() for _ in range(n)]
        for i, h in enumerate(headers):
            start = i * n
            end = (i + 1) * n - 1
            h['FILENUM'] = '%i-%i' % (start, end)
        headers[0]['FILENUM'] = 'UNKNOWN'
        basehead, headers = u.order_headers(headers)
        table = u.get_keyword_values(headers[0], headers)
        u.update_basehead(basehead, table, headers)

        expected = '%i-%i' % (n, (n ** 2) - 1)
        assert basehead['FILENUM'] == expected

        # test filenum combo -- only one found
        for h in headers:
            h['FILENUM'] = '1'
        expected = '1'
        u.update_basehead(basehead, table, headers)
        assert basehead['FILENUM'] == expected

        # test if none found
        # test filenum combo -- only one found
        for h in headers:
            del h['FILENUM']
        expected = 'UNKNOWN'
        u.update_basehead(basehead, table, headers)
        assert basehead['FILENUM'] == expected

    def test_make_header(self):

        result = u.make_header(headers=None, checkheader=False)
        assert isinstance(result, fits.Header)

        result = u.make_header(headers=fits.Header(), checkheader=False)
        assert isinstance(result, fits.Header)

        result = u.make_header(headers=[], checkheader=False)
        assert result is None

        result = u.make_header(headers=1, checkheader=False)
        assert result is None

        result = u.make_header(headers=[fits.Header(), 1], checkheader=False)
        assert result is None

        header = fits.Header({'DATE-OBS': '1970-01-01T00:00:00'})
        result, success = u.make_header(headers=header, checkheader=True)
        assert isinstance(result, fits.Header)
        assert not success

    def test_expected_warnings(self):

        from astropy import log

        header = fits.Header()
        header['ALTI_STA'] = 'UNKNOWN'
        header['DATE-OBS'] = 'UNKNOWN'

        with log.log_to_list() as log_list:
            u.make_header(headers=header, checkheader=True, check_all=True)

        log_list = '\n'.join([str(x) for x in log_list])
        assert 'Required keyword DATE-OBS ' \
               'has wrong value (UNKNOWN)' in log_list
        assert 'Required keyword ALTI_STA has wrong type' in log_list
        assert 'Required keyword NODBEAM not found' in log_list

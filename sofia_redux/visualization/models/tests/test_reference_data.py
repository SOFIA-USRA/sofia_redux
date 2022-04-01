from sofia_redux.visualization.models import reference_model
from sofia_redux.visualization.utils import unit_conversion
import pytest
import logging
import astropy.units as u
import numpy as np


class TestReferenceData(object):

    def test_init(self):
        obj = reference_model.ReferenceData()
        assert isinstance(obj.line_list, dict)
        assert len(obj.line_list) == 0

    def test_add(self):
        obj1 = reference_model.ReferenceData()
        obj2 = reference_model.ReferenceData()
        names = {'H1': [12.3872, 13.1022, 13.5213, 19.0619, 24.52, 25.988]}
        other = {'H2': [12.8136]}
        obj1.line_list = names
        obj2.line_list = other
        new = obj1 + obj2
        assert obj1.line_list == new.line_list

    def test_add_err(self):
        obj1 = reference_model.ReferenceData()
        obj2 = []
        with pytest.raises(ValueError) as err:
            obj1 + obj2
        assert 'Invalid type for addition' in str(err)

    def test_repr(self):
        obj = reference_model.ReferenceData()
        result = repr(obj)
        assert 'Reference data:' in result

    def test_add_linelist(self, line_list, line_list_csv, line_list_pipe,
                          line_list_whitespace):
        files = [line_list_csv, line_list_pipe, line_list_whitespace]
        for fn in files:
            obj = reference_model.ReferenceData()
            result = obj.add_line_list(fn)
            assert result
            assert obj.line_list.keys() == line_list.keys()
            assert all([isinstance(v, list)
                        for v in obj.line_list.values()])
            assert all([isinstance(v[0], float)
                        for v in obj.line_list.values()])
            assert all([res[0] == float(cor)
                        for res, cor in zip(list(obj.line_list.values()),
                                            list(line_list.values()))])

    def test_add_linelist_simple(self, line_list, line_list_simple):
        obj = reference_model.ReferenceData()
        result = obj.add_line_list(line_list_simple)
        assert result
        assert all([res[0] == float(cor)
                    for res, cor in zip(list(obj.line_list.values()),
                                        list(line_list.values()))])

    def test_add_linelist_empty(self):
        obj = reference_model.ReferenceData()
        result = obj.add_line_list('/a/bad/path')
        assert result is False
        assert len(obj.line_list) == 0

    def test_add_linelist_fail1(self, caplog, mocker, line_list_simple):
        caplog.set_level(logging.DEBUG)
        obj = reference_model.ReferenceData()

        mocker.patch.object(obj, '_read_single_column', side_effect=ValueError)
        result = obj.add_line_list(line_list_simple)

        assert 'Error in reading line list:' in caplog.text
        assert result is False

    def test_add_linelist_fail2(self, mocker, line_list_empty, caplog):
        caplog.set_level(logging.INFO)
        obj = reference_model.ReferenceData()
        result = obj.add_line_list(line_list_empty)
        assert result is True
        assert 'Line list is empty' in caplog.text

    def test_read_single_column(self, caplog):
        caplog.set_level(logging.DEBUG)
        obj = reference_model.ReferenceData()
        lines = ['3.0392023', '3.0916927', '3.2969917', '3.7405565',
                 '4.0522623']
        obj._read_single_column(lines)
        assert 'Reading line list without labels' in caplog.text
        for line in lines:
            label = f'{float(line):.5g} um'
            assert obj.line_list[label] == [float(line.strip())]

    def test_read_multiple_column(self, caplog, line_list_duplicates):
        caplog.set_level(logging.DEBUG)
        obj = reference_model.ReferenceData()
        obj._read_multiple_columns(line_list_duplicates, ' ')
        assert 'Reading line list with labels' in caplog.text
        for line in line_list_duplicates:
            parts = line.strip().split(' ')
            transition = ' '.join(parts[1:]).strip()
            wavelength = float(parts[0])
            assert wavelength in obj.line_list[transition]

    def test_convert_list_unit_simple(self, mocker):
        obj = reference_model.ReferenceData()
        target_unit = {'x': 'um', 'y': 'Jy'}
        names = [None, [], {}]
        for name in names:
            conv_line_list = obj.line_list
            result = obj.convert_line_list_unit(target_unit['x'], name)
            assert result == conv_line_list

    def test_convert_list_unit_err(self, mocker):
        obj = reference_model.ReferenceData()
        mocker.patch.object(unit_conversion, 'convert_wave',
                            side_effect=ValueError)
        target_unit = {'x': 'um', 'y': 'Jy'}
        names = {'H1': [12.3872, 13.1022, 13.5213, 19.0619, 24.52, 25.988],
                 'H2': [12.8136]}
        obj.line_unit = u.nm
        obj.line_list = names.copy()
        result = obj.convert_line_list_unit(target_unit['x'], names.copy())
        assert result == names

        # obj.line_list should be unchanged
        assert obj.line_list == names

    def test_convert_list_unit(self):
        obj = reference_model.ReferenceData()
        target_unit = {'x': 'nm', 'y': 'Jy'}
        names = {'H1': [12.3872, 13.1022, 13.5213, 19.0619, 24.52, 25.988],
                 'H2': [12.8136]}
        obj.line_unit = u.um
        obj.line_list = names.copy()
        result = obj.convert_line_list_unit(target_unit['x'], names.copy())
        assert np.allclose(result['H1'], [n * 1000 for n in names['H1']])
        assert np.isclose(result['H2'][0], names['H2'][0] * 1000)

        # obj.line_list should be unchanged
        assert obj.line_list == names

        # names as list
        names_list = ['H1', 'H2']
        result = obj.convert_line_list_unit(target_unit['x'], names_list)
        assert np.allclose(result['H1'], [n * 1000 for n in names['H1']])
        assert np.isclose(result['H2'][0], names['H2'][0] * 1000)
        assert obj.line_list == names

    @pytest.mark.parametrize('targets,state,',
                             [('ref_line', True), ('ref_label', True),
                              ('all', True)])
    def test_set_visibility(self, targets, state):
        obj = reference_model.ReferenceData()

        obj.set_visibility(targets, state)

        # ignored if key is bad
        obj.set_visibility('bad', True)
        assert 'bad' not in obj.enabled

        if targets == 'all':
            target = ['ref_line', 'ref_label']
            for t in target:
                assert obj.enabled[t] == state
        else:
            assert obj.enabled[targets] == state

    @pytest.mark.parametrize('targets', ['ref_line', 'ref_label'])
    def test_get_visibility(self, targets):
        obj = reference_model.ReferenceData()
        obj.get_visibility(targets)
        assert isinstance(obj.enabled[targets], bool)

    def test_get_visibility_err(self):
        obj = reference_model.ReferenceData()
        targets = 'random'
        result = obj.get_visibility(targets)
        assert result is None

    def test_unload_data(self):
        obj = reference_model.ReferenceData()
        obj.unload_data()
        assert isinstance(obj.line_list, dict)
        assert len(obj.line_list) == 0
        assert obj.enabled['ref_line'] is False
        assert obj.enabled['ref_label'] is False

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Parameters class."""

import configobj
import pytest
from sofia_redux.pipeline.parameters import Parameters, ParameterSet


class TestParameters(object):

    def make_default_param(self):
        """Make a set of parameters from a default dictionary."""
        default = {'test_step': [{'key': 'test_key_1',
                                  'value': 'test_value'},
                                 {'key': 'test_key_2',
                                  'name': 'Test Key 2',
                                  'dtype': 'int',
                                  'options': [1, 2],
                                  'option_index': 1},
                                 {'key': 'test_key_3',
                                  'options': [1, 2],
                                  'option_index': 3,
                                  'description': 'test_description'},
                                 {'key': 'test_key_4',
                                  'wtype': 'check_box',
                                  'value': True,
                                  'hidden': True},
                                 {'key': 'test_key_5',
                                  'dtype': 'bool'}]}
        param = Parameters(default)
        param.add_current_parameters('test_step')
        return param

    def test_parameter_default(self):
        param = self.make_default_param()

        # test that the step was added to current
        assert param.stepnames == ['test_step']
        assert len(param.current) == 1
        parset = param.current[0]
        assert type(parset) == ParameterSet

        # test that default values were added for any non-specified keys
        # and correct values were added for all others
        assert parset['test_key_1']['value'] == 'test_value'
        assert parset['test_key_1']['dtype'] == 'str'
        assert parset['test_key_1']['wtype'] == 'text_box'
        assert parset['test_key_1']['name'] == 'test_key_1'
        assert parset['test_key_1']['options'] is None
        assert parset['test_key_1']['option_index'] == 0
        assert parset['test_key_1']['description'] is None
        assert parset['test_key_1']['hidden'] is False

        assert parset['test_key_2']['value'] == 2
        assert parset['test_key_2']['dtype'] == 'int'
        assert parset['test_key_2']['wtype'] == 'combo_box'
        assert parset['test_key_2']['name'] == 'Test Key 2'
        assert parset['test_key_2']['options'] == [1, 2]
        assert parset['test_key_2']['option_index'] == 1
        assert parset['test_key_2']['description'] is None
        assert parset['test_key_2']['hidden'] is False

        assert parset['test_key_3']['value'] is None
        assert parset['test_key_3']['dtype'] == 'str'
        assert parset['test_key_3']['wtype'] == 'combo_box'
        assert parset['test_key_3']['name'] == 'test_key_3'
        assert parset['test_key_3']['options'] == [1, 2]
        assert parset['test_key_3']['option_index'] == 3
        assert parset['test_key_3']['description'] == 'test_description'
        assert parset['test_key_3']['hidden'] is False

        assert parset['test_key_4']['value'] is True
        assert parset['test_key_4']['dtype'] == 'bool'
        assert parset['test_key_4']['wtype'] == 'check_box'
        assert parset['test_key_4']['name'] == 'test_key_4'
        assert parset['test_key_4']['options'] is None
        assert parset['test_key_4']['option_index'] == 0
        assert parset['test_key_4']['description'] is None
        assert parset['test_key_4']['hidden'] is True

        assert parset['test_key_5']['value'] is None
        assert parset['test_key_5']['dtype'] == 'bool'
        assert parset['test_key_5']['wtype'] == 'check_box'
        assert parset['test_key_5']['name'] == 'test_key_5'
        assert parset['test_key_5']['options'] is None
        assert parset['test_key_5']['option_index'] == 0
        assert parset['test_key_5']['description'] is None
        assert parset['test_key_5']['hidden'] is False

        # test that correct values are returned from get_value
        assert parset.get_value('test_key_1') == 'test_value'
        assert parset.get_value('test_key_2') == 2
        assert parset.get_value('test_key_3') is None
        assert parset.get_value('test_key_4') is True
        assert parset.get_value('test_key_5') is None

    def test_set_value(self):
        param = self.make_default_param()
        parset = param.current[0]

        # text value
        assert parset.get_value('test_key_1') == 'test_value'
        parset.set_value('test_key_1', 'new_value')
        assert parset.get_value('test_key_1') == 'new_value'

        # option index value
        assert parset.get_value('test_key_2') == 2
        # set by index
        parset.set_value('test_key_2', option_index=0)
        assert parset.get_value('test_key_2') == 1
        # set by value
        parset.set_value('test_key_2', value=2)
        assert parset.get_value('test_key_2') == 2
        # set to non-enumerated value
        parset.set_value('test_key_2', value='q')
        assert parset.get_value('test_key_2') == 'q'

        # options value
        assert parset.get_value('test_key_3') is None
        parset.set_value('test_key_3', options=[1, 2, 3, 4, 5])
        assert parset.get_value('test_key_3') == 4

        # bool value
        assert parset.get_value('test_key_4') is True
        parset.set_value('test_key_4', value=False)
        assert parset.get_value('test_key_4') is False
        parset.set_value('test_key_4', value='false')
        assert parset.get_value('test_key_4') is False
        parset.set_value('test_key_4', value='0')
        assert parset.get_value('test_key_4') is False
        parset.set_value('test_key_4', value='1')
        assert parset.get_value('test_key_4') is True
        parset.set_value('test_key_4', value=True)
        assert parset.get_value('test_key_4') is True

        # new value
        with pytest.raises(KeyError):
            parset.get_value('test_key_6')
        parset.set_value('test_key_6', value='new')
        assert parset.get_value('test_key_6') == 'new'

        # set hidden
        assert parset['test_key_6']['hidden'] is False
        parset.set_value('test_key_6', value='new2', hidden=True)
        assert parset.get_value('test_key_6') == 'new2'
        assert parset['test_key_6']['hidden'] is True
        # verify unchanged if not passed
        parset.set_value('test_key_6', value='new3')
        assert parset.get_value('test_key_6') == 'new3'
        assert parset['test_key_6']['hidden'] is True

    def test_from_config(self, tmpdir, capsys):
        param = self.make_default_param()
        parset = param.current[0]

        # from configobj
        default = {'test_step': {'test_key_1': 'value_from_config_1'}}
        co = configobj.ConfigObj(default)
        co.filename = 'test.cfg'
        param.from_config(co)
        assert parset.get_value('test_key_1') == 'value_from_config_1'
        assert "Setting parameters from " \
               "configuration input" in capsys.readouterr().out

        # from file on disk
        co.filename = str(tmpdir.join('test.cfg'))
        co.write()
        param.from_config(co)
        assert "Setting parameters from " \
               "configuration file" in capsys.readouterr().out

        # with bad filename
        co['test_step']['test_key_1'] = 'value_from_config_2'
        co.filename = {'bad': 'value'}
        param.from_config(co)
        assert parset.get_value('test_key_1') == 'value_from_config_2'

        # from dictionary
        default = {'test_step': {'test_key_1': 'value_from_dict_1'}}
        param.from_config(default)
        assert parset.get_value('test_key_1') == 'value_from_dict_1'

        # with index specified
        default = {'1: test_step': {'test_key_1': 'value_from_dict_2'}}
        param.from_config(default)
        assert parset.get_value('test_key_1') == 'value_from_dict_2'

        # with bad index
        default = {'2: test_step': {'test_key_1': 'value_from_dict_3'}}
        param.from_config(default)
        assert parset.get_value('test_key_1') == 'value_from_dict_3'

        # with bad name: ignored
        default = {'1: test_bad_step': {'test_key_1': 'value_from_dict_4'}}
        param.from_config(default)
        assert parset.get_value('test_key_1') == 'value_from_dict_3'

    def test_to_config(self):
        param = self.make_default_param()
        parset = param.current[0]

        co = param.to_config()
        assert isinstance(co, configobj.ConfigObj)

        assert co['1: test_step']['test_key_1'] == \
            parset.get_value('test_key_1')
        assert co['1: test_step']['test_key_2'] == \
            parset.get_value('test_key_2')
        assert co['1: test_step']['test_key_3'] == \
            parset.get_value('test_key_3')
        assert co['1: test_step']['test_key_5'] == \
            parset.get_value('test_key_5')

        # hidden value is not written
        with pytest.raises(KeyError):
            print(co['1: test_step']['test_key_4'])

    def test_to_text(self):
        # simpler parameter object
        param = Parameters({'test_step': [{'key': 'test_key',
                                           'value': 'test_value'}]})
        param.add_current_parameters('test_step')
        par_text = param.to_text()
        assert '\n'.join(par_text) == \
               '[1: test_step]\n    test_key = test_value'

    @pytest.mark.parametrize('data', ['string', '1', '0.0', 'False', {1: 2}])
    def test_get_str_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'str'

    @pytest.mark.parametrize('data', [1.0, 0.0, -5.0, 1e5])
    def test_get_float_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'float'

    @pytest.mark.parametrize('data', [1, 0, -5])
    def test_get_int_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'int'

    @pytest.mark.parametrize('data', [True, False])
    def test_get_bool_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'bool'

    @pytest.mark.parametrize(
        'data', [['string'], ['1', 'a'], ['0.0', 1],
                 ['False', True], [], [{1: 2}]])
    def test_get_strlist_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'strlist'

    @pytest.mark.parametrize(
        'data', [[1.0], [0.0, -1], [-5.0, 'a'], [1e5, 1.0, 10]])
    def test_get_floatlist_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'floatlist'

    @pytest.mark.parametrize(
        'data', [[1], [0, -1], [-5, 'a'], [1, 1.0, 1e5]])
    def test_get_intlist_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'intlist'

    @pytest.mark.parametrize(
        'data', [[True, False], [True], [False], [True, 'a']])
    def test_get_boollist_type(self, data):
        dtype = Parameters.get_param_type(data)
        assert dtype == 'boollist'

    @pytest.mark.parametrize(
        'data', ['string', 1, 0.0, False, ['s1', 's2'], {1: 2}])
    def test_fix_str_type(self, data):
        fixed = Parameters.fix_param_type(data, 'str')
        assert type(fixed) == str
        assert fixed == str(data)

    @pytest.mark.parametrize('data', ['string', 1, True, 'True'])
    def test_fix_bool_true(self, data):
        fixed = Parameters.fix_param_type(data, 'bool')
        assert type(fixed) == bool
        assert fixed is True

    @pytest.mark.parametrize(
        'data', ['False', 0, 0.000, 'falSE', False, ''])
    def test_fix_bool_false(self, data):
        fixed = Parameters.fix_param_type(data, 'bool')
        assert type(fixed) == bool
        assert fixed is False

    @pytest.mark.parametrize('data', ['1', 1, 1.0])
    def test_fix_int(self, data):
        fixed = Parameters.fix_param_type(data, 'int')
        assert type(fixed) == int
        assert fixed == 1

    @pytest.mark.parametrize('data', ['1', 1, 1.0])
    def test_fix_float(self, data):
        fixed = Parameters.fix_param_type(data, 'float')
        assert type(fixed) == float
        assert fixed == 1.0

    @pytest.mark.parametrize(
        'data', ["['1','1','1']", "[1, 1, 1]",
                 [1, 1, 1], ['1', '1', '1']])
    def test_fix_strlist(self, data):
        fixed = Parameters.fix_param_type(data, 'strlist')
        assert type(fixed) == list
        assert len(fixed) == 3
        for el in fixed:
            assert type(el) == str
            assert el == '1'

    @pytest.mark.parametrize(
        'data', ["['1','1','1']", "[1, 1, 1]",
                 [1, 1, 1], ['1', '1', '1']])
    def test_fix_intlist(self, data):
        fixed = Parameters.fix_param_type(data, 'intlist')
        assert type(fixed) == list
        assert len(fixed) == 3
        for el in fixed:
            assert type(el) == int
            assert el == 1

    @pytest.mark.parametrize(
        'data', ["['1','1','1']", "[1, 1, 1]",
                 [1, 1, 1], ['1', '1', '1']])
    def test_fix_floatlist(self, data):
        fixed = Parameters.fix_param_type(data, 'floatlist')
        assert type(fixed) == list
        assert len(fixed) == 3
        for el in fixed:
            assert type(el) == float
            assert el == 1.0

    @pytest.mark.parametrize(
        'data', ["['0','0','0']", "[False, 0, 0.0]",
                 [0, False, 0.0], ['', '0', False]])
    def test_fix_boollist_false(self, data):
        fixed = Parameters.fix_param_type(data, 'boollist')
        assert type(fixed) == list
        assert len(fixed) == 3
        for el in fixed:
            assert type(el) == bool
            assert el is False

    @pytest.mark.parametrize(
        'data', ["['1','1','True']", "[True, 1, 1.0]",
                 [1, True, 1.0], ['1', 'a', True]])
    def test_fix_boollist_true(self, data):
        fixed = Parameters.fix_param_type(data, 'boollist')
        assert type(fixed) == list
        assert len(fixed) == 3
        for el in fixed:
            assert type(el) == bool
            assert el is True

    @pytest.mark.parametrize('data', [{1: 2}, 'a'])
    def test_fix_wrong_type(self, data):
        fixed = Parameters.fix_param_type(data, 'int')
        assert type(fixed) == str
        assert fixed == str(data)

        fixed = Parameters.fix_param_type(data, 'float')
        assert type(fixed) == str
        assert fixed == str(data)

        fixed = Parameters.fix_param_type(data, 'intlist')
        assert type(fixed) == str
        assert fixed == str(data)

        fixed = Parameters.fix_param_type(data, 'floatlist')
        assert type(fixed) == str
        assert fixed == str(data)

    @pytest.mark.parametrize('data', [[1, 'a']])
    def test_fix_wrong_type_okay(self, data):
        fixed = Parameters.fix_param_type(data, 'intlist')
        assert type(fixed) == list
        assert fixed == data

        fixed = Parameters.fix_param_type(data, 'floatlist')
        assert type(fixed) == list
        assert fixed == data

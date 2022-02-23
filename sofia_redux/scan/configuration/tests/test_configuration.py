# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import json
import shutil

from astropy import log
from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.configuration.aliases import Aliases
from sofia_redux.scan.configuration.conditions import Conditions
from sofia_redux.scan.configuration.dates import DateRangeOptions
from sofia_redux.scan.configuration.iterations import IterationOptions
from sofia_redux.scan.configuration.objects import ObjectOptions
from sofia_redux.toolkit.utilities.multiprocessing import in_windows_os


def test_init():
    configuration = Configuration()
    configuration.set_error_handling(False)
    assert os.path.isdir(configuration.config_path)
    assert configuration.config_path == configuration.options['configpath']
    with pytest.raises(NotADirectoryError) as err:
        Configuration(configuration_path='_not_a_directory_')
    assert "Configuration directory not found" in str(err.value)

    with log.log_to_list() as log_list:
        Configuration(configuration_path='_not_a_directory_',
                      allow_error=True, verbose=True)
    assert len(log_list) == 1
    assert "Configuration directory not found" in log_list[0].msg


def test_contains():
    c = Configuration()
    c.put('foo.bar', '1')
    assert c.__contains__('foo.bar')
    c.forget('foo.bar')
    assert not c.__contains__('foo.bar')


def test_copy():
    configuration = Configuration()
    configuration.config_files.append('testing')
    new = configuration.copy()
    assert isinstance(new, Configuration)
    assert new != configuration
    assert len(new.config_files) == 1 and new.config_files[0] == 'testing'


def test_clear(config_file):
    c = Configuration()
    c.read_configuration(config_file)
    handlers = [c.aliases, c.conditions, c.dates, c.iterations,
                c.objects, c.fits]
    assert len(c) > 1
    n_options = [len(h) for h in handlers]
    c.clear()
    assert len(c) == 1 and 'configpath' in c
    for i, h in enumerate(handlers):
        if n_options[i] > 0:
            assert len(h) < n_options[i]
    assert len(c.locked) == 0
    assert len(c.disabled) == 0
    assert len(c.whitelisted) == 0
    assert len(c.forgotten) == 0
    assert len(c.applied_conditions) == 0
    assert len(c.config_files) == 0


def test_delitem(initialized_configuration):
    c = initialized_configuration
    assert 'o1s1.v1' in c
    del c['o1s1.v1']
    assert 'o1s1.v1' not in c
    c = Configuration()
    c.options = None
    with pytest.raises(KeyError) as err:
        del c['a']
    assert "Options are not initialized" in str(err.value)


def test_exists():
    c = Configuration()
    c.put('foo.bar.baz', '1')
    assert c.exists('foo.bar.baz')
    c.put('a', '1')
    assert c.exists('a')
    assert not c.exists('foo.bar.a')
    c.options['foo'] = {'bar': ['1']}
    assert not c.exists('foo.bar.baz')


def test_get(initialized_configuration):
    c = initialized_configuration
    assert c.get('testvalue1') == 'foo'
    assert c.get('nonexistent', default='default') == 'default'
    assert c.get('options1') == 'True'


def test_get_branch(initialized_configuration):
    c = initialized_configuration
    c.forget('options1')
    assert c.get_branch('options1', default=None) is None
    c.recall('options1')
    assert isinstance(c.get_branch('options1'), dict)

    default = '{?testvalue1}'
    assert c.get_branch('options1.nonexistent', default=default, unalias=True
                        ) == 'foo'
    assert c.get_branch('options1.nonexistent', default=default, unalias=False
                        ) == default
    c.put('options1.new_value', default)
    assert c.get_branch('options1.new_value', unalias=True) == 'foo'
    assert c.get_branch('options1.new_value', unalias=False) == default


def test_set_error_handling():
    c = Configuration()
    c.set_error_handling(False)
    assert not c.allow_error
    for handler in [c.aliases, c.conditions, c.dates, c.iterations, c.objects,
                    c.fits]:
        assert not handler.allow_error

    c.set_error_handling(True)
    assert c.allow_error
    for handler in [c.aliases, c.conditions, c.dates, c.iterations, c.objects,
                    c.fits]:
        assert handler.allow_error


def test_set_verbosity():
    c = Configuration()
    c.set_verbosity(False)
    assert not c.verbose
    for handler in [c.aliases, c.conditions, c.dates, c.iterations, c.objects,
                    c.fits]:
        assert not handler.verbose

    c.set_verbosity(True)
    assert c.verbose
    for handler in [c.aliases, c.conditions, c.dates, c.iterations, c.objects,
                    c.fits]:
        assert handler.verbose


def test_get_section_handler():
    configuration = Configuration()
    assert isinstance(configuration.get_section_handler('aliases'), Aliases)
    assert isinstance(configuration.get_section_handler('conditionals'),
                      Conditions)
    assert isinstance(configuration.get_section_handler('date'),
                      DateRangeOptions)
    assert isinstance(configuration.get_section_handler('iteration'),
                      IterationOptions)
    assert isinstance(configuration.get_section_handler('object'),
                      ObjectOptions)
    assert configuration.get_section_handler('foo') is None


def test_parse_to_section():
    configuration = Configuration()
    assert not configuration.parse_to_section('foo', {'bar': 'baz'})
    assert configuration.parse_to_section('conditionals',
                                          {'a=b': {'add': 'c'}})
    assert configuration.parse_to_section('conditionals', {'c=d': 'e'})
    assert configuration.parse_to_section('conditionals',
                                          {'e=True': 'blacklist=f'})
    c = configuration.conditions.options
    assert c['a=b'] == {'add': 'c'}
    assert c['c=d'] == {'add': 'e'}
    assert c['e=True'] == {'blacklist': 'f'}

    assert configuration.parse_to_section('date', {'*--*': 'add=alldates'})
    assert configuration.dates.options['*--*'] == {'add': 'alldates'}
    assert configuration.parse_to_section(
        'date', {'2017-05-01--2017-06-01': 'may2017'})
    assert configuration.dates.options['2017-05-01--2017-06-01'] == {
        'add': 'may2017'}

    assert configuration.parse_to_section('aliases', 'foo=bar.baz')
    assert configuration.aliases('foo') == 'bar.baz'


def test_set_instrument():
    configuration = Configuration()
    configuration.set_instrument(None)
    assert configuration.instrument_name is None
    configuration.set_instrument('hawc_plus')
    assert configuration.instrument_name == 'hawc_plus'

    class MyInst(object):
        def __init__(self):
            self.name = 'test_instrument'

    inst = MyInst()
    configuration.set_instrument(inst)
    assert configuration.instrument_name == 'test_instrument'

    with pytest.raises(ValueError) as err:
        configuration.set_instrument(configuration)
    assert "Could not parse" in str(err.value)


def test_current_iteration(config_options):
    configuration = Configuration()
    configuration.read_configuration(config_options, validate=True)
    assert configuration.current_iteration is None
    configuration.current_iteration = 2
    assert configuration.current_iteration == 2
    assert configuration.get_int('o1s1.v1') == 10


def test_max_iteration():
    configuration = Configuration()
    assert configuration.max_iteration is None
    configuration.parse_key_value('rounds', '3')
    assert configuration.max_iteration == 3

    configuration.max_iteration = 5
    assert configuration.iterations.max_iteration == 5
    configuration.max_iteration = '4'
    assert configuration.max_iteration == 4


def test_blacklisted():
    configuration = Configuration()
    assert isinstance(configuration.blacklisted, set)
    assert len(configuration.blacklisted) == 0

    configuration.disabled.add('foo')
    configuration.disabled.add('bar')
    assert len(configuration.blacklisted) == 0
    configuration.locked.add('foo')
    assert 'foo' in configuration.blacklisted
    assert len(configuration.blacklisted) == 1
    configuration.locked.add('bar')
    black = configuration.blacklisted
    assert 'foo' in black and 'bar' in black and len(black) == 2
    configuration.disabled.remove('foo')
    assert 'bar' in configuration.blacklisted
    assert 'foo' not in configuration.blacklisted


def test_preserved_cards():
    configuration = Configuration()
    p = configuration.preserved_cards
    assert isinstance(p, dict) and len(p) == 0
    configuration.fits = None
    p = configuration.preserved_cards
    assert isinstance(p, dict) and len(p) == 0


def test_user_path():
    configuration = Configuration()
    assert configuration.user_path == os.path.join(
        os.path.expanduser('~'), '.sofscan')


def test_expected_path():
    configuration = Configuration()
    user_path = configuration.user_path
    if os.path.isdir(user_path):  # pragma: no cover
        assert configuration.expected_path == user_path
    else:  # pragma: no cover
        assert configuration.expected_path == configuration.config_path


def test_resolve_filepath():
    configuration = Configuration()
    path = configuration.expected_path
    assert configuration.resolve_filepath('abc') == os.path.join(path, 'abc')
    test_file = os.sep + 'abc'
    assert configuration.resolve_filepath(test_file) == test_file


def test_get_configuration_filepath():
    configuration = Configuration()
    full_path = f'{os.sep}abc'
    partial_path = 'abc'
    configuration.parse_key_value('full_path', full_path)
    configuration.parse_key_value('partial_path', partial_path)
    path = configuration.expected_path
    assert configuration.get_configuration_filepath('full_path') == full_path
    assert configuration.get_configuration_filepath(
        'partial_path') == os.path.join(path, partial_path)
    assert configuration.get_configuration_filepath('foo') is None


@pytest.mark.skipif(in_windows_os(), reason='Path differences')
def test_find_configuration_files():
    configuration = Configuration()
    configuration.set_instrument('hawc_plus')
    assert 'configpath' in configuration.options
    files = configuration.find_configuration_files('default.cfg')
    for f in files:
        assert os.path.isfile(f)
    assert os.path.join(configuration.config_path, 'default.cfg') in files

    # Test aliasing
    files = configuration.find_configuration_files('{?configpath}/default.cfg')
    assert len(files) == 1
    assert os.path.isfile(files[0])
    assert configuration.config_path + '/default.cfg' in files

    # Test bad full file path
    files = configuration.find_configuration_files(os.sep + '_fake_directory_'
                                                   + os.sep + 'foo.fits')
    assert len(files) == 0

    # Test bad partial file path
    files = configuration.find_configuration_files('foo.not_a_fits_file')
    assert len(files) == 0

    configuration.verbose = True
    with log.log_to_list() as log_list:
        configuration.find_configuration_files(os.sep + '_not_a_file_')
    assert "File not found" in log_list[0].msg

    with log.log_to_list() as log_list:
        _ = configuration.find_configuration_files('foo.not_a_fits_file')
    assert "No matching configuration files" in log_list[0].msg


def test_priority_file():
    configuration = Configuration()
    configuration.set_instrument('hawc_plus')
    f = configuration.priority_file('default.cfg')
    assert f.endswith(r'hawc_plus' + os.sep
                      + r'default.cfg') and os.path.isfile(f)

    # Check bad file
    f = configuration.priority_file('foo')
    assert f is None

    configuration.options['foo'] = 'default.cfg'
    f = configuration.priority_file('foo')
    assert f.endswith('hawc_plus' + os.sep
                      + 'default.cfg') and os.path.isfile(f)


def test_update(config_options):
    configuration = Configuration()
    assert len(configuration) == 1
    configuration.update(config_options)
    assert len(configuration) > 1


def test_read_configuration_file():
    configuration = Configuration()
    f = configuration.priority_file('default.cfg')

    with pytest.raises(ValueError) as err:
        configuration.read_configuration_file('__not_a_file__')
    assert "Not a file" in str(err.value)

    configuration.read_configuration_file(f, validate=False)
    assert len(configuration.options) > 1

    # Check the file will not be read if it's already in the config_file attr.
    configuration = Configuration()
    configuration.config_files.append(f)
    configuration.read_configuration_file(f)
    assert len(configuration.options) == 1

    configuration = Configuration(allow_error=True, verbose=True)
    with log.log_to_list() as log_list:
        configuration.read_configuration_file('_not_a_file_')
    assert len(log_list) == 1
    assert 'Not a file' in log_list[0].msg


def test_read_configuration(config_file):

    configuration = Configuration()

    # Test on a configuration file
    filename = config_file
    configuration.read_configuration(filename, validate=True)

    # Check aliases work
    assert configuration['o1s1.v1'] == configuration.options['options1'][
        'suboptions1']['v1']

    # Check dates are populated
    assert configuration.dates['2020-07-25'] == {'add': ['alldates',
                                                         'jul2020']}

    # Check iterations are populated and rounds are set.
    assert configuration.max_iteration == 5
    # Check that iteration conditions are updated
    configuration.parse_key_value('add', 'aug2021')
    configuration.validate()
    configuration.set_iteration(5, validate=True)
    assert configuration['lastiteration']

    old_configuration = configuration
    configuration = Configuration()
    configuration.read_configuration(old_configuration)
    assert 'aug2021' in configuration

    configuration = Configuration()
    old_configuration.enabled = False
    configuration.read_configuration(old_configuration)
    assert 'aug2021' not in configuration

    configuration = Configuration(allow_error=True, verbose=True)
    with log.log_to_list() as log_list:
        configuration.read_configuration(1)
    assert "Configuration must be of type" in log_list[0].msg

    configuration.set_error_handling(False)
    with pytest.raises(ValueError) as err:
        configuration.read_configuration(1)
    assert "Configuration must be of type" in str(err.value)


def test_read_configurations(config_file):
    file2 = config_file + '2'
    shutil.copyfile(config_file, file2)
    files = ','.join([config_file, file2])
    configuration = Configuration()
    configuration.read_configurations(files)
    os.remove(file2)
    assert config_file in configuration.config_files
    assert file2 in configuration.config_files


def test_validate(initialized_configuration):
    c = initialized_configuration
    assert c['o2s2.v2'] == 'f'
    c.parse_key_value('add', 'switch3')
    assert c['o2s2.v2'] == 'f'
    c.validate()
    assert c['o2s2.v2'] == 'z'


def test_set_object(initialized_configuration):
    c = initialized_configuration
    c.set_object('source1', validate=True)
    assert c.get_bool('src1')
    c.set_object('source2', validate=True)
    assert c['testvalue1'] == 'baz'


def test_set_iteration(initialized_configuration):
    c = initialized_configuration
    assert c['testvalue2'] == 'bar'
    c.parse_key_value('add', 'jul2020')
    c.validate()
    c.set_iteration(c.max_iteration, validate=True)
    with pytest.raises(KeyError) as err:
        _ = c['testvalue2']
    assert "'testvalue2'" in str(err.value)


def test_set_date(initialized_configuration):
    c = initialized_configuration
    c.set_date('2020-07-25')
    assert c.get_bool('jul2020')
    c.set_date('2021-08-03')
    assert c.get_bool('aug2021')


def test_set_serial(initialized_configuration):
    c = initialized_configuration.copy()
    c.set_serial(2)
    assert c.get_bool('scans2to3')
    assert c.get_bool('scans1to5')
    c = initialized_configuration.copy()
    c.set_serial(4)
    assert not c.get_bool('scans2to3')
    assert c.get_bool('scans1to5')

    c = initialized_configuration.copy()
    c.blacklist('serial')
    c.set_serial(2)
    assert not c.get_bool('scans2to3')
    assert not c.get_bool('scans1to5')


def test_apply_configuration_options():
    configuration = Configuration()
    configuration.apply_configuration_options({'add': 'foo'})
    assert configuration.get_bool('foo')
    configuration.clear()
    configuration.apply_configuration_options({'add': ['foo', 'bar']})
    assert configuration.get_bool('foo') and configuration.get_bool('bar')
    configuration.clear()

    configuration.apply_configuration_options({'iteration': {'1': 'add=foo'}})
    assert configuration.iterations.get(1) == {'add': 'foo'}
    configuration.clear()

    configuration.apply_configuration_options({'testvalue': 'foo'})
    assert configuration['testvalue'] == 'foo'

    configuration.clear()
    configuration.apply_configuration_options(None)
    assert len(configuration) == 1


def test_update_sections():
    configuration = Configuration()
    options = {'aliases': {'v': 'testvalue'},
               'date': {'2020-07-01--2020-07-30': {'pointing': ['-0.1', '1']}},
               'iteration': {'1': {'add': 'i1'}},
               'object': {'source1': {'add': 's1'}},
               'conditionals': {'s1': {'add': 'bar'}}}
    configuration.update_sections(options, validate=True)
    assert configuration.dates.get('2020-07-25')['pointing'][0] == '-0.1'
    assert configuration.aliases('v') == 'testvalue'
    assert configuration.iterations.get(1) == {'add': 'i1'}
    assert configuration.objects.get('source1') == {'add': 's1'}
    assert configuration.conditions.options['s1'] == {'add': 'bar'}
    configuration.set_object('source1')
    assert configuration.get_bool('s1')
    assert configuration.get_bool('bar')


def test_parse_configuration_body(config_options):
    c = Configuration()
    c.parse_configuration_body(config_options)
    handlers = [c.aliases, c.conditions, c.dates, c.iterations,
                c.objects, c.fits]
    # Check no options have been passed to the handlers.
    for handler in handlers:
        assert len(handler) == 0

    # Check that the main configuration options have been updated.
    assert len(c) > 1
    assert c['testvalue1'] == 'foo'
    assert 'testvalue3' in c.locked


def test_apply_commands(initialized_configuration):
    c = initialized_configuration
    assert 'testvalue3' in c.locked
    original_value = c['testvalue3']

    # Test the update command (dict)
    commands = {'update': {'new_key1': 'a', 'new_key2': 'b'}}
    c.apply_commands(commands)
    assert c['new_key1'] == 'a' and c['new_key2'] == 'b'

    # Test the update command (list)
    commands = {'update': [('new_key1', 'c'), ('new_key2', 'd')]}
    c.apply_commands(commands)
    assert c['new_key1'] == 'c' and c['new_key2'] == 'd'

    # Test ordering
    commands = {'unlock': 'testvalue3', 'update': {'testvalue3': 'unlocked'},
                'lock': 'testvalue3'}

    # This should not do anything
    c.apply_commands(commands, command_order=['lock', 'update', 'unlock'])
    assert c['testvalue3'] == original_value
    c.lock('testvalue3')
    assert 'testvalue3' in c.locked

    # This should have the desired effect
    c.apply_commands(commands, command_order=['unlock', 'update', 'lock'])
    assert c['testvalue3'] == 'unlocked' and c['testvalue3'] != original_value
    assert 'testvalue3' in c.locked


def test_put():
    c = Configuration()
    c.blacklist('lock_me')
    c.put('lock_me', '1')
    assert 'lock_me' in c.disabled and 'lock_me' in c.locked
    assert 'lock_me' not in c
    c.put('lock_me', '1', check=False)
    assert 'lock_me' not in c.disabled and 'lock_me' in c.locked
    assert 'lock_me' not in c

    c.put('foo', 'bar', '1')
    assert c['foo.bar'] == '1'


def test_read_fits(fits_file, initialized_configuration):
    filename = fits_file
    primary_header = fits.getheader(filename)
    c = initialized_configuration.copy()
    assert 'fits.STORE1' not in c
    c.read_fits(primary_header)
    assert c.get_int('fits.STORE1') == 1
    assert c.get_int('fits.STORE2') == 2
    assert c.get_int('fits.STORE3') == 3
    assert c.get_float('refdec') == 45.0
    assert c.get_float('refra') == 12.5

    assert 'finally' not in c
    c.set_object(c['refsrc'])
    assert c['finally']
    assert c.fits.preserved_cards == {
        'STORE2': (2, 'Stored value 2'), 'STORE1': (1, 'Stored value 1')}

    # Check other file extensions can be read
    c = initialized_configuration.copy()
    c.read_fits(filename, extension=1)
    assert c.get_int('fits.STORE1') == 11


def test_merge_fits_options(fits_file, initialized_configuration):
    filename = fits_file
    c = initialized_configuration
    c.fits.update_header(filename)
    assert 'fits.STORE1' not in c
    c.merge_fits_options()
    assert 'fits.STORE1' in c


def test_key_value_to_dict():
    assert Configuration.key_value_to_dict('a.b.c', 1) == {
        'a': {'b': {'c': 1}}}
    assert Configuration.key_value_to_dict('a', 1) == {'a': 1}


def test_matching_wildcard():
    string_array = ['file1.fits', 'file2.fits', 'foo', 'bar']
    assert Configuration.matching_wildcard(string_array, '*') == string_array
    assert Configuration.matching_wildcard(string_array, '*.fits') == [
        'file1.fits', 'file2.fits']
    assert Configuration.matching_wildcard(string_array, 'bar') == ['bar']


def test_matching_wildcard_keys(initialized_configuration):
    c = initialized_configuration
    assert c.matching_wildcard_keys('testvalue*') == [
        'testvalue1', 'testvalue2', 'testvalue3']


def test_flatten(initialized_configuration):
    c = initialized_configuration
    options = c.options.copy()
    # Add an empty value for testing coverage
    options['blank'] = {}

    flat = c.flatten(options)
    # Test .value keys are parsed correctly
    assert flat['options1'] == 'True'

    # Check a standard dot-key
    assert flat['serial.2-3.add'] == 'scans2to3'

    # Test unaliasing
    options['o2s1.some_value'] = 'hello'
    flat_unaliased = c.flatten(options, unalias=True)
    assert 'o2s1.some_value' not in flat_unaliased
    assert flat_unaliased['options2.suboptions1.some_value'] == 'hello'

    flat_aliased = c.flatten(options, unalias=False)
    assert flat_aliased['o2s1.some_value'] == 'hello'
    assert 'options2.suboptions1.some_value' not in flat_aliased


def test_merge_options(config_options):
    c = Configuration()
    c.merge_options(config_options)
    # Check single values
    assert c['testvalue1'] == 'foo'
    # Check nested values
    assert c['options1.suboptions2.v1'] == '3'
    # Check some handlers are populated
    assert c.conditions.options['switch3'] == {'o2s2.v2': 'z'}

    # Set the value for a sub-branch
    assert not c.is_configured('options2')
    assert c.has_option('options2')
    assert c['options2.value1'] == 'a'
    c.merge_options({'options2': 'branch_value'})
    assert c['options2'] == 'branch_value'

    # Overwrite certain options
    c.merge_options({'options2': {'suboptions1': {'v1': 'x', 'v2': 'y'}}})
    assert c['o2s1.v1'] == 'x'

    # Override singular value with branch, but keep value
    c.merge_options({'testvalue1': {'v1_options': ['1', '2', '3']}})
    assert c['testvalue1'] == 'foo'
    assert c['testvalue1.v1_options'][0] == '1'


def test_dot_key_in_set(initialized_configuration):
    c = initialized_configuration
    assert c.dot_key_in_set('final', ['iteration.-1'])


def test_parse_key_value(config_file, initialized_configuration):
    c = initialized_configuration
    # Test wildcards
    c.parse_key_value('testvalue*', 'is_wildcard')
    assert c['testvalue1'] == 'is_wildcard'
    assert c['testvalue2'] == 'is_wildcard'
    # Locked values should remain so
    assert c['testvalue3'] == 'lock_me'

    filename = config_file
    # Test commands
    c = Configuration()
    c.parse_key_value('config', filename)
    assert c['testvalue1'] == 'foo'

    c.parse_key_value('blacklist', 'testvalue1')
    assert 'testvalue1' not in c and 'testvalue1' in c.blacklisted

    c.parse_key_value('whitelist', 'testvalue1')
    assert 'testvalue1' not in c.blacklisted

    c.parse_key_value('forget', 'switch1')
    assert 'switch1' not in c
    c.parse_key_value('recall', 'switch1')
    assert 'switch1' in c

    c.parse_key_value('lock', 'switch1')
    assert 'switch1' in c.locked
    c.parse_key_value('unlock', 'switch1')
    assert 'switch1' not in c.locked

    c.parse_key_value('add', 'newkey=7')
    assert c.get_int('newkey') == 7

    c.parse_key_value('add', ['key1', 'key2', 'key3'])
    for key in ['key1', 'key2', 'key3']:
        assert c.get_bool(key)

    c.parse_key_value('lock', 'key*')
    for key in ['key1', 'key2', 'key3']:
        assert key in c.locked

    initial_rounds = c.max_iteration
    new_rounds = initial_rounds + 1
    c.parse_key_value('rounds', new_rounds)
    assert c.max_iteration == new_rounds

    # Test standard put
    c.parse_key_value('putkey', 'putvalue')
    assert c['putkey'] == 'putvalue'

    # Test sections
    c.parse_key_value('object', {'moon': {'add': 'cheese'}})
    assert 'moon' in c.objects
    assert 'moon' not in c


def test_blacklist():
    c = Configuration(verbose=True)
    c.options['foo'] = 'bar'
    c.blacklist('foo')
    assert 'foo' in c.locked and 'foo' in c.disabled and 'foo' in c.blacklisted

    c.options['locked_val'] = 'a'
    c.lock('locked_val')
    with log.log_to_list() as log_list:
        c.blacklist('locked_val')
    assert "Cannot blacklist locked option" in log_list[0].msg

    old_level = log.level
    log.setLevel("INFO")
    c.verbose = False
    with log.log_to_list() as log_list:
        c.blacklist('locked_val')
    assert len(log_list) == 0
    log.setLevel(old_level)

    # Blacklisted options are skipped over...
    c.blacklist('foo')
    assert 'foo' in c.blacklisted


def test_whitelist():
    c = Configuration(verbose=True)
    c.blacklist('foo')
    assert 'foo' in c.blacklisted
    c.whitelist('foo')
    assert 'foo' not in c.blacklisted
    assert 'foo' not in c.locked
    assert 'foo' in c.disabled

    c.lock('bar')
    with log.log_to_list() as log_list:
        c.whitelist('bar')
    assert "Cannot whitelist locked option" in log_list[0].msg

    old_level = log.level
    log.setLevel("INFO")
    c.verbose = False
    with log.log_to_list() as log_list:
        c.whitelist('bar')
    log.setLevel(old_level)
    assert len(log_list) == 0


def test_lock():
    c = Configuration()
    c.lock('foo')
    assert 'foo' in c.locked
    # Locked options are skipped over
    c.lock('foo')


def test_unlock():
    c = Configuration()
    # Blacklisted options cannot be unlocked
    c.blacklist('foo')
    c.unlock('foo')
    assert 'foo' in c.locked

    c.lock('bar')
    c.unlock('bar')
    assert 'bar' not in c.locked

    # Unlock a nonexistent options
    c.unlock('baz')
    assert 'baz' not in c.locked


def test_forget():
    c = Configuration(verbose=True)
    c.lock('foo')
    with log.log_to_list() as log_list:
        c.forget('foo')
    assert "Cannot forget locked option" in log_list[0].msg

    c.verbose = False
    old_level = log.level
    log.setLevel("INFO")
    with log.log_to_list() as log_list:
        c.forget('foo')
    log.setLevel(old_level)
    assert len(log_list) == 0

    c.blacklist('bar')
    c.blacklist('baz')
    c.forget('blacklist')
    assert 'bar' not in c.blacklisted and 'baz' not in c.blacklisted
    assert 'bar' in c.disabled and 'baz' in c.disabled

    c.conditions.options['value1=2'] = {'add': '3'}
    c.forget('conditions')
    assert len(c.conditions) == 0

    c.forget('forget_me_not')
    assert 'forget_me_not' in c.disabled


def test_recall():
    c = Configuration(verbose=True)
    c.lock('foo')
    with log.log_to_list() as log_list:
        c.recall('foo')
    assert "Cannot recall locked option" in log_list[0].msg

    c.verbose = False
    old_level = log.level
    log.setLevel("INFO")
    with log.log_to_list() as log_list:
        c.recall('foo')
    log.setLevel(old_level)
    assert len(log_list) == 0

    c.recall('bar')
    assert 'bar' not in c.disabled
    c.forget('bar')
    assert 'bar' in c.disabled
    c.recall('bar')
    assert 'bar' not in c.disabled


def test_has_option(initialized_configuration):
    c = initialized_configuration
    assert c.has_option('testvalue1')
    c.blacklist('testvalue1')
    assert not c.has_option('testvalue1')


def test_is_configured(initialized_configuration):
    c = initialized_configuration
    assert c.is_configured('options1')
    assert not c.is_configured('options2')


def test_set_option():
    c = Configuration()
    c.set_option('test_key')
    assert c['test_key']

    c.set_option('another_key', 'foo')
    assert c['another_key'] == 'foo'
    assert isinstance(c.options['another_key'], dict)


def test_get_options(initialized_configuration):
    c = initialized_configuration
    assert c.get_options('testvalue1', default='a_default') == 'a_default'
    options = c.get_options('options1')
    assert isinstance(options, dict)
    assert 'value' not in options


def test_is_locked(initialized_configuration):
    c = initialized_configuration
    assert c.is_locked('testvalue3')


def test_is_disabled(initialized_configuration):
    c = initialized_configuration
    c.disabled.add('disable_me')
    assert c.is_disabled('disable_me')


def test_is_blacklisted(initialized_configuration):
    c = initialized_configuration
    c.blacklist('testvalue1')
    assert c.is_blacklisted('testvalue1')


def test_add_new_branch():
    c = Configuration()
    c.add_new_branch('foo=bar')
    assert c['foo'] == 'bar'
    assert c.options['foo'] == {'value': 'bar'}
    c.add_new_branch('test_key', 1)
    assert c['test_key'] == 1


def test_get_keys(initialized_configuration):
    c = initialized_configuration
    assert c.get_keys() == ['options1', 'options2', 'serial', 'fits']
    assert c.get_keys('options1') == ['value', 'suboptions1', 'suboptions2']


def test_get_preserved_header_keys(initialized_configuration):
    c = initialized_configuration
    assert c.get_preserved_header_keys() == {'STORE1', 'STORE2'}


def test_preserve_header_keys(initialized_configuration, fits_header):
    c = initialized_configuration
    c.read_fits(fits_header)
    c.preserve_header_keys()
    assert c.fits.preserved_cards['STORE1'] == (1, 'Stored value 1')
    assert c.fits.preserved_cards['STORE2'] == (2, 'Stored value 2')
    assert 'STORE3' not in c.fits.preserved_cards


def test_get_filepath():
    c = Configuration()
    c.set_instrument('hawc_plus')
    c.put('config_file', 'default.cfg')

    assert c.get_filepath('foo') is None
    assert os.path.isfile(c.get_filepath('config_file', get_all=False))
    files = c.get_filepath('config_file', get_all=True)
    assert isinstance(files, list)
    for filename in files:
        assert os.path.isfile(filename)

    assert os.path.isfile(
        c.get_filepath('foo', default='default.cfg', get_all=False))
    assert c.get_filepath('foo', default='foo.cfg') is None


def test_purge(initialized_configuration):
    c = initialized_configuration
    c.purge('options1.suboptions2')
    assert 'suboptions2' not in c.options['options1']
    c.purge('options1')
    assert 'options1' not in c.options
    # purge a non-existent value
    c.purge('options1')


def test_get_flat_alphabetical(initialized_configuration):
    c = initialized_configuration
    c.disabled.add('testvalue1')
    a = c.get_flat_alphabetical()
    del a['configpath']
    assert a == {'fits.addkeys': ['STORE1', 'STORE2'],
                 'options1': 'True',
                 'options1.suboptions1.v1': '1',
                 'options1.suboptions1.v2': '2',
                 'options1.suboptions2.v1': '3',
                 'options1.suboptions2.v2': '3',
                 'options2.suboptions1.v1': 'c',
                 'options2.suboptions1.v2': 'd',
                 'options2.suboptions2.v1': 'e',
                 'options2.suboptions2.v2': 'f',
                 'options2.value1': 'a',
                 'options2.value2': 'b',
                 'refdec': '{?fits.OBSDEC}',
                 'refra': '{?fits.OBSRA}',
                 'refsrc': '{?fits.OBJECT}',
                 'serial.1-5.add': 'scans1to5',
                 'serial.2-3.add': 'scans2to3',
                 'switch1': 'True',
                 'switch2': 'False',
                 'testvalue1': 'foo',
                 'testvalue2': 'bar',
                 'testvalue3': 'lock_me'}


def test_get_active_options(initialized_configuration):
    c = initialized_configuration
    options = c.get_active_options()
    assert 'value1' in options['options2']
    c.disabled.add('options2.value1')
    options = c.get_active_options()
    assert 'value1' not in options['options2']


def test_order_options(initialized_configuration):
    c = initialized_configuration
    c.order_options()


def test_edit_header(initialized_configuration, fits_header):
    c = initialized_configuration
    c.disabled.add('testvalue1')
    h = fits.Header()
    c.edit_header(h)
    config_values = json.loads(h['CNFGVALS'])
    del config_values['configpath']
    assert config_values == {
        'options1': {
            'suboptions1': {'v1': '1', 'v2': '2'},
            'suboptions2': {'v1': '3', 'v2': '3'}},
        'refdec': '{?fits.OBSDEC}',
        'refra': '{?fits.OBSRA}',
        'refsrc': '{?fits.OBJECT}',
        'switch1': 'True',
        'switch2': 'False',
        'testvalue2': 'bar',
        'testvalue3': 'lock_me',
        'fits': {'addkeys': ['STORE1', 'STORE2']},
        'options2': {
            'value1': 'a',
            'value2': 'b',
            'suboptions1': {'v1': 'c', 'v2': 'd'},
            'suboptions2': {'v1': 'e', 'v2': 'f'}},
        'serial': {'1-5': {'add': 'scans1to5'}, '2-3': {'add': 'scans2to3'}},
        'aliases': {
            'final': 'iteration.-1',
            'i': 'iteration',
            'i1': 'iteration.1',
            'o1s1': 'options1.suboptions1',
            'o1s2': 'options1.suboptions2',
            'o2s1': 'options2.suboptions1',
            'o2s2': 'options2.suboptions2'},
        'conditions': {
            'aug2021': {'final': {'add': 'lastiteration'}},
            'jul2020': {'final': {'forget': 'testvalue2'}},
            'refra>12': {'add': 'switch3'},
            'switch2': {'final': {'forget': 'testvalue1'}},
            'switch3': {'o2s2': {'v2': 'z'}}},
        'objects': {
            'asphodel': {'add': 'finally'},
            'source1': {'add': 'src1'},
            'source2': {'testvalue1': 'baz'}},
        'iterations': {
            '0': {'6': {'o1s1': {'v1': '20'}}},
            '2': {'o1s1': {'v1': '10'}}},
        'dates': {
            '*--*': {'add': 'alldates'},
            '2020-07-01--2020-07-31': {'add': 'jul2020'},
            '2021-08-01--2021-08-31': {'add': 'aug2021'}}}

    c.read_fits(fits_header)
    h = fits.Header()
    c.edit_header(h)
    config_values = json.loads(h['CNFGVALS'])
    del config_values['configpath']
    assert 'OBSRA' in c.options['fits']
    assert 'OBSRA' not in config_values['fits']


def test_add_preserved_header_keys(initialized_configuration, fits_header):
    c = initialized_configuration
    h = fits.Header()
    c.add_preserved_header_keys(h)
    assert len(h) == 0

    c.read_fits(fits_header)
    c.preserve_header_keys()
    h = fits.Header()
    c.add_preserved_header_keys(h)
    assert h['STORE1'] == 1
    assert h['STORE2'] == 2
    assert 'STORE3' not in h


def test_set_outpath(tmpdir):
    c = Configuration()
    c.set_outpath()
    assert c.work_path == os.getcwd()

    temp_directory = str(tmpdir.mkdir('config_testing_'))
    subdir = os.path.join(temp_directory, 'outpath')
    c.put('outpath', subdir)
    with pytest.raises(ValueError) as err:
        c.set_outpath()
    assert 'does not exist' in str(err.value)

    c.put('outpath.create', True)
    c.set_outpath()
    assert os.path.isdir(subdir)


def test_configuration_difference(initialized_configuration, fits_header):
    c = initialized_configuration
    c.read_fits(fits_header)
    c2 = c.copy()
    c.disabled.add('testvalue2')
    del c2.options['testvalue1']
    c2.options['random_key'] = 'random_value'
    c2.options['options1']['suboptions2']['v2'] = '4'
    c2.options['testvalue2'] = 'abcdefg'
    del c2.aliases.options['o1s2']
    c2.conditions.options['switch2']['final']['forget'] = 'testvalue4'
    c2.fits.options['STORE1'] = 'abc'
    c2.dates.options['2021-08-01--2021-08-31'] = {'add': 'august2021'}
    c2.objects.options['source4'] = {'add': 'another_source'}
    diff = c.configuration_difference(c2)
    assert diff.options == {'testvalue1': 'foo',
                            'options1': {'suboptions2': {'v2': '3'}}}
    assert diff.fits.options == {'STORE1': '1'}
    assert diff.objects.options == {}  # Not in the original
    assert diff.conditions.options == {
        'switch2': {'final': {'forget': 'testvalue1'}}}
    assert diff.dates.options == {'2021-08-01--2021-08-31': {'add': 'aug2021'}}
    assert diff.aliases.options == {'o1s2': 'options1.suboptions2'}


def test_lock_rounds():
    c = Configuration()
    c.lock_rounds(99)
    assert c.max_iteration == 99
    c.lock_rounds(7)  # previously locked at 99
    assert c.max_iteration == 99


def test_check_trigger(initialized_configuration):
    c = initialized_configuration
    assert c.check_trigger('testvalue1=foo')
    assert not c.check_trigger('testvalue1=bar')

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.example.info.info import ExampleInfo


def test_init():
    info = ExampleInfo()
    assert info.name == 'example'


def test_get_name():
    info = ExampleInfo()
    assert info.get_name() == 'example'
    info.instrument = None
    assert info.get_name() == 'example'


def test_get_file_id():
    assert ExampleInfo.get_file_id() == 'EXPL'

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.scan.source_models.fits_properties.fits_properties import \
    FitsProperties


def test_init():
    f = FitsProperties()
    assert f.creator == 'SOFSCAN'
    assert 'Universities Space Research Association' in f.copyright
    assert f.filename == ''
    assert f.object_name == ''
    assert f.telescope_name == ''
    assert f.instrument_name == ''
    assert f.observer_name == ''
    assert f.observation_date_string == ''


def test_referenced_attributes():
    f = FitsProperties()
    assert f.referenced_attributes == set([])


def test_copy():

    class FitsPropertiesRef(FitsProperties):
        def __init__(self):
            super().__init__()
            self.array = np.arange(10)  # Has a copy method
            self.ref_array = np.arange(10)

        @property
        def referenced_attributes(self):
            return {'ref_array'}

    f = FitsPropertiesRef()
    f.instrument_name = 'foo'
    f2 = f.copy()
    assert f2 == f and f is not f2
    assert f.ref_array is f2.ref_array
    assert f.array is not f2.array and np.allclose(f.array, f2.array)


def test_eq():
    f = FitsProperties()
    f.instrument_name = 'foo'
    f2 = f.copy()
    assert f == f and f2 == f
    f2.instrument_name = 'bar'
    assert f != f2
    assert f != 1


def test_set_filename():
    f = FitsProperties()
    f.set_filename('foo')
    assert f.filename == 'foo'


def test_set_creator_name():
    f = FitsProperties()
    f.set_creator_name('foo')
    assert f.creator == 'foo'


def test_set_copyright():
    f = FitsProperties()
    f.set_copyright('foo')
    assert f.copyright == 'foo'


def test_set_object_name():
    f = FitsProperties()
    f.set_object_name('foo')
    assert f.object_name == 'foo'


def test_set_telescope_name():
    f = FitsProperties()
    f.set_telescope_name('foo')
    assert f.telescope_name == 'foo'


def test_set_instrument_name():
    f = FitsProperties()
    f.set_instrument_name('foo')
    assert f.instrument_name == 'foo'


def test_set_observer_name():
    f = FitsProperties()
    f.set_observer_name('foo')
    assert f.observer_name == 'foo'


def test_set_observation_date_string():
    f = FitsProperties()
    f.set_observation_date_string('foo')
    assert f.observation_date_string == 'foo'


def test_parse_header():
    f = FitsProperties()
    header = fits.Header()
    header['OBJECT'] = 'Moon'
    header['TELESCOP'] = 'Sofia'
    header['INSTRUME'] = 'HAWC+'
    header['OBSERVER'] = 'Foo Bar'
    header['DATE-OBS'] = '2022-02-22'
    f.parse_header(header)
    assert f.object_name == 'Moon'
    assert f.telescope_name == 'Sofia'
    assert f.instrument_name == 'HAWC+'
    assert f.observer_name == 'Foo Bar'
    assert f.observation_date_string == '2022-02-22'


def test_edit_header():
    f = FitsProperties()
    f.object_name = 'Moon'
    f.telescope_name = 'Sofia'
    f.instrument_name = 'HAWC+'
    f.observer_name = 'Foo Bar'
    f.observation_date_string = '2022-02-22'
    header = fits.Header()
    f.edit_header(header)
    assert header['OBJECT'] == 'Moon'
    assert header['TELESCOP'] == 'Sofia'
    assert header['INSTRUME'] == 'HAWC+'
    assert header['OBSERVER'] == 'Foo Bar'
    assert header['DATE-OBS'] == '2022-02-22'
    assert header['CREATOR'] == f.creator

    assert header.comments['OBJECT'] == "Observed object's name."
    assert header.comments['TELESCOP'] == 'Name of telescope.'
    assert header.comments['INSTRUME'] == 'Name of instrument used.'
    assert header.comments['OBSERVER'] == 'Name of observer(s).'
    assert header.comments['DATE-OBS'] == 'Start of observation.'
    assert f.copyright.startswith(header.comments['CREATOR'])


def test_get_table_entry():
    assert FitsProperties().get_table_entry('foo') is None


def test_info():
    f = FitsProperties()
    f.filename = 'foo'
    f.object_name = 'bar'
    s = f.info(header_string='baz')
    assert s == ' Image File: foo. -> \n\n[bar]\nbaz'


def test_brief():
    f = FitsProperties()
    f.object_name = 'foo'
    s = f.brief(header_string='bar')
    assert s == '[foo]\nbar'


def test_reset_processing():
    f = FitsProperties()
    f0 = f.copy()
    f.reset_processing()  # Nothing happens
    assert f == f0

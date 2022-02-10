# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from sofia_redux.scan.reduction.version import ReductionVersion


def test_init():
    rv = ReductionVersion()
    assert rv.home == '.'
    assert rv.work_path == '.'


def test_add_history(mocker):
    mocker.patch.object(ReductionVersion, 'version', '1.2.3')
    mocker.patch.object(ReductionVersion, 'revision', '')
    header = fits.Header()
    ReductionVersion.add_history(header)
    assert 'Reduced: SOFSCAN v1.2.3' in str(header['HISTORY'])


def test_get_full_version(mocker):
    mocker.patch.object(ReductionVersion, 'version', '1.2.3')
    mocker.patch.object(ReductionVersion, 'revision', None)
    assert ReductionVersion.get_full_version() == '1.2.3'

    mocker.patch.object(ReductionVersion, 'revision', '')
    assert ReductionVersion.get_full_version() == '1.2.3'

    mocker.patch.object(ReductionVersion, 'revision', 'dev')
    assert ReductionVersion.get_full_version() == '1.2.3 (dev)'

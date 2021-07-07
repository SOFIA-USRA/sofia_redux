# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import stat
import sys

import pytest
import getpass
from sofia_redux.toolkit.utilities.func import goodfile


@pytest.mark.skipif(getpass.getuser().strip().lower() == 'root',
                    reason='Cannot test as root user')
@pytest.mark.skipif(sys.platform == 'win32',
                    reason='Cannot test on windows')
def test_standard(tmpdir):
    filename = str(tmpdir.mkdir('test_goodfile').join('testfile.txt'))
    with open(filename, 'w') as f:
        print('hello', file=f)

    assert not goodfile(1, verbose=True)

    os.chmod(filename, ~stat.S_IRUSR)
    assert not goodfile(filename, read=True, write=False, execute=False,
                        verbose=True)
    os.chmod(filename, stat.S_IRUSR)
    assert goodfile(filename, read=True, write=False, execute=False,
                    verbose=True)
    os.chmod(filename, ~stat.S_IWUSR)
    assert not goodfile(filename, read=False, write=True, execute=False,
                        verbose=True)
    os.chmod(filename, stat.S_IWUSR)
    assert goodfile(filename, read=False, write=True, execute=False,
                    verbose=True)
    os.chmod(filename, ~stat.S_IXUSR)
    assert not goodfile(filename, read=False, write=False, execute=True,
                        verbose=True)
    os.chmod(filename, stat.S_IXUSR)
    assert goodfile(filename, read=False, write=False, execute=True,
                    verbose=True)
    os.remove(filename)
    assert not goodfile(filename, read=False, write=False, execute=False,
                        verbose=True)

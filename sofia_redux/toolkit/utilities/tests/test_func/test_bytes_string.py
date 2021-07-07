# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.func import bytes_string


def test_standard():
    assert bytes_string(0) == '0B'
    exponents = np.arange(9) * 3
    values = [2 * (10 ** int(e)) for e in exponents]
    bvals = [bytes_string(v) for v in values]

    assert bvals == ['2.0B', '1.95KB', '1.91MB', '1.86GB', '1.82TB',
                     '1.78PB', '1.73EB', '1.69ZB', '1.65YB']

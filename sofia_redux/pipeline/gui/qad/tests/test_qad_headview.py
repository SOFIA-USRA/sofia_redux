# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the QAD HeaderViewer class."""

from astropy import log
from astropy.io import fits
from astropy.io.fits.tests import FitsTestCase
import pytest

from sofia_redux.pipeline.gui.qad.qad_headview import HeaderViewer

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


@pytest.mark.skipif("not HAS_PYQT5")
class TestHeaderViewer(object):
    """Test the QAD header viewer."""
    @pytest.fixture(autouse=True, scope='function')
    def mock_app(self, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)

    def make_file(self, fname='test0.fits'):
        """Retrieve a test FITS file."""
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data(fname)
        return ffile

    def test_headview(self, qtbot):
        try:
            import pandas
            log.debug('Pandas version: {}'.format(pandas.__version__))
            has_pandas = True
        except ImportError:
            has_pandas = False

        hv = HeaderViewer()
        qtbot.addWidget(hv)

        ffile1 = self.make_file()
        ffile2 = self.make_file('blank.fits')

        headers = {}
        nrow = 0
        for fpath in [ffile1, ffile2]:
            hdul = fits.open(fpath)
            headers[fpath] = []
            for hdu in hdul:
                headers[fpath].append(hdu.header)
                nrow += 1

        hv.load(headers)

        # test the table function

        # with no filter text
        hv.table()
        html = hv.textEdit.toHtml()
        assert '<table' not in html

        # with filter text
        hv.findText.setText('NAXIS, NAXIS1')
        hv.table()
        html = hv.textEdit.toHtml()
        if has_pandas:
            assert '<table' in html
            # number of rows is number of
            # extensions + 1 for the header
            assert html.count('<tr>') == nrow + 1
            # number of cells is number of rows * 4
            # (index, filename, naxis, naxis1)
            assert html.count('<td>') == 4 * (nrow + 1)

        # reset
        hv.findText.setText('')
        hv.table()
        html = hv.textEdit.toHtml()
        assert '<table' not in html

        # also test when headers are not in a list
        headers[ffile1] = headers[ffile1][0]
        headers[ffile2] = headers[ffile2][0]
        nrow = 2

        hv.load(headers)
        hv.findText.setText('NAXIS, NAXIS1')
        hv.table()
        html = hv.textEdit.toHtml()
        if has_pandas:
            assert '<table' in html
            # number of rows is number of
            # extensions + 1 for the header
            assert html.count('<tr>') == nrow + 1
            # number of cells is number of rows * 4
            # (index, filename, naxis, naxis1)
            assert html.count('<td>') == 4 * (nrow + 1)

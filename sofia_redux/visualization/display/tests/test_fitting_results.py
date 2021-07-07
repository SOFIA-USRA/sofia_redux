#  Licensed under a 3-clause BSD style license - see LICENSE.rst
import pandas as pd
import pytest
import copy
import os

from astropy.units import Quantity
from matplotlib import axes as mpa
import numpy as np

from sofia_redux.visualization.display.fitting_results import FittingResults
from sofia_redux.visualization.utils.model_fit import ModelFit

PyQt5 = pytest.importorskip('PyQt5')


@pytest.fixture(scope='function')
def fit_data():
    # mock models with parameters from astropy
    # Gaussian/linear compound fits
    class Model1:
        amplitude_0 = Quantity(-33.56199975)
        mean_0 = Quantity(32.88534166)
        stddev_0 = Quantity(0.06703216)
        slope_1 = Quantity(4.10364425)
        intercept_1 = Quantity(1.18345362)

    class Model2:
        amplitude_0 = Quantity(-33.56199975)
        mean_0 = Quantity(32.88534166)
        stddev_0 = Quantity(0.06703216)
        slope_1 = Quantity(4.10364425)
        intercept_1 = Quantity(1.18345362)

    class Model3:
        amplitude_0 = Quantity(-39.68602448)
        mean_0 = Quantity(32.90751934)
        stddev_0 = Quantity(0.05113242)
        slope_1 = Quantity(25.94818499)
        intercept_1 = Quantity(-708.0214212)

    class Model4:
        amplitude_0 = Quantity(-33.28810285)
        mean_0 = Quantity(32.92237997)
        stddev_0 = Quantity(0.04374616)
        slope_1 = Quantity(10.76873205)
        intercept_1 = Quantity(-220.08146439)

    fit_params = {'file1.fits': {0: {'fit': Model1, 'x_field': 'wavepos',
                                     'y_field': 'spectral_flux',
                                     'lower_limit': 32.63,
                                     'upper_limit': 33.13,
                                     'baseline': 136.13},
                                 1: {'fit': Model2, 'x_field': 'wavepos',
                                     'y_field': 'spectral_flux',
                                     'lower_limit': 32.63,
                                     'upper_limit': 33.13,
                                     'baseline': 142.56}},
                  'file2.fits': {0: {'fit': Model3, 'x_field': 'wavepos',
                                     'y_field': 'spectral_flux',
                                     'lower_limit': 32.63,
                                     'upper_limit': 33.13,
                                     'baseline': 145.86}},
                  'file3.fits': {0: {'fit': Model4, 'x_field': 'wavepos',
                                     'y_field': 'spectral_flux',
                                     'lower_limit': 32.63,
                                     'upper_limit': 33.13,
                                     'baseline': 134.45}}}
    return fit_params


@pytest.fixture(scope='function')
def expected_table():
    expected = [['filename', 'order', 'x_field', 'y_field', 'mean',
                 'stddev', 'amplitude', 'baseline', 'base_intercept',
                 'base_slope', 'lower_limit', 'upper_limit'],
                ['file1.fits', '0', 'wavepos', 'spectral_flux',
                 '32.88534166', '0.06703216', '-33.56199975', '136.13',
                 '1.18345362', '4.10364425', '32.63', '33.13'],
                ['file1.fits', '1', 'wavepos', 'spectral_flux',
                 '32.88534166', '0.06703216', '-33.56199975', '142.56',
                 '1.18345362', '4.10364425', '32.63', '33.13'],
                ['file2.fits', '0', 'wavepos', 'spectral_flux',
                 '32.90751934', '0.05113242', '-39.68602448', '145.86',
                 '-708.0214212', '25.94818499', '32.63', '33.13'],
                ['file3.fits', '0', 'wavepos', 'spectral_flux',
                 '32.92237997', '0.04374616', '-33.28810285', '134.45',
                 '-220.08146439', '10.76873205', '32.63', '33.13']]
    return expected


class TestFittingResults(object):
    def test_init(self, empty_view):
        fr = FittingResults(empty_view)
        assert isinstance(fr, PyQt5.QtWidgets.QDialog)
        assert isinstance(fr.model_fits, list)
        assert len(fr.model_fits) == 0
        assert isinstance(fr.table_header, list)
        assert len(fr.table_header) == 0
        assert isinstance(fr.ax, mpa.Axes)

    def test_add_results(self, empty_view, fit_data, gauss_model_fit,
                         moffat_model_fit, line_model_fit):
        fr = FittingResults(empty_view)

        # add data from 3 files, one with two orders
        fr.add_results(gauss_model_fit)
        assert fr.model_fits[0] == gauss_model_fit
        assert fr.table_widget.rowCount() == 1
        assert fr.table_widget.columnCount() == 10

        # check cell contents
        columns = ['Show', 'Order', 'X Field', 'Y Field', 'Mid Point',
                   'FWHM', 'Amplitude', 'Baseline', 'Slope', 'Type']
        table_cols = [fr.table_widget.horizontalHeaderItem(j).text()
                      for j in range(fr.table_widget.columnCount())]
        assert set(table_cols) == set(columns)

        # test for row values
        def row_values(i):
            table_values = list()
            for j in range(fr.table_widget.columnCount()):
                try:
                    item = fr.table_widget.item(i, j).text()
                except AttributeError:
                    item = None
                table_values.append(item)
            return table_values

        fit = gauss_model_fit.get_fit()
        for i in range(fr.table_widget.rowCount()):
            filename = gauss_model_fit.model_id
            assert fr.table_widget.verticalHeaderItem(i).text() == filename
            values = [None, str(gauss_model_fit.get_order()),
                      f'{gauss_model_fit.get_fields("x")} '
                      f'[{gauss_model_fit.get_units("x")}]',
                      f'{gauss_model_fit.get_fields("y")} '
                      f'[{gauss_model_fit.get_units("y")}]',
                      f'{fit.mean_0.value:.5g}',
                      f'{fit.amplitude_0.value:.5g}',
                      f'{gauss_model_fit.get_baseline():.5g}',
                      f'{gauss_model_fit.get_fwhm():.5g}',
                      f'{gauss_model_fit.get_mid_point():.5g}',
                      'gauss, linear',
                      f'{fit.slope_1.value:.5g}',
                      ]
            table_values = row_values(i)
            assert set(table_values) == set(values)

        # add one more result: does not overwrite previous
        fr.add_results(copy.copy(gauss_model_fit))
        assert fr.table_widget.rowCount() == 2
        assert len(fr.model_fits) == 2

        # add another result with a different fit type, but
        # matching display params
        fr.add_results(moffat_model_fit)
        assert fr.table_widget.rowCount() == 3
        assert len(fr.model_fits) == 3
        table_values = row_values(2)
        assert 'NA' not in table_values
        assert 'moffat, linear' in table_values

        # add another result with different display parameters
        fr.add_results(line_model_fit)
        assert fr.table_widget.rowCount() == 4
        assert len(fr.model_fits) == 4
        table_values = row_values(3)
        assert 'NA' in table_values
        assert 'linear' in table_values

    def test_clear_fit(self, empty_view, gauss_model_fit):
        fr = FittingResults(empty_view)

        # no data: table has one dummy row, no parameters stored
        assert fr.table_widget.rowCount() == 1
        assert len(fr.model_fits) == 0

        # populate table
        fr.add_results(gauss_model_fit)
        assert fr.table_widget.rowCount() == 1
        assert len(fr.model_fits) == 1

        # clear it: no rows, no parameters
        fr.clear_fit()
        assert fr.table_widget.rowCount() == 0
        assert len(fr.model_fits) == 0

        # repopulate: 4 rows, one set of parameters again
        fr.add_results(gauss_model_fit)
        assert fr.table_widget.rowCount() == 1
        assert len(fr.model_fits) == 1

    def test_save_results(self, mocker, tmpdir, empty_view,
                          gauss_model_fit, moffat_model_fit,
                          expected_table):
        fr = FittingResults(empty_view)
        fr.add_results(gauss_model_fit)
        fr.add_results(moffat_model_fit)
        num_columns = 15

        # test canceled dialog - no error
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog,
                            'getSaveFileName',
                            return_value=[''])
        fr.save_results()

        # mock file dialog to return tmp path
        outname = str(tmpdir.join('test.csv'))
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog,
                            'getSaveFileName',
                            return_value=[outname])
        fr.save_results()
        assert os.path.isfile(outname)

        # check values saved against expected
        saved = pd.read_csv(outname)
        assert saved.shape == (2, num_columns)

        # test save with one row selected: only that row should be saved
        outname = str(tmpdir.join('test2.csv'))
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog,
                            'getSaveFileName',
                            return_value=[outname])

        # mock first row selection
        class TestIndex(object):
            def row(self):
                return 0

        class TestModel(object):
            def selectedRows(self):
                return [TestIndex()]
        mocker.patch.object(fr.table_widget, 'selectionModel',
                            return_value=TestModel())
        fr.save_results()
        assert os.path.isfile(outname)
        saved = pd.read_csv(outname)
        assert saved.shape == (1, num_columns)

    @pytest.mark.parametrize('kind,dtype,count',
                             [('string', dict, 1), ('dict', dict, 1),
                              ('list', list, 1), ('html', str, 1),
                              ('bad', None, 0)])
    def test_format_parameters(self, empty_view, gauss_model_fit,
                               kind, dtype, count):
        fr = FittingResults(empty_view)
        fr.add_results(gauss_model_fit)

        par = fr.format_parameters(kind)
        assert isinstance(par, list)
        assert len(par) == count
        if count > 0:
            assert isinstance(par[0], dtype)

    def test_change_units(self, empty_view, gauss_model_fit):
        fr = FittingResults(empty_view)
        fr.add_results(gauss_model_fit)
        gauss_model_fit.visible = True

        # without return_new, None is returned,
        # but units are still updated
        units = {'x': 'um', 'y': 'Jy'}
        updated = fr.change_units(units)
        assert updated is None
        assert gauss_model_fit.units == units

        units = {'x': 'nm', 'y': 'W/m2'}
        fr.change_units(units)
        assert gauss_model_fit.units == units

        # with return_new, model fits are returned
        units = {'x': 'um', 'y': 'Jy'}
        updated = fr.change_units(units, return_new=True)
        assert gauss_model_fit.units == units
        assert isinstance(updated, list)
        assert isinstance(updated[0], ModelFit)
        assert updated[0].visible

        # bad units hide the model, units stay the same
        bad_units = {'x': 'um', 'y': 'bad'}
        updated = fr.change_units(bad_units, return_new=True)
        assert updated[0].units == units
        assert not updated[0].visible

    def test_change_units_with_pane(self, empty_view, gauss_model_fit,
                                    one_dim_pane):
        fr = FittingResults(empty_view)
        fr.add_results(gauss_model_fit)

        # mismatched pane: fit not updated
        units = {'x': 'nm', 'y': 'W/m2'}
        one_dim_pane.units = units
        fr.change_units(units, panes=[one_dim_pane])
        assert gauss_model_fit.units != units

        # matching pane: fit updated
        units = {'x': 'nm', 'y': 'W/m2'}
        one_dim_pane.ax = gauss_model_fit.axis
        fr.change_units(units, panes=[one_dim_pane])
        assert gauss_model_fit.units == units

    def test_hide_all_fits(self, empty_view, gauss_model_fit,
                           moffat_model_fit, line_model_fit):
        # add some fits and make sure they're visible
        fits = [gauss_model_fit, moffat_model_fit, line_model_fit]
        fr = FittingResults(empty_view)
        fr.add_results(fits)
        for fit in fits:
            fit.visible = True

        # hide all
        fr.hide_all_fits()
        for fit in fits:
            assert not fit.visible

    def test_update_figure(self, empty_view, gauss_model_fit):
        fr = FittingResults(empty_view)
        gauss_model_fit.dataset = {'x': np.arange(10),
                                   'y': np.arange(10)}

        # figure and textview updated for passing fit
        fr._update_figure([gauss_model_fit])
        assert 'Pass' in fr.last_fit_values.toPlainText()
        assert len(fr.ax.lines) == 3

        # empty fits are ignored
        gauss_model_fit.status = 'Empty'
        fr._update_figure([gauss_model_fit])
        assert fr.last_fit_values.toPlainText() == ''
        assert len(fr.ax.lines) == 0

        # failed fits are reported, data is plotted
        gauss_model_fit.status = 'bad'
        fr._update_figure([gauss_model_fit])
        assert 'Bad' in fr.last_fit_values.toPlainText()
        assert len(fr.ax.lines) == 1

    def test_gather_models(self, empty_view, gauss_model_fit):
        fr = FittingResults(empty_view)

        # no error with no models
        assert fr.gather_models() == list()

        # add a model
        fr.add_results([gauss_model_fit])
        assert fr.gather_models() == [gauss_model_fit]

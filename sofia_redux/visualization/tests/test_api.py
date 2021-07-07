# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import pytest

from sofia_redux.visualization import eye, log

PyQt5 = pytest.importorskip('PyQt5')


class TestAPI(object):

    def test_full_run(self, empty_eye_app, tmp_path, forcast_fits_image):
        empty_eye_app.load(forcast_fits_image)
        empty_eye_app.add_panes(layout='rows', n_panes=3, kind='spectrum')
        empty_eye_app.assign_data('split')
        fields = {'x': 'wavepos', 'y': 'spectral_flux'}
        units = {'x': 'um', 'y': 'Jy'}
        orders_to_enable = [1, 3, 4, 6]
        panes = range(empty_eye_app.number_panes())
        models = empty_eye_app.models.keys()
        orders = dict.fromkeys(panes,
                               dict.fromkeys(models, orders_to_enable))
        scale = {'x': 'linear', 'y': 'log'}

        empty_eye_app.set_fields(fields=fields, panes='all')
        empty_eye_app.set_units(units=units, panes='all')
        empty_eye_app.set_orders(orders=orders)
        empty_eye_app.set_scale(scales=scale, panes='all')
        empty_eye_app.toggle_controls()
        empty_eye_app.generate()
        filename = tmp_path / 'full_run.png'
        empty_eye_app.save(filename)

        assert os.path.isfile(filename)

    def test_set_parent(self, empty_eye_app):
        # set a widget as the view parent
        parent = PyQt5.QtWidgets.QWidget()
        empty_eye_app.set_parent(parent)
        assert empty_eye_app.view.parent is parent

        # try setting something invalid - that works too
        empty_eye_app.set_parent('bad')
        assert empty_eye_app.view.parent == 'bad'

    def test_load_fail(self, empty_eye_app):
        with pytest.raises(FileNotFoundError):
            empty_eye_app.load(['/non/existent/files.fits'])
        with pytest.raises(TypeError):
            empty_eye_app.load([5092])

    def test_load(self, empty_eye_app, spectral_hduls,
                  spectral_filenames, capsys):
        # load HDULs
        empty_eye_app.load(spectral_hduls)
        assert len(empty_eye_app.models) == len(spectral_hduls)
        log_ = capsys.readouterr().out
        assert log_.count('Loading HDUList') == len(spectral_hduls)

        empty_eye_app.unload()

        # load filenames
        empty_eye_app.load(spectral_filenames)
        log_ = capsys.readouterr().out
        assert len(empty_eye_app.models) == len(spectral_filenames)
        assert (log_.count('Loading from filename')
                == len(spectral_filenames))

    def test_panes_grid(self, empty_eye_app):
        shape = (3, 2)
        empty_eye_app.add_panes(n_panes=6, layout='grid', kind='spectrum')
        assert empty_eye_app.number_panes() == 6
        assert empty_eye_app.get_pane_layout() == shape

    def test_panes_rows(self, empty_eye_app):
        shape = (6, 1)
        empty_eye_app.add_panes(n_panes=6, layout='rows', kind='spectrum')
        assert empty_eye_app.get_pane_layout() == shape

    def test_panes_cols(self, empty_eye_app):
        shape = (1, 6)
        empty_eye_app.add_panes(n_panes=6, layout='columns', kind='spectrum')
        assert empty_eye_app.get_pane_layout() == shape

    @pytest.mark.parametrize('kind', ['spectrum', 'onedim',
                                      ['spectrum', 'spectrum',
                                       'onedim', 'onedim']])
    def test_panes_valid_kind(self, empty_eye_app, kind):
        # TODO: 'image' and 'twodim' are also accepted here,
        #  but not yet implemented
        shape = (2, 2)
        empty_eye_app.add_panes(n_panes=4, kind=kind)
        assert empty_eye_app.number_panes() == 4
        assert empty_eye_app.get_pane_layout() == shape

    @pytest.mark.parametrize('kind', [None, 'bad',
                                      ['spectrum', 'spectrum',
                                       'bad', 'onedim'],
                                      ['spectrum', 'spectrum']])
    def test_panes_invalid_kind(self, empty_eye_app, kind):
        with pytest.raises(ValueError) as err:
            empty_eye_app.add_panes(n_panes=4, kind=kind)
        assert 'kind' in str(err)

    def test_assign_data_split(self, empty_eye_app, spectral_filenames):
        empty_eye_app.load(spectral_filenames)
        empty_eye_app.add_panes(n_panes=4, kind='spectrum')

        # 5 files, split evenly between panes; no conflicts in units
        empty_eye_app.assign_data('split')
        assert empty_eye_app.models_per_pane() == [2, 1, 1, 1]

    def test_assign_data_first(self, empty_eye_app, spectral_filenames,
                               capsys):
        empty_eye_app.load(spectral_filenames)
        empty_eye_app.add_panes(n_panes=4, layout='grid', kind='spectrum')
        empty_eye_app.assign_data('first')
        # 5 files, but only 3 are matching forcast,
        # 2 are mismatched exes and are discarded
        # all valid models are in first pane
        assert empty_eye_app.models_per_pane() == [3, 0, 0, 0]
        assert 'Incompatible units' in capsys.readouterr().err

    def test_assign_data_last(self, empty_eye_app, spectral_filenames):
        empty_eye_app.load(spectral_filenames)
        empty_eye_app.add_panes(n_panes=4, layout='grid', kind='spectrum')
        empty_eye_app.assign_data('last')
        # all valid models are in last pane
        assert empty_eye_app.models_per_pane() == [0, 0, 0, 3]

    def test_assign_data_bad_mode(self, empty_eye_app, spectral_filenames):
        empty_eye_app.load(spectral_filenames)
        empty_eye_app.add_panes(n_panes=4, layout='grid', kind='spectrum')
        with pytest.raises(ValueError) as err:
            empty_eye_app.assign_data('bad')
        assert 'Invalid data assignment mode' in str(err)

    def test_assign_data_assigned(self, empty_eye_app, spectral_filenames):
        empty_eye_app.load(spectral_filenames)
        empty_eye_app.add_panes(n_panes=4, layout='grid', kind='spectrum')

        # no indices provided
        with pytest.raises(ValueError) as err:
            empty_eye_app.assign_data('assigned')
        assert 'Invalid format' in str(err)

        # wrong number of indices
        indices = [1, 2, 3]
        with pytest.raises(ValueError) as err:
            empty_eye_app.assign_data('assigned', indices=indices)
        assert 'Length of `indices` must match' in str(err)

        # invalid index values
        indices = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError) as err:
            empty_eye_app.assign_data('assigned', indices=indices)
        assert 'Values in `indices` must be valid' in str(err)

        # valid index values
        indices = [1, 2, 1, 3, 3]
        empty_eye_app.assign_data('assigned', indices=indices)
        assert 'Values in `indices` must be valid' in str(err)
        assert empty_eye_app.models_per_pane() == [0, 2, 1, 2]

    def test_set_current_pane(self, empty_eye_app, spectral_filenames):
        empty_eye_app.load(spectral_filenames)
        empty_eye_app.add_panes(n_panes=4)
        # in range
        empty_eye_app.set_current_pane(1)
        assert empty_eye_app.view.figure.current_pane == 1
        # out of range - stays the same
        empty_eye_app.set_current_pane(4)
        assert empty_eye_app.view.figure.current_pane == 1

    def test_set_fields(self, populated_spectral_eye, capsys, caplog):
        fields = {'x': 'wavepos', 'y': 'spectral_flux', 'y_alt': None}
        populated_spectral_eye.set_fields(x_field=fields['x'],
                                          y_field=fields['y'],
                                          panes='all')

        current_fields = populated_spectral_eye.get_fields()
        for pane_fields in current_fields:
            assert pane_fields == fields

        new_fields = {'x': 'spectral_flux', 'y': 'spectral_error',
                      'y_alt': None}
        populated_spectral_eye.set_fields(x_field=new_fields['x'],
                                          y_field=new_fields['y'],
                                          panes=[0, ])
        current_fields = populated_spectral_eye.get_fields()
        assert current_fields == [new_fields, fields]

        # bad field name logs error but continues
        capsys.readouterr()
        bad_fields = {'x': 'wavepos', 'y': 'image_flux', 'y_alt': None}
        last_pane = populated_spectral_eye.view.figure.panes[-1]
        populated_spectral_eye.set_fields(x_field=bad_fields['x'],
                                          y_field=bad_fields['y'],
                                          panes=[last_pane])
        assert 'Invalid field provided for ' \
               'axis y: image_flux' in capsys.readouterr().out
        assert populated_spectral_eye.get_fields() == [new_fields, fields]

        new_fields = {'x': 90, 'y': 'image_flux'}
        with pytest.raises(TypeError):
            populated_spectral_eye.set_fields(x_field=new_fields['x'],
                                              y_field=new_fields['y'])

    def test_set_units(self, populated_spectral_eye):
        # requires consistent data to succeed - remove the exes pane
        populated_spectral_eye.remove_panes([1])
        units = {'x': 'um', 'y': 'Jy'}
        correct_units = {'x': 'um', 'y': 'Jy', 'y_alt': ''}

        populated_spectral_eye.set_fields(x_field='wavepos',
                                          y_field='spectral_flux')
        populated_spectral_eye.set_units(units=units)

        current_units = populated_spectral_eye.get_units()
        for pane_units in current_units:
            assert pane_units == correct_units

        new_units = {'x': 'nm', 'y': 'erg / (s cm2 Hz)'}
        new_correct_units = {'x': 'nm', 'y': 'erg / (s cm2 Hz)', 'y_alt': ''}
        populated_spectral_eye.set_units(units=new_units)
        current_units = populated_spectral_eye.get_units()
        for pane_units in current_units:
            assert pane_units == new_correct_units

    def test_set_bad_units(self, empty_eye_app):
        # bad unit format
        with pytest.raises(TypeError) as err:
            empty_eye_app.set_units('um')
        assert 'must be dict' in str(err)

    def test_set_orders(self, populated_multi_order_eye):
        orders_to_enable = [1, 3, 4, 6]
        panes = range(populated_multi_order_eye.number_panes())
        models = populated_multi_order_eye.models.keys()

        orders = dict.fromkeys(panes,
                               dict.fromkeys(models, orders_to_enable))
        populated_multi_order_eye.set_orders(orders=orders)
        current_orders = populated_multi_order_eye.get_orders()
        assert len(current_orders) == len(orders)
        for i, current_pane_orders in current_orders.items():
            for model_id, model_orders in current_pane_orders.items():
                assert set(model_orders).issubset(set(orders_to_enable))

    def test_set_bad_orders(self, empty_eye_app):
        # bad order format
        with pytest.raises(TypeError) as err:
            empty_eye_app.set_orders([1, 2])
        assert 'must be dict' in str(err)

    def test_set_scale(self, populated_spectral_eye):
        new_scale = {'x': 'linear', 'y': 'log'}
        correct_scale = {'x': 'linear', 'y': 'log', 'y_alt': None}
        populated_spectral_eye.set_scale(scales=new_scale, panes='all')

        scales = populated_spectral_eye.get_scale(panes='all')
        for scale in scales:
            assert scale == correct_scale

    def test_set_bad_scale(self, empty_eye_app):
        # bad scale format
        with pytest.raises(TypeError) as err:
            empty_eye_app.set_scale('log')
        assert 'must be dict' in str(err)

    def test_toggle_controls(self, empty_eye_app, capsys, mocker):
        # panel not visible, since show is mocked
        empty_eye_app.toggle_controls()
        empty_eye_app.toggle_cursor()
        empty_eye_app.toggle_file_panel()
        empty_eye_app.toggle_pane_panel()
        empty_eye_app.toggle_order_panel()
        empty_eye_app.toggle_axis_panel()
        empty_eye_app.toggle_plot_panel()
        empty_eye_app.toggle_analysis_panel()
        assert 'Showing controls' in capsys.readouterr().out
        assert 'Hide' in empty_eye_app.view.collapse_controls_button.toolTip()
        assert 'Hide' in empty_eye_app.view.collapse_cursor_button.toolTip()
        assert 'Hide' \
            in empty_eye_app.view.collapse_file_choice_button.toolTip()
        assert 'Hide' in empty_eye_app.view.collapse_pane_button.toolTip()
        assert 'Hide' in empty_eye_app.view.collapse_order_button.toolTip()
        assert 'Hide' in empty_eye_app.view.collapse_axis_button.toolTip()
        assert 'Hide' in empty_eye_app.view.collapse_plot_button.toolTip()
        assert 'Hide' in empty_eye_app.view.collapse_analysis_button.toolTip()

        mocker.patch.object(empty_eye_app.view.control_frame, 'isVisible',
                            return_value=True)
        mocker.patch.object(empty_eye_app.view.cursor_frame, 'isVisible',
                            return_value=True)
        mocker.patch.object(empty_eye_app.view.file_choice_panel, 'isVisible',
                            return_value=True)
        mocker.patch.object(empty_eye_app.view.pane_panel, 'isVisible',
                            return_value=True)
        mocker.patch.object(empty_eye_app.view.order_panel, 'isVisible',
                            return_value=True)
        mocker.patch.object(empty_eye_app.view.axis_panel, 'isVisible',
                            return_value=True)
        mocker.patch.object(empty_eye_app.view.plot_panel, 'isVisible',
                            return_value=True)
        mocker.patch.object(empty_eye_app.view.analysis_panel, 'isVisible',
                            return_value=True)
        empty_eye_app.toggle_controls()
        empty_eye_app.toggle_cursor()
        empty_eye_app.toggle_file_panel()
        empty_eye_app.toggle_pane_panel()
        empty_eye_app.toggle_order_panel()
        empty_eye_app.toggle_axis_panel()
        empty_eye_app.toggle_plot_panel()
        empty_eye_app.toggle_analysis_panel()
        assert 'Hiding controls' in capsys.readouterr().out
        assert 'Show' in empty_eye_app.view.collapse_controls_button.toolTip()
        assert 'Show' in empty_eye_app.view.collapse_cursor_button.toolTip()
        assert 'Show' \
               in empty_eye_app.view.collapse_file_choice_button.toolTip()
        assert 'Show' in empty_eye_app.view.collapse_pane_button.toolTip()
        assert 'Show' in empty_eye_app.view.collapse_order_button.toolTip()
        assert 'Show' in empty_eye_app.view.collapse_axis_button.toolTip()
        assert 'Show' in empty_eye_app.view.collapse_plot_button.toolTip()
        assert 'Show' in empty_eye_app.view.collapse_analysis_button.toolTip()

    def test_save_figure(self, populated_multi_order_eye, tmp_path, capsys):
        populated_multi_order_eye.assign_data('split')
        populated_multi_order_eye.generate()
        filename = tmp_path / 'multi_order.png'
        populated_multi_order_eye.save(filename)
        log_ = capsys.readouterr().out
        assert os.path.isfile(filename)
        assert 'Saving image' in log_

    def test_remove_data(self, populated_spectral_eye, capsys, mocker):
        # remove all panes and reassign one per model
        all_files = populated_spectral_eye.view.model_collection.copy()
        populated_spectral_eye.remove_panes()
        assert populated_spectral_eye.number_panes() == 0
        assert len(all_files) > 4

        populated_spectral_eye.add_panes(layout='rows', n_panes=len(all_files),
                                         kind='spectrum')
        populated_spectral_eye.assign_data('split')
        assert populated_spectral_eye.number_panes() == len(all_files)

        # remove a middle file
        populated_spectral_eye.remove_data(all_files[2])
        new_files = populated_spectral_eye.view.model_collection.copy()
        assert len(new_files) == len(all_files) - 1
        assert all_files[2] not in new_files
        assert len(populated_spectral_eye.view.figure.panes[0].models) == 1
        assert len(populated_spectral_eye.view.figure.panes[2].models) == 0
        assert len(populated_spectral_eye.view.figure.panes[-1].models) == 1

        # remove the first and last files
        populated_spectral_eye.remove_data([all_files[0], all_files[-1]])
        new_files = populated_spectral_eye.view.model_collection.copy()
        assert len(new_files) == len(all_files) - 3
        assert all_files[0] not in new_files
        assert all_files[-1] not in new_files
        assert len(populated_spectral_eye.view.figure.panes[0].models) == 0
        assert len(populated_spectral_eye.view.figure.panes[-1].models) == 0

        # call remove with nothing selected: nothing happens
        populated_spectral_eye.remove_data()
        test_files = populated_spectral_eye.view.model_collection.copy()
        assert test_files == new_files

        # mock selection of first remaining file -- should be removed
        mocker.patch.object(populated_spectral_eye.view,
                            'current_files_selected',
                            return_value=[new_files[0]])
        populated_spectral_eye.remove_data()
        test_files = populated_spectral_eye.view.model_collection.copy()
        assert len(test_files) == len(new_files) - 1
        assert new_files[0] not in test_files

        # remove all files: should warn but ignore already removed files
        populated_spectral_eye.remove_data(all_files)
        for i in range(len(all_files)):
            assert len(populated_spectral_eye.view.figure.panes[i].models) == 0
        assert capsys.readouterr().err.count('not found') == 4

    def test_display_selected(self, populated_spectral_eye, mocker):
        # remove all panes and add one back in
        all_files = populated_spectral_eye.view.model_collection.copy()
        populated_spectral_eye.remove_panes()
        populated_spectral_eye.add_panes()
        assert populated_spectral_eye.number_panes() == 1

        # nothing selected
        populated_spectral_eye.display_selected_model()
        assert len(populated_spectral_eye.view.figure.panes[0].models) == 0

        # mock selection of first file -- one model displayed
        mocker.patch.object(populated_spectral_eye.view,
                            'current_files_selected',
                            return_value=[all_files[0]])
        populated_spectral_eye.display_selected_model()
        assert len(populated_spectral_eye.view.figure.panes[0].models) == 1

        # now select first two -- 2 models displayed
        mocker.patch.object(populated_spectral_eye.view,
                            'current_files_selected',
                            return_value=all_files[0:2])
        populated_spectral_eye.display_selected_model()
        assert len(populated_spectral_eye.view.figure.panes[0].models) == 2

        # select a bad filename - raises error
        mocker.patch.object(populated_spectral_eye.view,
                            'current_files_selected',
                            return_value='bad_file')
        with pytest.raises(RuntimeError) as err:
            populated_spectral_eye.display_selected_model()
        assert 'Cannot locate model' in str(err)
        assert len(populated_spectral_eye.view.figure.panes[0].models) == 2

    def test_eye_logs(self, mocker, qtbot, qapp, capsys, tmpdir):
        mocker.patch.object(PyQt5.QtWidgets.QMainWindow, 'show',
                            return_value=None)
        mocker.patch.object(PyQt5.QtWidgets, 'QApplication',
                            return_value=qapp)
        # patch home directory, as used in system log setup
        base_loc = str(tmpdir.join('event_logs'))
        mocker.patch('os.path.expanduser', return_value=base_loc)

        # generate some simple log messages
        def log_test_messages():
            log.debug('test debug')
            log.info('test info')
            log.warning('test warning')
            log.error('test error')
            log.critical('test critical')

        # no arguments: terminal log level is critical, no system log
        app = eye.Eye()
        log_test_messages()
        capt = capsys.readouterr()
        app.close()

        for value in ['debug', 'info']:
            assert value not in capt.out
        for value in ['warning', 'error']:
            assert value not in capt.err
        assert 'critical' in capt.err
        assert not os.path.isdir(base_loc)

        # specify system logs
        class Args:
            filenames = []
            system_logs = True
            log_level = 'DEBUG'
        app = eye.Eye(Args)
        log_test_messages()
        app.close()

        assert os.path.isdir(base_loc)
        log_file = os.path.join(base_loc, os.listdir(base_loc)[0])
        with open(log_file) as fh:
            log_lines = ''.join(fh.readlines())
        for value in ['debug', 'info', 'warning', 'error', 'critical']:
            assert value in log_lines

        # mock an error in making the event_logs directory
        mocker.patch('os.makedirs', side_effect=IOError('test'))
        with pytest.raises(IOError) as err:
            eye.Eye(Args)
        assert 'Unable to create log directory' in str(err)

    def test_command_line_names(self, open_mock, spectral_filenames, capsys):
        # start the eye with filenames in the arguments
        class Args:
            filenames = spectral_filenames
            log_level = 'INFO'
        app = eye.Eye(Args)
        assert app.view.model_collection == spectral_filenames
        assert 'Reading in files' in capsys.readouterr().out
        app.close()

    def test_open_eye(self, mocker, empty_eye_app, capsys):
        open_mock = mocker.patch.object(empty_eye_app.view, 'open_eye')
        empty_eye_app.open_eye()
        assert open_mock.called
        assert 'Opening Eye' in capsys.readouterr().out

    def test_reset(self, populated_spectral_eye):
        assert len(populated_spectral_eye.view.model_collection) > 0
        assert populated_spectral_eye.number_panes() > 0
        populated_spectral_eye.reset()
        assert len(populated_spectral_eye.view.model_collection) == 0
        assert populated_spectral_eye.number_panes() == 0

    def test_delete_later(self, mocker, empty_eye_app):
        delete_mock = mocker.patch.object(empty_eye_app.view, 'deleteLater')
        empty_eye_app.deleteLater()
        assert delete_mock.called

    def test_add_data(self, mocker, empty_eye_app, spectral_filenames, capsys):
        # no filename specified: calls filedialog
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[spectral_filenames])
        empty_eye_app.add_data()
        capt = capsys.readouterr()
        for fname in spectral_filenames:
            assert f'Adding data from {fname}' in capt.out
        new_files = empty_eye_app.view.model_collection
        assert new_files == spectral_filenames

        # explicitly add a file
        empty_eye_app.add_data([spectral_filenames[0]])
        assert f'Adding data from {spectral_filenames[0]}' \
            in capsys.readouterr().out
        # already there, so no change
        new_files = empty_eye_app.view.model_collection
        assert new_files == spectral_filenames

    def test_add_model(self, capsys, tmpdir, empty_eye_app,
                       spectral_filenames, grism_hdul, simple_fits_data):
        # try not specifying filename or hdul
        with pytest.raises(RuntimeError) as err:
            empty_eye_app._add_model()
        assert 'provide either' in str(err)

        # try adding both a filename and an hdul
        with pytest.raises(RuntimeError) as err:
            empty_eye_app._add_model(filename=spectral_filenames[0],
                                     hdul=grism_hdul)
        assert 'not both' in str(err)

        # add a good filename
        fname = empty_eye_app._add_model(filename=spectral_filenames[0])
        assert fname == spectral_filenames[0]
        assert 'Adding model from filename' in capsys.readouterr().out

        # add a nonexistent filename
        fname = empty_eye_app._add_model(filename='bad_file.fits')
        assert fname is None
        assert 'No such file' in capsys.readouterr().err

        # add an unsupported file
        bad_file = str(tmpdir.join('simple.fits'))
        simple_fits_data.writeto(bad_file)
        fname = empty_eye_app._add_model(filename=bad_file)
        assert fname is None
        assert 'Input data is not supported' in capsys.readouterr().err

        # add a good hdul
        fname = empty_eye_app._add_model(hdul=grism_hdul)
        assert fname == grism_hdul.filename()
        assert 'Adding model from hdul' in capsys.readouterr().out

        # add a good hdul, no filename associated
        separate_hdul = fits.HDUList(grism_hdul)
        separate_hdul.filename = None
        fname = empty_eye_app._add_model(hdul=separate_hdul)
        assert fname == grism_hdul[0].header.get('FILENAME')
        assert 'Adding model from hdul' in capsys.readouterr().out

        # add a good hdul, no filename at all
        del grism_hdul[0].header['FILENAME']
        fname = empty_eye_app._add_model(hdul=fits.HDUList(grism_hdul))
        assert fname == 'UNKNOWN'
        assert 'Adding model from hdul' in capsys.readouterr().out

        # add an unsupported hdul
        simple_fits_data[0].header['FILENAME'] = 'new.fits'
        fname = empty_eye_app._add_model(hdul=simple_fits_data)
        assert fname is None
        assert 'Input data is not supported' in capsys.readouterr().err

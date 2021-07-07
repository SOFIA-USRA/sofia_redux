#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.visualization.display.cursor_location import CursorLocation

PyQt5 = pytest.importorskip('PyQt5')


@pytest.fixture(scope='function')
def cursor_data():
    data_coords = {'file1.fits':
                   [{'order': 0, 'bin': 85,
                     'bin_x': 31.78, 'bin_y': 152.14,
                     'x_field': 'wavepos', 'y_field': 'spectral_flux',
                     'color': '#2848ad', 'visible': True},
                    {'order': 1, 'bin': 90,
                     'bin_x': 31.78, 'bin_y': 152.14,
                     'x_field': 'wavepos', 'y_field': 'spectral_flux',
                     'color': '#7b85d4', 'visible': True}],
                   'file2.fits':
                   [{'order': 0, 'bin': 85,
                     'bin_x': 31.78, 'bin_y': 162.10,
                     'x_field': 'wavepos', 'y_field': 'spectral_flux',
                     'color': '#59d4db', 'visible': True}],
                   'file3.fits':
                   [{'order': 0, 'bin': 85,
                     'bin_x': 31.78, 'bin_y': 159.08,
                     'x_field': 'wavepos', 'y_field': 'spectral_flux',
                     'color': '#f37738', 'visible': True}]}
    cursor_coords = [31.781183, 77.476260]
    return data_coords, cursor_coords


class TestCursorLocation(object):
    def test_init(self, empty_view, mocker):
        parent = empty_view
        mock_close = mocker.patch.object(parent, 'closed_cursor_popout',
                                         return_value=None)

        # default
        cl = CursorLocation(parent)
        assert isinstance(cl, PyQt5.QtWidgets.QDialog)

        # parent callback is called on close
        cl.close()
        assert mock_close.called

    def test_update_points(self, empty_view, cursor_data):
        cl = CursorLocation(empty_view)

        # 3 files, first one with two orders
        data_coords, cursor_coords = cursor_data

        # fill table with four rows, nine columns
        cl.update_points(data_coords, cursor_coords)
        assert cl.table_widget.rowCount() == 4
        assert cl.table_widget.columnCount() == 9

        # check cell contents
        columns = ['order', 'color', 'x_field', 'y_field',
                   'cursor_x', 'cursor_y', 'bin_x', 'bin_y', 'bin']
        check_order = 0
        for i in range(cl.table_widget.rowCount()):
            if check_order < 2:
                filename = f'file{i + 1 - check_order}.fits'
            else:
                filename = f'file{i}.fits'
            assert cl.table_widget.verticalHeaderItem(i).text() == filename
            for j in range(cl.table_widget.columnCount()):
                item_text = cl.table_widget.item(i, j).text()
                if columns[j] == 'color':
                    assert item_text == ''
                    assert isinstance(cl.table_widget.item(i, j).icon(),
                                      PyQt5.QtGui.QIcon)
                elif columns[j] == 'cursor_x':
                    value = cursor_coords[0]
                    assert item_text == f'{value:.3g}'
                elif columns[j] == 'cursor_y':
                    value = cursor_coords[1]
                    assert item_text == f'{value:.3g}'
                else:
                    if check_order < 2:
                        value = data_coords[filename][check_order][columns[j]]
                    else:
                        value = data_coords[filename][0][columns[j]]
                    if isinstance(value, str):
                        assert item_text == value
                    else:
                        assert item_text == f'{value:.3g}'
            check_order += 1

    def test_update_points_visibility(self, empty_view, cursor_data):
        cl = CursorLocation(empty_view)
        data_coords, cursor_coords = cursor_data

        # all visible: 4 rows, no blank cells
        cl.update_points(data_coords, cursor_coords)
        assert cl.table_widget.rowCount() == 4
        for i in range(cl.table_widget.rowCount()):
            for j in range(cl.table_widget.columnCount()):
                item_text = cl.table_widget.item(i, j).text()
                assert item_text != '-'

        # last one invisible: 4 rows, only the last has blank cells
        data_coords['file3.fits'][0]['visible'] = False
        cl.update_points(data_coords, cursor_coords)
        for i in range(cl.table_widget.rowCount()):
            for j in range(cl.table_widget.columnCount()):
                item_text = cl.table_widget.item(i, j).text()
                if i == 3:
                    assert item_text == '-'
                else:
                    assert item_text != '-'

        # all invisible: 3 rows (one per file, ignoring extra order),
        # all values '-'
        for fn in data_coords:
            for order in data_coords[fn]:
                order['visible'] = False
        cl.update_points(data_coords, cursor_coords)
        assert cl.table_widget.rowCount() == 3
        for i in range(cl.table_widget.rowCount()):
            for j in range(cl.table_widget.columnCount()):
                item_text = cl.table_widget.item(i, j).text()
                assert item_text == '-'

    def test_no_update(self, empty_view, cursor_data):
        cl = CursorLocation(empty_view)

        # no file loaded: 1 row, all cells empty
        cl.update_points({}, [10, 10])
        assert cl.table_widget.rowCount() == 1
        for i in range(cl.table_widget.rowCount()):
            for j in range(cl.table_widget.columnCount()):
                assert cl.table_widget.item(i, j) is None

        # loaded data, then no update: table stays the same
        cl.update_points(*cursor_data)
        assert cl.table_widget.rowCount() == 4
        expected = []
        for i in range(cl.table_widget.rowCount()):
            row = []
            for j in range(cl.table_widget.columnCount()):
                assert cl.table_widget.item(i, j) is not None
                row.append(cl.table_widget.item(i, j).text())
            expected.append(row)

        cl.update_points({}, [10, 10])
        assert cl.table_widget.rowCount() == 4
        for i in range(cl.table_widget.rowCount()):
            for j in range(cl.table_widget.columnCount()):
                assert cl.table_widget.item(i, j).text() == expected[i][j]

    def test_update_points_bad_data(self, empty_view, cursor_data):
        cl = CursorLocation(empty_view)
        data_coords, cursor_coords = cursor_data

        # set a bad data type in the coords
        data_coords['file1.fits'][0]['x_field'] = None
        cl.update_points(data_coords, cursor_coords)

        assert cl.table_widget.item(0, 2).text() == '######'

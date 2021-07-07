# Licensed under a 3-clause BSD style license - see LICENSE.rst

import argparse
import datetime
import os
import pathlib
import logging
from typing import List, Tuple, Union, Optional, Dict

import astropy.io.fits as pf

from sofia_redux.visualization import log
from sofia_redux.visualization import signals, setup
from sofia_redux.visualization.display import view
from sofia_redux.visualization.models import model
from sofia_redux.visualization.utils.logger import StreamLogger

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    HAS_PYQT5 = False
    QtCore, QtGui = None, None

    # duck type parents to allow class definition
    class QtWidgets:

        class QWidget:
            pass
else:
    HAS_PYQT5 = True

__all__ = ['Eye']


class Eye(object):
    """
    Run the Eye of SOFIA.

    This class provides the primary control interface to the display
    functions provided by the Eye viewer.  It is intended to be
    instantiated from the standalone interface
    (`sofia_redux.visualization.controller`) or the pipeline
    interface (`sofia_redux.visualization.redux_viewer`), but may
    also be directly instantiated and controlled, via the API interface.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command-line arguments to pass to the Eye interface.  Expected
        attributes are 'filenames', containing a list of input files
        to load, 'log_level' specifying a terminal log level (e.g. 'DEBUG'),
        or 'system_logs' = True, to specify that logs should be written
        to disk.
    view_ : `view.View`, optional
        A previously instantiated View object, to register with the
        Eye controller.  If not provided, a new View instance will be
        generated.
    """

    def __init__(self, args=None, view_=None):
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for the Eye.')

        # set terminal log level
        if hasattr(args, 'log_level'):
            self.log_level = str(args.log_level).upper()
        else:
            self.log_level = 'CRITICAL'
        self._setup_log_terminal()

        # set up log file if desired
        if hasattr(args, 'system_logs') and args.system_logs:
            self._setup_log_file()

        self.models = dict()
        self.model_index = 0
        if view_ is None:
            self.signals = signals.Signals()
            self.view = view.View(self.signals)
        else:
            self.signals = view_.signals
            self.view = view_
        self.setup_eye()

        if args:
            log.debug('Applying command line arguments')
            self._apply_args(args)

    @staticmethod
    def _setup_log_file() -> None:
        """Setup the system log file handler."""
        # set overall log level to debug
        log.setLevel('DEBUG')

        # make hidden home directory if necessary
        base_loc = os.path.expanduser(os.path.join('~', '.eye_of_sofia',
                                                   'event_logs'))
        try:
            os.makedirs(base_loc, exist_ok=True)
        except IOError:
            raise IOError(f'Unable to create log directory at {base_loc}')

        # make log file name in home directory from current time
        template = os.path.join(
            base_loc, 'eos_event_%Y-%m-%d_%H-%M-%S.log')
        fname = datetime.datetime.now().strftime(template)
        fhand = logging.FileHandler(fname, 'at')
        fhand.setLevel('DEBUG')
        fhand.setFormatter(logging.Formatter(
            "%(asctime)s - %(origin)s - %(levelname)s - %(message)s"))
        log.addHandler(fhand)
        log.info(f'Event log initiated at: {fname}')

    def _setup_log_terminal(self) -> None:
        """Set logging level for the stream logger."""
        # terminal is typically CRITICAL only but may
        # be overridden in arguments
        for hand in log.handlers:
            if isinstance(hand, StreamLogger):
                hand.setLevel(self.log_level)

    def open_eye(self) -> None:
        """Open the GUI."""
        log.debug('Opening Eye')
        self.view.open_eye()

    def close(self) -> None:
        """Close the GUI."""
        self.view.close()

    def reset(self) -> None:
        """
        Reset to an empty view.

        Data is unloaded and all viewer panes are deleted.
        """
        self.unload()
        self.view.reset()

    def deleteLater(self) -> None:
        """Delete the view when the control is deleted."""
        self.view.deleteLater()

    def setup_eye(self) -> None:
        """Setup the GUI and signal connections."""
        obj = setup.Setup(self)
        obj.setup_all()
        log.debug('Setup Eye')

    def _apply_args(self, args: argparse.Namespace) -> None:
        """Initialize the Eye with command line arguments."""
        if hasattr(args, 'filenames') and args.filenames:
            log.info('Reading in files from command line')
            self.add_data(filename=args.filenames)

    def add_data(self, filename: Optional[str] = None) -> None:
        """
        Add data.

        Parameters
        ----------
        filename : list of str, optional
            Absolute paths of FITS files to add to the Eye.
            If not provided, prompt the user for the filename.
        """
        if not filename:
            filename = QtWidgets.QFileDialog.getOpenFileNames(
                self.view, caption="Select Data File(s)",
                filter="FITS files (*.fits);;"
                       "All files (*)")[0]
        if filename:
            added = False
            for fname in filename:
                log.debug(f'Adding data from {fname}')
                added_filename = self._add_model(fname)
                if added_filename is not None:
                    added = True
                    self.view.display_filenames(added_filename)
            if added:
                self.signals.atrophy.emit()

    def _add_model(self, filename: str = '',
                   hdul: Optional[pf.hdu.hdulist.HDUList] = None) -> str:
        """
        Create a Model from provided data and add it to the Eye.

        The data can be provided by either a filename
        or the HDUL itself, but only one can be given.

        Parameters
        ----------
        filename : str, optional
            Name of the file to read.
        hdul : astropy.io.fits.HDUList, optional
            HDU list contents of a FITS file

        Returns
        -------
        filename : str
            Name of the file loaded, which also serves as the
            key to find the model in self.models.

        Raises
        ------
        RuntimeError :
            Raised if none or both `filename` and `hdul` are
            provided.
        """
        if filename and hdul is not None:
            raise RuntimeError('Eye._add_model can only accept `filename` '
                               'or `hdu`, not both')
        if filename:
            log.info('Adding model from filename')
            if filename not in self.models.keys():
                try:
                    m = model.Model.add_model(filename=filename)
                except FileNotFoundError:
                    log.warning(f'No such file: {filename}')
                    filename = None
                except (NotImplementedError, OSError,
                        RuntimeError, KeyError) as err:
                    log.debug(f'Error encountered: {str(err)}')
                    log.warning('Input data is not supported.')
                    filename = None
                else:
                    log.debug(f'Model index: {self.model_index}')
                    m.index = self.model_index
                    self.model_index += 1
                    self.models[filename] = m
        elif hdul is not None:
            log.info('Adding model from hdul')
            try:
                filename = hdul.filename()
            except TypeError:
                filename = hdul[0].header.get('FILENAME', None)
            if not filename:
                filename = hdul[0].header.get('FILENAME', 'UNKNOWN')
            log.debug(f'Found filename: {filename}')
            if filename not in self.models.keys():
                try:
                    m = model.Model.add_model(hdul=hdul)
                except (NotImplementedError, OSError,
                        RuntimeError, KeyError) as err:
                    log.debug(f'Error encountered: {str(err)}')
                    log.warning('Input data is not supported.')
                    filename = None
                else:
                    log.debug(f'Model index: {self.model_index}')
                    m.index = self.model_index
                    self.model_index += 1
                    self.models[filename] = m
        else:
            raise RuntimeError('Need to provide either a filename or HDUL')

        return filename

    # API
    def set_parent(self, parent: QtWidgets.QWidget) -> None:
        """
        Set the parent widget for the view.

        Parameters
        ----------
        parent : QtWidgets.QWidget
            The parent widget.
        """
        self.view.parent = parent

    def load(self, data_list: list) -> None:
        """
        Load a list of data files into the Eye.

        Parameters
        ----------
        data_list : list of str or astropy.io.fits.HDUList
            Data to load.
        """
        log.info(f'Loading file list ({len(data_list)} items)')
        if not isinstance(data_list, list):
            data_list = [data_list]
        for data in data_list:
            if isinstance(data, (str, pathlib.Path)):
                try:
                    hdul = pf.open(data, memmap=False)
                except IOError:
                    raise FileNotFoundError(f'File {data} not found')
                log.debug(f'Loading from filename {data}')
            elif isinstance(data, pf.hdu.hdulist.HDUList):
                hdul = data
                log.debug(f'Loading HDUList directly {data}')
            else:
                message = (f'Eye.load can only accept filenames or '
                           f'HDUList objects. Provided {type(data)}')
                log.error(message)
                raise TypeError(message)

            added_filename = self._add_model(hdul=hdul)
            hdul.close()
            if added_filename is not None:
                self.view.display_filenames(added_filename)

        self.signals.atrophy.emit()

    def unload(self) -> None:
        """Remove all loaded data."""
        filenames = list(self.models.keys())
        self.remove_data(filenames=filenames)

    def add_panes(self, layout='grid', n_panes=1, kind='spectrum') -> None:
        """
        Create blank panes in the figure.

        Parameters
        ----------
        layout : ['grid', 'rows', 'columns'], optional
            Layout strategy.
        n_panes : int, optional
            Number of panes to create.
        kind : ['spectrum', 'image'], optional
            Type of pane to create. Can also be a list of length
            `rows` * `cols` to create a mixture.

        Raises
        ------
        ValueError :
            If the layout to create cannot be inferred from the
            provided arguments.
        """
        if kind is None:
            raise ValueError('Must specify pane type with `kind` '
                             'keyword.')
        if isinstance(kind, list):
            if len(kind) != n_panes:
                raise ValueError('Length of `kind` must be either one '
                                 'or the number of panes being added.')
        elif isinstance(kind, str):
            if kind not in ['spectrum', 'onedim',
                            'image', 'twodim']:
                raise ValueError(f'Invalid kind: {kind}')
            else:
                kind = [kind] * n_panes
        self.view.add_panes(n_panes, kind=kind, layout=layout)

    def number_panes(self) -> int:
        """
        Retrieve the number of open panes.

        Returns
        -------
        int
            The pane count.
        """
        return self.view.pane_count()

    def get_pane_layout(self) -> Union[None, Tuple[int, int]]:
        """
        Retrieve the current pane layout.

        Returns
        -------
        geometry : tuple of int, or None
            If there is an active layout, (nrow, ncol) is returned.
            Otherwise, None.
        """
        return self.view.pane_layout()

    def assign_data(self, mode: str,
                    indices: Optional[List[int]] = None) -> None:
        """
        Assign models to panes.

        Parameters
        ----------
        mode : ['split', 'first', 'last', 'assigned']
            How to assign the data. `Split` will split
            the models equally between the panes, `first`
            will assign all the models to the first pane,
            `last` will assign all the models to the last
            pane, and `assign` will set the models
            according to the values in `indices`.
        indices : list of int, optional
            List of pane indices to which models should be
            assigned. Must match the length of the current list
            of loaded models. Must contain valid pane index values.

        Raises
        ------
        ValueError :
            If an invalid mode is provided, or indices do not match
            models or panes.
        """
        log.debug(f'Assigning data to panes using {mode}')
        possible_modes = ['split', 'first', 'last', 'assigned']
        if mode not in possible_modes:
            raise ValueError(f'Invalid data assignment mode {mode}. '
                             f'Valid modes: {possible_modes}')
        elif mode == 'assigned':
            if not isinstance(indices, list):
                raise ValueError(f'Invalid format of `indices` '
                                 f'{type(indices)}. Must be a list')
            if len(indices) != len(self.models):
                raise ValueError(f'Length of `indices` must match number of '
                                 f'models ({len(indices)} !='
                                 f' {len(self.models)})')
            if not all([0 <= i < self.number_panes() for i in indices]):
                raise ValueError('Values in `indices` must be valid values '
                                 'corresponding to panes')
        self.view.assign_models(mode, self.models, indices)

    def models_per_pane(self) -> List[int]:
        """
        Retrieve the number of models in all active panes.

        Returns
        -------
        model_count : list of int
            The model count for each pane.
        """
        return self.view.models_per_pane()

    def set_current_pane(self, pane_id: int) -> None:
        """
        Set the current pane in the view.

        Parameters
        ----------
        pane_id : int
            Pane index to make current.
        """
        self.view.set_current_pane(pane_id)

    def set_fields(self, x_field: Optional[str] = None,
                   y_field: Optional[str] = None,
                   z_field: Optional[str] = None,
                   fields: Optional[str] = None,
                   panes: Optional[Union[str, List[int]]] = 'all') -> None:
        """
        Set the axis fields to show for panes.

        Either `fields` or `x_field`, `y_field`, and `z_field`
        should be set.

        Parameters
        ----------
        x_field : str, optional
            The x field to set, if not provided in `fields`.
        y_field : str, optional
            The y field to set, if not provided in `fields`.
        z_field : str, optional
            The z field to set, if not provided in `fields`.
        fields : dict, optional
            Should contain keys 'x', 'y', and 'z', specifying
            the field strings as values.
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.
        """
        if fields is None:
            fields = {'x': x_field, 'y': y_field, 'z': z_field}
        for field in fields.values():
            if field is not None:
                if not isinstance(field, str):
                    raise TypeError('Fields must be strings')
        self.view.set_fields(fields=fields, panes=panes)

    def get_fields(self, panes: Optional[Union[str, List[int]]] = 'all'
                   ) -> List:
        """
        Get the currently displayed fields for current panes.

        Parameters
        ----------
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.

        Returns
        -------
        fields : list of list of dict
            List of fields for each pane specified.
        """
        return self.view.get_fields(panes)

    def set_units(self, units: Dict,
                  panes: Optional[Union[str, List[int]]] = 'all') -> None:
        """
        Set new units for specified panes.

        Parameters
        ----------
        units : dict
            New units to apply.  Should contain 'x', 'y' keys.
        panes :  str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.
        """
        if not isinstance(units, dict):
            raise TypeError('Provided units must be dict')
        self.view.set_units(units=units, panes=panes)

    def get_units(self, panes: Optional[Union[str, List[int]]] = 'all') -> \
            List:
        """
        Get the current units for active panes.

        Parameters
        ----------
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.

        Returns
        -------
        units : list of list of dict
            List of units for each pane specified.
        """
        return self.view.get_units(panes)

    def set_orders(self, orders: Dict) -> None:
        """
        Set orders to enable.

        Parameters
        ----------
        orders : dict
            Keys are model IDs, values are lists of orders to enable.
        """
        if not isinstance(orders, dict):
            raise TypeError('Provided orders must be dict')
        self.view.set_orders(orders)

    def get_orders(self, panes: Optional[Union[str, List[int]]] = 'all') -> \
            Dict:
        """
        Get enabled orders.

        Parameters
        ----------
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.

        Returns
        -------
        orders : dict
            Keys are model IDs, values are lists of enabled orders.
        """
        return self.view.get_orders(panes)

    def set_scale(self, scales: Dict,
                  panes: Optional[Union[str, List[int]]] = 'all') -> None:
        """
        Set scale setting for active panes.

        Parameters
        ----------
        scales : dict
            Keys are 'x' and 'y', values are 'linear' or 'log'.
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.
        """
        if not isinstance(scales, dict):
            raise TypeError('Provided scales must be dict')
        self.view.set_scales(scales=scales, panes=panes)

    def get_scale(self,
                  panes: Optional[Union[str, List[int]]] = 'all') -> List:
        """
        Get scale setting for active panes.

        Parameters
        ----------
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.

        Returns
        -------
        scales : list
            Keys are 'x' and 'y', values are 'linear' or 'log'.
        """
        return self.view.get_scales(panes=panes)

    def toggle_controls(self) -> None:
        """Toggle the control panel visibility."""
        self.view.toggle_controls()

    def toggle_cursor(self) -> None:
        """Toggle the cursor panel visibility."""
        self.view.toggle_cursor()

    def toggle_file_panel(self) -> None:
        """Toggle the file panel visibility."""
        self.view.toggle_file_panel()

    def toggle_pane_panel(self) -> None:
        """Toggle the pane panel visibility."""
        self.view.toggle_pane_panel()

    def toggle_order_panel(self) -> None:
        """Toggle the order panel visibility."""
        self.view.toggle_order_panel()

    def toggle_axis_panel(self) -> None:
        """Toggle the axis panel visibility."""
        self.view.toggle_axis_panel()

    def toggle_plot_panel(self) -> None:
        """Toggle the plot panel visibility."""
        self.view.toggle_plot_panel()

    def toggle_analysis_panel(self) -> None:
        """Toggle the plot panel visibility."""
        self.view.toggle_analysis_panel()

    def generate(self) -> None:
        """Initiate or refresh the view."""
        self.signals.atrophy_bg_full.emit()
        self.view.refresh_loop()

    def save(self, filename: str, **kwargs) -> None:
        """
        Save the current view to an image.

        Parameters
        ----------
        filename : str
            File path to save to.
        kwargs : dict
            Optional arguments to pass to `view.View.save`.
        """
        log.info(f'Saving image to {filename}')
        self.view.save(filename, **kwargs)

    def remove_data(self, filenames: Optional[List[str]] = None) -> None:
        """
        Remove loaded data from the view.

        Parameters
        ----------
        filenames : list of str, optional
            If not provided, currently selected files in the file panel
            will be removed.
        """
        if not filenames:
            filenames = self.view.current_files_selected()
        if not filenames:
            return
        if isinstance(filenames, str):
            filenames = [filenames]
        for filename in filenames:
            try:
                del self.models[filename]
            except KeyError:
                log.warning(f'File {filename} not found')
                continue
            self.view.remove_data_from_all_panes(filename)

        # reset model index if count has gone to zero
        if len(self.models) == 0:
            self.model_index = 0

        self.signals.refresh_file_table.emit()
        self.signals.atrophy.emit()

    def remove_panes(self, panes: Optional[Union[str, List[int]]] = 'all') -> \
            None:
        """
        Remove currently displayed panes.

        Parameters
        ----------
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.
        """
        self.view.remove_panes(panes)

    def display_selected_model(self) -> None:
        """Display a selected file in the current pane."""
        filenames = self.view.current_files_selected()
        if not filenames:
            return
        for filename in filenames:
            model_ = self._select_model_with_filename(filename)
            self.view.display_model(model_)
        self.signals.atrophy.emit()

    def _select_model_with_filename(self, filename: str) -> model.Model:
        """
        Select a model by associated filename.

        Parameters
        ----------
        filename : str
            The filename to search for.

        Returns
        -------
        model : `models.Model`
            The high-level model matching the input filename.

        Raises
        ------
        RuntimeError
            If no model matching the filename is found.
        """
        model_ = None
        for k, v in self.models.items():
            if k == filename:
                model_ = v
        if model_ is None:
            raise RuntimeError(f'Cannot locate model matching '
                               f'{filename} in model list '
                               f'{self.models}.')
        else:
            return model_

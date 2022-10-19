import copy
import os

import astropy.units as u
import numpy as np
from typing import Dict, List, Optional, Union
import pandas as pd
from sofia_redux.visualization import log
from sofia_redux.visualization.utils import unit_conversion as uc


class ReferenceData(object):
    """
    Backend for handling reference data.

    This is the parent class of the Reference_Window class. User does not
    interact with it directly. It parses data from the .txt/.csv files.
    These files may contain one or two columns. However,
    the first column must contain wavelength/wavenumber. It handles data
    with or without a header. It enables following features:
    - changing of axis-units is based on user selection.
    - reset labels and lines

    Parameters
    ----------
    line_list : Dict
        Values are wavelengths and labels
    line_unit : str or Unit
        Wavelength Unit of a line
    enabled : Dict
        keys are `ref_line` and `ref_label`

    """

    def __init__(self):
        self.line_list = dict()
        self.line_unit = None
        self.enabled = {'ref_line': False,
                        'ref_label': False}

    def __add__(self, other):
        if isinstance(other, type(self)):
            new = ReferenceData()
            self.line_list.update(other.line_list)
            new.line_list = self.line_list.copy()
            for key, value in self.enabled.items():
                new.enabled[key] = value or other.enabled[key]
            new.line_unit = other.line_unit

            return new

        else:
            raise ValueError('Invalid type for addition')

    def __repr__(self):
        lines = list()
        lines.append('Reference data:')
        lines.append(f'\t{len(self.line_list)} lines loaded')
        lines.append(f'\tVisibility: {self.enabled}')
        return '\n'.join(lines)

    def add_line_list(self, filename: str) -> bool:

        """
        Obtains a list of spectral lines.

        It sets the visibility for labels and lines while assuming the
        default unit of wavelength to be `um`.

        Parameters
        ----------
        filename : str
            Name of the file containing reference data.

        Returns
        -------
        Bool
            True: when data is parsed successfully
            False: When there is an error with reading the file or has
            errors (UnicodeError, ValueError or TypeError) .
        """

        log.info(f'Loading line list from {filename}')
        if not os.path.isfile(filename):
            log.error(f'Line list file {filename} not found')
            return False

        try:
            # attempt standard formats
            self._read_line_list(filename)
        except ValueError:
            try:
                # attempt space delimited
                self._read_line_list_space_delim(filename)
            except (ValueError, TypeError, OSError, UnicodeError) as err:
                log.debug(f'Error in reading line list: {err}')
                return False
        except (TypeError, OSError, UnicodeError) as err:
            log.debug(f'Error in reading line list: {err}')
            return False

        # check for empty list
        if len(self.line_list) == 0:
            log.debug('Line list is empty')
            return False

        self.line_unit = u.um
        self.set_visibility(['ref_line', 'ref_label'], True)
        return True

    def _read_line_list(self, filename):
        # allows single column or
        # 2+ column with comma, tab, or | as delimiter
        log.debug('Attempting to read most common line list formats')
        data = pd.read_table(filename, header=None, sep=r'\,|\t+|\|',
                             engine='python', comment='#',
                             skip_blank_lines=True)
        shape = data.shape[1]
        if shape >= 2:
            for i in range(len(data[0])):
                transition = data[1][i].strip()
                wavelength = float(data[0][i])
                if transition in self.line_list:
                    self.line_list[transition].append(wavelength)
                else:
                    self.line_list[transition] = [wavelength]
        elif shape == 1:
            for i in range(len(data[0])):
                wavelength = float(data[0][i])
                transition = f'{float(wavelength):.5g} um'
                self.line_list[transition] = [wavelength]
        else:  # pragma: no cover
            raise ValueError('Unexpected line list format')

    def _read_line_list_space_delim(self, filename):
        # allows space-separated two column files with
        # the first column a number
        log.debug('Attempting to read space delimited line list')
        data = pd.read_table(filename, header=None, comment='#',
                             skip_blank_lines=True,
                             sep=r'(?<=\d)\s+', engine='python',
                             names=['wave', 'label'], usecols=(0, 1))

        for i in range(len(data['wave'])):
            transition = str(data['label'][i]).strip()
            wavelength = float(data['wave'][i])
            if transition in self.line_list:
                self.line_list[transition].append(wavelength)
            else:
                self.line_list[transition] = [wavelength]

    def set_visibility(self, targets, state):
        """
        Changes the visibility of lines and labels based on user input.
        """
        if not isinstance(targets, list):
            targets = [targets]
        if 'all' in targets:
            targets = ['ref_line', 'ref_label']
        for target in targets:
            if target in self.enabled:
                self.enabled[target] = bool(state)

    def get_visibility(self, target: str) -> Optional[bool]:
        try:
            return self.enabled[target]
        except KeyError:
            return None

    def convert_line_list_unit(self, target_unit: Union[str, u.Unit],
                               names: Optional[List[str]] = None
                               ) -> Dict[str, float]:
        """
        Convert line list to new units.

        Parameters
        ----------
        target_unit : str or Unit
            Unit to convert to.
        names : list or dict
            If list, return all wavelengths matching the name.
            If dict, matching wavelength values only are returned. Wavelengths
            should be specified in before units.

        Returns
        -------
        converted_list : dict
            If names is provided, matching lines are returned. Otherwise,
            the full line list is returned.
        """
        try:
            conv_line_list = {k: uc.convert_wave(v, self.line_unit,
                                                 target_unit)
                              for k, v in self.line_list.items()}
        except ValueError:
            conv_line_list = copy.deepcopy(self.line_list)

        converted = conv_line_list
        if names:
            if isinstance(names, list):
                converted = {k: v for k, v in conv_line_list.items()
                             if k in names}
            elif isinstance(names, dict):
                converted = dict()
                for name, waves in self.line_list.items():
                    if name in names:
                        for i, wave in enumerate(waves):
                            match = any([np.isclose(w, wave)
                                         for w in names[name]])
                            if match:
                                cwave = conv_line_list[name][i]
                                if name in converted:
                                    converted[name].append(cwave)
                                else:
                                    converted[name] = [cwave]
        return converted

    def unload_data(self):
        """
        Resets the line_list, line_unit, and enabled.
        """
        self.line_list = dict()
        self.line_unit = None
        self.enabled = {'ref_line': False,
                        'ref_label': False}

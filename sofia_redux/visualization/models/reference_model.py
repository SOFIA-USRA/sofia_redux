import copy
import re
import os

import astropy.units as u
import numpy as np
from typing import Dict, List, Optional, Union

from sofia_redux.visualization import log
from sofia_redux.visualization.utils import unit_conversion as uc


class ReferenceData(object):

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
        log.info(f'Loading line list from {filename}')
        if not os.path.isfile(filename):
            log.error(f'Line list file {filename} not found')
            return False
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except UnicodeError:  # pragma: no cover
            return False
        if len(lines) == 0:
            log.info('Line list is empty')
            return True
        delim = self._determine_delim(lines[0])
        try:
            if delim is None:
                self._read_single_column(lines)
            else:
                self._read_multiple_columns(lines[1:], delim)
        except (ValueError, TypeError) as err:
            log.debug(f'Error in reading line list: {err}')
            return False
        self.line_unit = u.um
        self.set_visibility(['ref_line', 'ref_label'], True)
        return True

    @staticmethod
    def _determine_delim(line: str) -> Optional[str]:
        if line.startswith('#'):
            clean_line = line.strip().replace(' ', '').replace('#', '')
            delim = re.findall(r'\W', clean_line)
            if delim:
                delim = delim[0]
            else:
                delim = ' '
        else:
            delim = None
        log.debug(f'Line list deliminator set to {delim}')
        return delim

    def _read_single_column(self, lines):
        log.debug('Reading line list without labels')
        for i, line in enumerate(lines):
            label = f'{float(line):.5g} um'
            self.line_list[label] = [float(line.strip())]

    def _read_multiple_columns(self, lines, delim):
        log.debug('Reading line list with labels')
        for line in lines:
            if line.strip():
                parts = line.strip().split(delim)
                transition = ' '.join(parts[1:]).strip()
                wavelength = float(parts[0])
                if transition in self.line_list:
                    self.line_list[transition].append(wavelength)
                else:
                    self.line_list[transition] = [wavelength]

    def set_visibility(self, targets, state):
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
        self.line_list = dict()
        self.line_unit = None
        self.enabled = {'ref_line': False,
                        'ref_label': False}

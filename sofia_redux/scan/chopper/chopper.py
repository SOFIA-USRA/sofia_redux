# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
import numpy as np
from astropy import units, log

from sofia_redux.scan.chopper import chopper_numba_functions as cnf

__all__ = ['Chopper']


class Chopper(ABC):

    def __init__(self, x=None, y=None, time=None, threshold=None):
        """
        Initialize the chopper object.

        Parameters
        ----------
        x : units.Quantity (numpy.ndarray)
            The chopper x-positions.
        y : units.Quantity (numpy.ndarray)
            The chopper y-positions.
        time : units.Quantity (numpy.ndarray)
            The time of each chopper (x, y) measurement.
        threshold : units.Quantity
            The distance threshold over which to define a "chop".
        """
        self.positions = 0  # 0 for indeterminate, -1 for sweeping mode
        self._frequency = np.nan * units.Unit('Hz')
        self._amplitude = 0.0 * units.Unit('arcsec')
        self.efficiency = np.nan
        self._angle = np.nan * units.Unit('deg')
        self.offset = None
        self.phases = None
        self.is_chopping = False
        if None not in [x, y, time, threshold]:
            self.analyze_xy(x, y, time, threshold)

    @property
    def frequency(self):
        """
        Return the chop frequency.

        Returns
        -------
        units.Quantity
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """
        Set the chop frequency.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if isinstance(value, units.Quantity):
            self._frequency = value.to(units.Unit('Hz'))
        else:
            self._frequency = value * units.Unit('Hz')

    @property
    def amplitude(self):
        """
        Return the chop amplitude.

        Returns
        -------
        units.Quantity
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        """
        Set the chop amplitude.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if isinstance(value, units.Quantity):
            self._amplitude = value.to(units.Unit('arcsec'))
        else:
            self._amplitude = value * units.Unit('arcsec')

    @property
    def angle(self):
        """
        Return the chop angle.

        Returns
        -------
        units.Quantity
        """
        return self._angle

    @angle.setter
    def angle(self, value):
        """
        Set the chop angle.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if isinstance(value, units.Quantity):
            self._angle = value.to(units.Unit('deg'))
        else:
            self._angle = value * units.Unit('deg')

    @property
    def stare_duration(self):
        """
        Return the stare duration of the chopper.

        Returns
        -------
        time : units.Quantity
        """
        stare = self.efficiency / (self.positions * self.frequency)
        return stare.decompose().to('second')

    def __str__(self):
        """
        Return a string representation of the chopper.

        Returns
        -------
        str
        """
        return (f'+/- {self.amplitude} at {self.angle}, {self.frequency}, '
                f'{100 * self.efficiency:.1f}')

    def analyze_xy(self, x, y, time, threshold):
        """
        Analyse the chopper signal to determine internal parameters.

        Parameters
        ----------
        x : units.Quantity
            The x-position of the chopper.
        y : units.Quantity
            The y-position of the chopper.
        time : units.Quantity
            The time at each (x, y) measurement.
        threshold : units.Quantity
            The distance threshold over which to consider a chop transition.

        Returns
        -------
        None
        """
        start, end, transitions, angle, distance = cnf.find_transitions(
            x.to(threshold.unit).value,
            y.to(threshold.unit).value,
            threshold.value)

        distance *= threshold.unit
        angle *= units.rad

        if transitions > 2:
            dt = time[end] - time[start]
        else:
            log.debug("Chopper not used.")
            self.is_chopping = False
            return

        amplitude = np.nanmedian(distance)
        if amplitude < threshold:
            log.debug("Small chopper fluctuations "
                      "(assuming chopper not used).")
            self.is_chopping = False
            return

        self.amplitude = amplitude
        self.positions = 2
        self.frequency = (transitions - 1) / (2 * dt)
        self.angle = angle
        steady = np.sum(np.abs(distance - self.amplitude) < threshold)
        self.efficiency = steady / distance.size

        log.debug(f"Chopper detected: {self}")
        self.is_chopping = True

    def get_chop_table_entry(self, name):
        """
        Return a parameter value for a given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        if name == 'chopfreq':
            return self.frequency.to('Hz')
        elif name == 'chopthrow':
            return self.amplitude.to('arcsec')
        elif name == 'chopeff':
            return self.efficiency
        else:
            return None

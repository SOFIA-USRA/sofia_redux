# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod

from sofia_redux.scan.info.info import Info

__all__ = ['WeatherInfo']


class WeatherInfo(Info):

    @abstractmethod
    def get_ambient_kelvins(self):
        """
        Get the ambient temperature in Kelvins.

        Returns
        -------
        kelvins : units.Quantity
        """
        pass

    @abstractmethod
    def get_ambient_pressure(self):
        """
        Get the ambient pressure.

        Returns
        -------
        pressure : units.Quantity
        """
        pass

    @abstractmethod
    def get_ambient_humidity(self):
        """
        Get the ambient humidity.

        Returns
        -------
        humidity : units.Quantity
        """
        pass

    @abstractmethod
    def get_wind_direction(self):
        """
        Return the wind direction.

        Returns
        -------
        direction : units.Quantity
        """
        pass

    @abstractmethod
    def get_wind_speed(self):
        """
        Return the wind speed.

        Returns
        -------
        speed : units.Quantity
        """
        pass

    @abstractmethod
    def get_wind_peak(self):
        """
        Return the wind peak.

        Returns
        -------
        speed : units.Quantity
        """
        pass

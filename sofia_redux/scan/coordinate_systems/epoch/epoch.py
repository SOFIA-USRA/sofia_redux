# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import BaseCoordinateFrame
from copy import deepcopy
import numpy as np

__all__ = ['Epoch', 'JulianEpoch', 'BesselianEpoch', 'J2000',
           'B1900', 'B1950']


class Epoch(ABC):

    default_epoch = 'J2000'
    default_format = 'jyear'

    def __init__(self, equinox=None, immutable=False):
        """
        Initialize an astronomical epoch.

        An epoch defines a time or set of times when an observation occurred.
        This is important when converting between an apparent equinox (time at
        when a celestial object was observed) and a standard equinox such as
        J2000.0 (January 1, 2000 at 12:00 TT).  The equinox if defined as the
        two places on the celestial sphere at which the ecliptic intersects
        with the celestial equator.  The Sun's ascending node is used as this
        origin for celestial coordinate systems, but moves westward wrt the
        celestial sphere due to perturbing forces.  An epoch defines the date
        for which the position of a celestial object applies.  Therefore,
        astronomical coordinates require both the date of the equinox and the
        epoch.

        The current standard for the equinox and epoch is J2000.0 with "J"
        signifying the Julian epoch.  Before 1984, the standard was
        B1950.0 with "B" signifying the Besselian epoch.

        Parameters
        ----------
        equinox : parameter, optional
            The given equinox can be a string, int, float, Time, Header,
            FK4, FK5, or Epoch, or numpy.ndarray or list
        immutable : bool, optional
            If `True`, the equinox will be fixed and cannot be altered by
            standard methods.
        """
        self.equinox = self.get_equinox(equinox)
        self.immutable = immutable

    def copy(self):
        """
        Return a copy of the Epoch.

        Returns
        -------
        Epoch
        """
        return deepcopy(self)

    def empty_copy(self):
        """
        Return a copy of the epoch without times.

        Returns
        -------
        Epoch
        """
        new = self.__class__()
        new.immutable = self.immutable
        new.equinox = None
        return new

    def __eq__(self, other):
        """
        Return whether this epoch is equal to another.

        Parameters
        ----------
        other : Epoch

        Returns
        -------
        equal : bool
        """
        if not isinstance(other, Epoch):
            return False
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        if self.singular is not other.singular:
            return False
        if self.singular:
            return self.year == other.year
        elif self.shape != other.shape:
            return False
        else:
            return np.all(self.year == other.year)

    def __getitem__(self, indices):
        """
        Return an epoch for the given indices.

        Parameters
        ----------
        indices : int or numpy.ndarray (int) or slice

        Returns
        -------
        Epoch
        """
        return self.get_indices(indices)

    @property
    def singular(self):
        """
        Return `True` if the epoch represents a single time.

        Returns
        -------
        bool
        """
        return self.equinox.shape == ()

    @property
    def ndim(self):
        """
        Return the number of dimensions of the time.

        Returns
        -------
        int
        """
        return self.equinox.ndim

    @property
    def shape(self):
        """
        Return the shape of the time array.

        Returns
        -------
        tuple (int)
        """
        if self.equinox is None:
            return ()
        return self.equinox.shape

    @property
    def size(self):
        """
        Return the number of time measurements.

        Returns
        -------
        int
        """
        return self.equinox.size

    @property
    def year(self):
        """
        Return the equinox year.

        Returns
        -------
        float or numpy.ndarray (float)
        """
        return self.julian_year

    @year.setter
    def year(self, value):
        """
        Set the year values

        Parameters
        ----------
        value : float or numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.set_year(value)

    @property
    def julian_year(self):
        """
        Return the Julian year.

        Returns
        -------
        float or numpy.ndarray
        """
        return self.equinox.jyear

    @property
    def besselian_year(self):
        """
        Return the Julian year.

        Returns
        -------
        float or numpy.ndarray
        """
        return self.equinox.byear

    @property
    def mjd(self):
        """
        Return the equinox MJD.

        Returns
        -------
        float or numpy.ndarray (float)
        """
        return self.equinox.mjd

    @mjd.setter
    def mjd(self, value):
        """
        Set the equinox MJD.

        Parameters
        ----------
        value : float or numpy.ndarray

        Returns
        -------
        None
        """
        self.set_mjd(value)

    @property
    def is_julian(self):
        """
        Return whether the epoch if Julian (True) or Besselian (False).

        Returns
        -------
        bool
        """
        return self.equinox.format[0].lower() == 'j'

    def __str__(self):
        """
        Return a string representation of the epoch.

        Returns
        -------
        str
        """
        if self.singular:
            return f'{self.year}'
        else:
            return (f'MJD {round(self.mjd.min(), 8)} -> '
                    f'{round(self.mjd.max(), 8)}')

    @classmethod
    def get_equinox(cls, equinox=None):
        """
        Return an astropy equinox frame from the given input.

        Parameters
        ----------
        equinox : parameter
            The given equinox can be a string, int, float, Time, Header,
            FK4, FK5, or Epoch, or numpy.ndarray or list

        Returns
        -------
        equinox : astropy.units.Time
        """
        if equinox is None:
            return cls.get_equinox(cls.default_epoch)
        elif isinstance(equinox, (Epoch, BaseCoordinateFrame)):
            return equinox.equinox
        elif isinstance(equinox, Time):
            return equinox
        elif isinstance(equinox, fits.Header):
            return cls.get_equinox_from_header(equinox)

        if isinstance(equinox, str):
            equinox = str(equinox.strip().upper())
            if equinox[0].isalpha():
                return Time(equinox)
            else:
                equinox = float(equinox)  # Allow error

        if isinstance(equinox, (int, float)):
            if equinox < 1984:  # Assume year
                return Time(f'B{equinox}')
            else:
                return Time(f'J{equinox}')

        if isinstance(equinox, (list, np.ndarray)):
            return Time(equinox, format=cls.default_format)

        try:  # pragma: no cover
            return Time(equinox)  # See if it works by chance
        except (ValueError, TypeError) as err:  # pragma: no cover
            log.warning(f"Could not parse {equinox} as equinox.")
            raise err

    @classmethod
    def get_epoch(cls, epoch):
        """
        Return an epoch for a given input.

        Parameters
        ----------
        epoch : thing
            The str, int, float, Time, Epoch for which to get the epoch.

        Returns
        -------
        JulianEpoch or BesselianEpoch
        """
        if not isinstance(epoch, Epoch):
            epoch = cls(equinox=epoch)

        if epoch.is_julian:
            return epoch.get_julian_epoch()
        else:
            return epoch.get_besselian_epoch()

    @classmethod
    def get_equinox_from_header(cls, header, alt=''):
        """
        Return an equinox from a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to read.
        alt : str, optional
            The optional alternate system.

        Returns
        -------
        equinox : astropy.units.Time
        """
        return cls.get_equinox(header.get(f'EQUINOX{alt}'))

    def edit_header(self, header, alt=''):
        """
        Edit a FITS header with the epoch data.

        Parameters
        ----------
        header : fits.header.Header
        alt : str, optional
            The optional alternate system.

        Returns
        -------
        None
        """
        if not self.singular:
            return  # Can only do this for a single value
        header[f'EQUINOX{alt}'] = (
            self.year, 'The epoch of the quoted coordinates')

    def set_year(self, year):
        """
        Set the year values.

        Parameters
        ----------
        year : int or float or numpy.ndarray or Time

        Returns
        -------
        None
        """
        if self.immutable:
            raise ValueError("Cannot alter immutable epochs.")

        if isinstance(year, Time):
            self.equinox[...] = year
        else:
            self.equinox[...] = Time(year, format=self.default_format)

    def set_mjd(self, mjd):
        """
        Set the MJD for the epoch.

        Parameters
        ----------
        mjd : int or float or Time or numpy.ndarray
           Note that if a int or float value is provided, those MJD values
           will be assumed to be in UTC scale.  If necessary, conversion to
           TT (Terrestrial Time) will occur if the current equinox is also in
           TT (default).

        Returns
        -------
        None
        """
        if self.immutable:
            raise ValueError("Cannot alter immutable epochs.")

        if isinstance(mjd, Time):
            self.equinox[...] = mjd
        else:
            self.equinox[...] = Time(mjd, format='mjd')

    def get_julian_epoch(self):
        """
        Get a Julian representation of the epoch.

        Returns
        -------
        JulianEpoch
        """
        return JulianEpoch(equinox=self.julian_year, immutable=self.immutable)

    def get_besselian_epoch(self):
        """
        Get a Besselian representation of the epoch.

        Returns
        -------
        BesselianEpoch
        """
        return BesselianEpoch(equinox=self.besselian_year,
                              immutable=self.immutable)

    def get_indices(self, indices):
        """
        Return selected times for given indices.

        Parameters
        ----------
        indices : slice or list or int or numpy.ndarray (int) or None
            The indices to extract.  If `None`, an empty copy of the epoch will
            be returned

        Returns
        -------
        Epoch
        """
        if self.singular:
            raise KeyError("Cannot retrieve indices for singular epochs.")

        new = self.empty_copy()
        if indices is None:
            return new

        if isinstance(indices, np.ndarray) and indices.shape == ():
            indices = int(indices)

        new.equinox = self.equinox[indices]
        return new


class JulianEpoch(Epoch):

    default_epoch = 'J2000'
    default_format = 'jyear'

    def __init__(self, equinox=None, immutable=False):
        """
        Initialize a Julian epoch.

        The JulianEpoch is an extension of the Epoch class which will always
        set and return Julian years when necessary.  The epoch and equinox will
        default to J2000 unless explicitly defined.

        Parameters
        ----------
        equinox : parameter, optional
            The given equinox can be a string, int, float, Time, Header,
            FK4, FK5, or Epoch, or numpy.ndarray or list
        immutable : bool, optional
            If `True`, the equinox will be fixed and cannot be altered by
            standard methods.
        """
        super().__init__(equinox=equinox, immutable=immutable)

    def copy(self):
        """
        Return a copy of the JulianEpoch.

        Returns
        -------
        JulianEpoch
        """
        return super().copy()

    @property
    def year(self):
        """
        Return the Julian year.

        Returns
        -------
        float or numpy.ndarray (float)
        """
        return self.julian_year

    @year.setter
    def year(self, value):
        """
        Set the year values

        Parameters
        ----------
        value : float or numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.set_year(value)

    def __str__(self):
        """
        Return a string representation of the Julian epoch.

        Returns
        -------
        str
        """
        if self.singular:
            return f'J{self.year}'
        else:
            return (f'Julian MJD {round(self.mjd.min(), 8)} -> '
                    f'{round(self.mjd.max(), 8)}')

    @classmethod
    def get_equinox(cls, equinox=None):
        """
        Return an astropy equinox frame from the given input.

        Parameters
        ----------
        equinox : parameter
            The given equinox can be a string, int, float, Time, Header,
            FK4, FK5, or Epoch, or numpy.ndarray or list

        Returns
        -------
        equinox : astropy.units.Time
        """
        if isinstance(equinox, (int, float)):
            return Time(f'J{equinox}')
        return super().get_equinox(equinox=equinox)

    def get_indices(self, indices):
        """
        Return selected epoch for given indices.

        Parameters
        ----------
        indices : slice or list or int or numpy.ndarray (int)
            The indices to extract.

        Returns
        -------
        JulianEpoch
        """
        return super().get_indices(indices)


class BesselianEpoch(Epoch):
    default_epoch = 'B1950'
    default_format = 'byear'

    def __init__(self, equinox=None, immutable=False):
        """
        Initialize a Besselian epoch.

        The BesselianEpoch is an extension of the Epoch class which will always
        set and return Besselian years when necessary.  The epoch and equinox
        will default to B1950 unless explicitly defined.

        Parameters
        ----------
        equinox : parameter, optional
            The given equinox can be a string, int, float, Time, Header,
            FK4, FK5, or Epoch, or numpy.ndarray or list
        immutable : bool, optional
            If `True`, the equinox will be fixed and cannot be altered by
            standard methods.
        """
        super().__init__(equinox=equinox, immutable=immutable)

    def copy(self):
        """
        Return a copy of the BesselianEpoch.

        Returns
        -------
        BesselianEpoch
        """
        return super().copy()

    @property
    def year(self):
        """
        Return the Besselian year.

        Returns
        -------
        float or numpy.ndarray (float)
        """
        return self.besselian_year

    @year.setter
    def year(self, value):
        """
        Set the year values

        Parameters
        ----------
        value : float or numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.set_year(value)

    def __str__(self):
        """
        Return a string representation of the Besselian epoch.

        Returns
        -------
        str
        """
        if self.singular:
            return f'B{self.year}'
        else:
            return (f'Besselian MJD {round(self.mjd.min(), 8)} -> '
                    f'{round(self.mjd.max(), 8)}')

    @classmethod
    def get_equinox(cls, equinox=None):
        """
        Return an astropy equinox frame from the given input.

        Parameters
        ----------
        equinox : parameter
            The given equinox can be a string, int, float, Time, Header,
            FK4, FK5, or Epoch, or numpy.ndarray or list

        Returns
        -------
        equinox : astropy.units.Time
        """
        if isinstance(equinox, (int, float)):
            return Time(f'B{equinox}')
        return super().get_equinox(equinox=equinox)

    def get_indices(self, indices):
        """
        Return selected epoch for given indices.

        Parameters
        ----------
        indices : slice or list or int or numpy.ndarray (int)
            The indices to extract.

        Returns
        -------
        BesselianEpoch
        """
        return super().get_indices(indices)


J2000 = JulianEpoch(equinox='J2000', immutable=True)
B1900 = BesselianEpoch(equinox='B1900', immutable=True)
B1950 = BesselianEpoch(equinox='B1950', immutable=True)

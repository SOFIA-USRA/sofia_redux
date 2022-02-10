# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.epoch import \
    precession_numba_functions as pnf

__all__ = ['Precession']


class Precession(ABC):

    JULIAN_CENTURY = 36525.0 * units.Unit('day')
    YEAR = 365.24219879 * units.Unit('day')
    YEAR_TO_CENTURY = (YEAR / JULIAN_CENTURY).decompose().value

    def __init__(self, from_epoch, to_epoch):
        """
        Initialize a precession object.

        The precession object is used to precess equatorial coordinates from
        one epoch to another using the procedure found in:

        Lederle, T. and Schwan, H., "Procedure for computing the apparent
            places of fundamental stars (APFS) from 1984 onwards",
            Astronomy and Astrophysics, vol. 134, no. 1, pp. 1–6, 1984.

        Parameters
        ----------
        from_epoch : Epoch
            The epoch to precess from.
        to_epoch : Epoch
            The epoch to precess to.
        """
        self.from_epoch = from_epoch
        self.to_epoch = to_epoch
        self.p = None
        if self.from_epoch != self.to_epoch:
            self.calculate_matrix()

    def copy(self):
        """
        Return a copy of the precession.

        Returns
        -------
        Precession
        """
        new = Precession(self.from_epoch.copy(), self.from_epoch)
        new.to_epoch = self.to_epoch.copy()
        if self.p is not None:
            new.p = self.p.copy()
        return new

    def __eq__(self, other):
        """
        Check whether this precession is equal to another.

        Parameters
        ----------
        other : Precession

        Returns
        -------
        bool
        """
        if self is other:
            return True
        if other.__class__ != self.__class__:
            return False
        if self.from_epoch != other.from_epoch:
            return False
        if self.to_epoch != other.to_epoch:
            return False
        return True

    @property
    def singular_epoch(self):
        """
        Return whether the precession consists of single time epochs.

        Returns
        -------
        is_singular : bool
        """
        return self.from_epoch.singular and self.to_epoch.singular

    @staticmethod
    def r2(phi):
        """
        Calculate the R2 matrix.

        The R2 matrix is a 3x3 array of the form:

          [[cos(phi), 0, sin(phi)],
           [0,        1, 0       ],
           [sin(phi), 0, cos(phi)]]

        Parameters
        ----------
        phi : astropy.units.Quantity
            The phi angle.  Either a single value or an array of shape (shape,)
            may be provided.

        Returns
        -------
        R2 : numpy.ndarray (float)
           The R2 matrix.  If a single `phi`` was provided, the array will
           be of shape (3, 3).  Otherwise, it will be of shape
           (phi.shape, 3, 3).
        """
        if isinstance(phi, np.ndarray) and phi.shape != ():
            singular = False
        else:
            singular = True

        c = np.cos(phi).value
        s = np.sin(phi).value

        if singular:
            return np.array(
                [[c, 0, -s],
                 [0, 1, 0],
                 [s, 0, c]], dtype=float)

        result = np.zeros(c.shape + (3, 3), dtype=float)
        result[..., 0, 0] = c
        result[..., 0, 2] = -s
        result[..., 1, 1] = 1
        result[..., 2, 0] = s
        result[..., 2, 2] = c
        return result

    @staticmethod
    def r3(phi):
        """
        Calculate the R3 matrix.

        The R3 matrix is a 3x3 array of the form:

          [[cos(phi),  sin(phi), 0],
           [-sin(phi), cos(phi), 0],
           [0,         0,        1]]

        Parameters
        ----------
        phi : astropy.units.Quantity
            The phi angle.

        Returns
        -------
        R2 : numpy.ndarray (float)
           The R3 matrix.  If a single `phi`` was provided, the array
           will be of shape (3, 3).  Otherwise, it will be of shape
           (phi.shape, 3, 3).
        """
        if isinstance(phi, np.ndarray) and phi.shape != ():
            singular = False
        else:
            singular = True

        c = np.cos(phi).value
        s = np.sin(phi).value

        if singular:
            return np.array(
                [[c, s, 0],
                 [-s, c, 0],
                 [0, 0, 1]], dtype=float)

        result = np.zeros(c.shape + (3, 3), dtype=float)
        result[..., 0, 0] = c
        result[..., 0, 1] = s
        result[..., 1, 0] = -s
        result[..., 1, 1] = c
        result[..., 2, 2] = 1
        return result

    def calculate_matrix(self):
        """
        Precess the coordinates from the from_epoch to the to_epoch.

        The precession matrix is a 3x3 matrix given by:

            m = x1.x2.x3

        where "." indicates dot matrix multiplication and

            x1 = |  cos(z) sin(z) 0 |
                 | -sin(z) cos(z) 0 |
                 |    0      0    1 |

            x2 = | cos(theta) 0 sin(theta) |
                 |     0      1     0      |
                 | sin(theta) 0 cos(theta) |

            x3 = |  cos(eta) sin(eta) 0 |
                 | -sin(eta) cos(eta) 0 |
                 |    0        0      1 |

            z = ((2305.6997 + (1.39744 + 0.000060 * tau) * tau +
                  (1.09543 + 0.000390 * tau + 0.018326 * t) * t) * t)

            theta = ((2003.8746 - (0.85405 + 0.000370 * tau) * tau -
                     (0.42707 + 0.000370 * tau + 0.041803 * t) * t) * t)

            eta = ((2305.6997 + (1.39744 + 0.000060 * tau) * tau +
                    (0.30201 - 0.000270 * tau + 0.017996 * t) * t) * t)

            tau = (epoch1_year - 2000) * year_to_century

            t = (epoch2_year - epoch1_year) * year_to_century

            year_to_century = 365.24219879 / 36525

        Note that z, theta and eta are in arcseconds.

        Returns
        -------
        None
        """
        from_year = self.from_epoch.julian_year
        to_year = self.to_epoch.julian_year
        arcsec = units.Unit('arcsec')

        tau = (from_year - 2000) * self.YEAR_TO_CENTURY
        t = (to_year - from_year) * self.YEAR_TO_CENTURY
        eta = ((2305.6997 + (1.39744 + 0.000060 * tau) * tau
                + (0.30201 - 0.000270 * tau + 0.017996 * t) * t) * t)
        z = ((2305.6997 + (1.39744 + 0.000060 * tau) * tau
              + (1.09543 + 0.000390 * tau + 0.018326 * t) * t) * t)
        theta = ((2003.8746 - (0.85405 + 0.000370 * tau) * tau
                 - (0.42707 + 0.000370 * tau + 0.041803 * t) * t) * t)

        x1 = self.r3(-z * arcsec)
        x2 = self.r2(theta * arcsec)
        x3 = self.r3(-eta * arcsec)
        if self.singular_epoch:
            self.p = x1.dot(x2).dot(x3)
        else:
            self.p = np.einsum('aij,ajk->aik', x1, x2)
            self.p = np.einsum('aij,ajk->aik', self.p, x3)

    def precess(self, equatorial):
        """
        Precess the coordinates in-place.

        Parameters
        ----------
        equatorial : EquatorialCoordinates
            The equatorial coordinates to precess.

        Returns
        -------
        None
        """
        if self.p is None:
            return

        ra = np.atleast_1d(equatorial.ra.to('radian').value)
        dec = np.atleast_1d(equatorial.dec.to('radian').value)

        if self.singular_epoch:
            pnf.precess_single(
                p=self.p, ra=ra, dec=dec,
                cos_lat=np.atleast_1d(equatorial.cos_lat),
                sin_lat=np.atleast_1d(equatorial.sin_lat))
        else:
            pnf.precess_times(
                p=self.p, ra=ra, dec=dec,
                cos_lat=np.atleast_1d(equatorial.cos_lat),
                sin_lat=np.atleast_1d(equatorial.sin_lat))

        if equatorial.singular:
            ra = ra[0]
            dec = dec[0]

        equatorial.set_ra(ra * units.Unit('radian'), copy=False)
        equatorial.set_dec(dec * units.Unit('radian'), copy=False)
        equatorial.epoch = self.to_epoch

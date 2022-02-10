# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy.time import Time
import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.epoch.epoch import (
    B1950, J2000, Epoch, JulianEpoch, BesselianEpoch)
from sofia_redux.scan.coordinate_systems.celestial_coordinates import \
    CelestialCoordinates

__all__ = ['PrecessingCoordinates']


class PrecessingCoordinates(CelestialCoordinates):

    def __init__(self, coordinates=None, unit='degree',
                 copy=True, epoch=J2000):
        """
        Initialize a PrecessingCoordinates object.

        The precessing coordinates extend celestial coordinates by adding an
        epoch (time) to the coordinate set.  In addition to the functionality
        provided by celestial coordinates, the coordinate set may be precessed
        to a new epoch if desired.

        Parameters
        ----------
        coordinates : list or tuple or array-like or units.Quantity, optional
            The coordinates used to populate the object during initialization.
            The first (0) value or index should represent longitudinal
            coordinates, and the second should represent latitude.
        unit : units.Unit or str, optional
            The angular unit for the spherical coordinates.  The default is
            'degree'.
        copy : bool, optional
            Whether to explicitly perform a copy operation on the input
            coordinates when storing them into these coordinates.  Note that it
            is extremely unlikely for the original coordinates to be passed in
            as a reference due to the significant checks performed on them.
        epoch : Epoch or str or int or float or Time or fits.Header
            Information from which to set the epoch for these coordinates.
        """

        self.epoch = None
        self.set_epoch(epoch)
        super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def copy(self):
        """
        Return a copy of the PrecessingCoordinates.

        Returns
        -------
        PrecessingCoordinates
        """
        return super().copy()

    def empty_copy(self):
        """
        Return an unpopulated instance of the coordinates.

        Returns
        -------
        PrecessingCoordinates
        """
        new = super().empty_copy()
        if self.epoch.singular:
            new.epoch = self.epoch
        elif isinstance(self.epoch, BesselianEpoch):
            new.epoch = B1950
        else:
            new.epoch = J2000
        return new

    @abstractmethod
    def precess_to_epoch(self, new_epoch):  # pragma: no cover
        """
        Precess from one epoch to another.

        Parameters
        ----------
        new_epoch : Epoch

        Returns
        -------
        None
        """
        pass

    def __eq__(self, other):
        """
        Test if these precessing coordinates are equal to another.

        Parameters
        ----------
        other : PrecessingCoordinates

        Returns
        -------
        bool
        """
        if not super().__eq__(other):
            return False
        return self.epoch == other.epoch

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        PrecessingCoordinates
        """
        return super().__getitem__(indices)

    def __str__(self):
        """
        Create a string representation of the equatorial coordinates.

        Returns
        -------
        str
        """
        if self.coordinates is None:
            return f'Empty coordinates ({self.epoch})'

        if self.singular:
            return f'LON={self.longitude} LAT={self.latitude} ({self.epoch})'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return (f'LON={np.nanmin(self.longitude)}->'
                        f'{np.nanmax(self.longitude)} '
                        f'DEC={np.nanmin(self.latitude)}->'
                        f'{np.nanmax(self.latitude)} ({self.epoch})')

    @property
    def empty_copy_skip_attributes(self):
        """
        Return attributes that are set to None on an empty copy.

        Returns
        -------
        attributes : set (str)
        """
        skip = super().empty_copy_skip_attributes
        skip.add('epoch')
        return skip

    def copy_coordinates(self, coordinates):
        """
        Copy the contents of another coordinate system.

        Parameters
        ----------
        coordinates : SphericalCoordinates

        Returns
        -------
        None
        """
        super().copy_coordinates(coordinates)
        if isinstance(coordinates, PrecessingCoordinates):
            if coordinates.epoch is not None:
                self.epoch = coordinates.epoch.copy()
            else:
                self.epoch = None
        else:
            self.epoch = None

    def set_epoch(self, epoch):
        """
        Set the epoch for the precessing coordinates.

        Parameters
        ----------
        epoch : Epoch or str or int or float or Time or fits.Header

        Returns
        -------
        None
        """
        self.epoch = Epoch.get_epoch(epoch)

    def precess(self, new_epoch):
        """
        Precess the coordinates to a new epoch.

        Parameters
        ----------
        new_epoch : Epoch

        Returns
        -------
        None
        """
        if self.epoch == new_epoch:
            return
        if self.epoch is None:
            raise ValueError("Undefined from epoch.")
        elif new_epoch is None:
            raise ValueError("Undefined to epoch.")
        self.precess_to_epoch(new_epoch)

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with precessing coordinate information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to modify.
        key_stem : str
            The name of the header key to update.
        alt : str, optional
            The alternative coordinate system.

        Returns
        -------
        None
        """
        super().edit_header(header, key_stem, alt=alt)
        header[f'RADESYS{alt}'] = ('FK5' if self.epoch.is_julian else 'FK4',
                                   'Reference convention.')
        self.epoch.edit_header(header, alt=alt)

    def parse_header(self, header, key_stem, alt='', default=None):
        """
        Set the coordinate from the header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to read.
        key_stem : str
        alt : str, optional
            The alternate coordinate system.
        default : astropy.units.Quantity (numpy.ndarray)
            The (x, y) default coordinate.

        Returns
        -------
        None
        """
        super().parse_header(header, key_stem, alt=alt, default=default)
        if 'RADESYS' in header:
            system = header['RADESYS']
        elif f'EQUINOX{alt}' in header:
            equinox = header[f'EQUINOX{alt}']
            if isinstance(equinox, str):
                if equinox[0].isalpha():
                    system = 'FK4' if equinox[0].lower() == 'b' else 'FK5'
                else:
                    system = 'FK4' if float(equinox) < 1984 else 'FK5'
            else:
                system = 'FK4' if header[f'EQUINOX{alt}'] < 1984 else 'FK5'
        else:
            system = 'FK5'
        system = system.strip().upper()

        if system in ['FK4', 'FK4-NO-E']:
            self.epoch = BesselianEpoch(header)
        else:
            self.epoch = JulianEpoch(header)

    def convert(self, from_coordinates, to_coordinates):
        """
        Convert one type of coordinates to another.

        The `to_coordinates` will be updated in-place.

        Parameters
        ----------
        from_coordinates : CelestialCoordinates or PrecessingCoordinates
        to_coordinates : CelestialCoordinates or PrecessingCoordinates

        Returns
        -------
        None
        """
        if isinstance(to_coordinates, PrecessingCoordinates):
            to_epoch = to_coordinates.epoch
        else:
            to_epoch = None
        if isinstance(from_coordinates, PrecessingCoordinates):
            from_epoch = from_coordinates.epoch
        else:
            from_epoch = None

        super().convert(from_coordinates, to_coordinates)
        if to_epoch is not None and from_epoch is not None:
            to_coordinates.epoch = from_epoch
            to_coordinates.precess(to_epoch)
            to_coordinates.epoch = to_epoch

        # TODO: This was the original, but did not work there either
        # super().convert(from_coordinates, to_coordinates)
        # if from_coordinates.__class__ == to_coordinates.__class__:
        #     if isinstance(to_coordinates, PrecessingCoordinates):
        #         to_coordinates.precess(to_coordinates.epoch)

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : slice or list or int or numpy.ndarray (int)
            The indices to extract.

        Returns
        -------
        FlaggedData
        """
        new = super().get_indices(indices)
        if new.coordinates is None:
            return new

        if not self.epoch.singular:
            new.epoch = self.epoch[indices]

        return new

    def insert_blanks(self, insert_indices):
        """
        Insert blank (NaN) values at the requested indices.

        Follows the logic of :func:`numpy.insert`.

        Parameters
        ----------
        insert_indices : numpy.ndarray (int)

        Returns
        -------
        None
        """
        super().insert_blanks(insert_indices)
        if self.epoch.singular:
            return

        old_time = self.epoch.equinox
        times = old_time.jyear

        new_time_values = np.insert(times, insert_indices, 0.0)
        new_time = old_time.__class__(new_time_values, format=old_time.format,
                                      scale=old_time.scale)
        self.epoch.equinox = new_time

    @staticmethod
    def precession_required(epoch1, epoch2):
        """
        Determine if precession is required when converting between epochs.

        Parameters
        ----------
        epoch1 : Epoch or None
            The epoch to convert from.
        epoch2 : Epoch or None
            The epoch to convert to.

        Returns
        -------
        precess, epoch1, epoch2 : bool, Epoch, Epoch
            Whether precession is required, and the epochs to convert between.
        """
        if epoch1 is None and epoch2 is None:
            return False, epoch1, epoch2

        if epoch1 is None:
            if epoch2.singular:
                return False, epoch2.copy(), epoch2
            return True, epoch2.get_epoch(epoch2.default_epoch), epoch2

        if epoch2 is None:
            if epoch1.singular:
                return False, epoch1, epoch1.copy()
            return True, epoch1, epoch1.get_epoch(epoch1.default_epoch)

        if epoch1.singular and epoch2.singular:
            return epoch1 != epoch2, epoch1, epoch2

        return True, epoch1, epoch2

    def merge(self, other):
        """
        Append other coordinates to the end of these.

        The other coordinates will be precessed to this epoch if possible.  If
        this epoch contains more than one equinox, no precession is performed
        and the final equinox will contain those of the other equinox, expanded
        and appended as necessary.

        Parameters
        ----------
        other : PrecessingCoordinates

        Returns
        -------
        None
        """
        e1 = self.epoch
        e2 = other.epoch if isinstance(other, PrecessingCoordinates) else None
        precess, e1, e2 = self.precession_required(e1, e2)

        self.epoch = e1  # Update the epoch if required
        if not precess:
            super().merge(other)
            return

        if e1.singular:  # Precess epoch 2 onto epoch 1
            precessed = self.copy()
            self.convert(other, precessed)
            super().merge(precessed)
            return

        # If epoch 1 is an array and epoch 2 is not
        if e2.singular:
            # Need to convert all epoch2 times to an array.
            e2 = e2.copy()
            e2.equinox = Time(np.full(other.size, e2.equinox.value),
                              scale=e2.equinox.scale,
                              format=e2.equinox.format)

        size_1 = self.size
        size_2 = other.size
        super().merge(other)

        new_time_values = np.empty(size_1 + size_2, dtype=float)

        t_format = 'byear' if isinstance(e1, BesselianEpoch) else 'jyear'
        new_time_values[:size_1] = getattr(e1.equinox, t_format)
        new_time_values[size_1:] = getattr(e2.equinox, t_format)
        self.epoch.equinox = Time(new_time_values, scale=e1.equinox.scale,
                                  format=t_format)

    def paste(self, other, indices):
        """
        Paste new coordinate values at the given indices.

        Parameters
        ----------
        other : PrecessingCoordinates
        indices : numpy.ndarray (int)

        Returns
        -------
        None
        """
        e1 = self.epoch
        e2 = other.epoch if isinstance(other, PrecessingCoordinates) else None
        precess, e1, e2 = self.precession_required(e1, e2)
        self.epoch = e1  # Update the epoch if required

        if not precess:
            super().paste(other, indices)
            return

        if e1.singular:  # Precess epoch 2 onto epoch 1
            precessed = self.copy()
            self.convert(other, precessed)
            super().paste(precessed, indices)
            return

        super().paste(other, indices)
        e1.equinox[indices] = e2.equinox

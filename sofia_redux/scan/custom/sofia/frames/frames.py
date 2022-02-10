# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.frames.horizontal_frames import HorizontalFrames
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.telescope_coordinates import \
    TelescopeCoordinates

__all__ = ['SofiaFrames']


class SofiaFrames(HorizontalFrames):

    def __init__(self):
        super().__init__()
        self.utc = None
        self.object_equatorial = None
        self.sofia_location = None
        self.instrument_vpa = None
        self.telescope_vpa = None
        self.chop_vpa = None
        self.pwv = None

    @property
    def default_field_types(self):
        """
        Used to define the default values for data arrays.

        Returns a dictionary of structure {field: default_value}.  The default
        values have the following effects:

        type - empty numpy array of the given type.
        value - full numpy array of the given value.
        astropy.units.Unit - empty numpy array (float) in the given unit.
        astropy.units.Quantity - full numpy array of the given quantity.

        If a tuple is provided, the array will have additional axes appended
        such that the first element gives the type as above, and any additional
        integers give additional axes dimensions,  e.g. (0.0, 2, 3) would
        result in a numpy array filled with zeros of shape (self.size, 2, 3).

        Returns
        -------
        fields : dict
        """
        fields = super().default_field_types
        fields.update({
            'utc': np.nan,
            'instrument_vpa': np.nan * units.Unit('deg'),
            'telescope_vpa': np.nan * units.Unit('deg'),
            'chop_vpa': np.nan * units.Unit('deg'),
            'pwv': np.nan * units.Unit('um'),
            'object_equatorial': (EquatorialCoordinates, 'degree'),
            'sofia_location': (GeodeticCoordinates, 'degree')
        })
        return fields

    def project(self, position, projector, indices=None):
        """
        Project positions to offsets.

        Parameters
        ----------
        position : Coordinate2D
            The (x, y) focal plane offsets.
        projector : AstroProjector
            The projector to store and determine the projected offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to project.

        Returns
        -------
        offset : Coordinate2D
            The projected (x, y) offsets.
        """
        if isinstance(projector.coordinates, TelescopeCoordinates):
            projector.set_refererence_coordinates()
            offset = self.get_native_offset(position, indices=indices)
            projector.coordinates.add_native_offset(offset)
            return projector.project()
        else:
            return super().project(position, projector, indices=indices)

    def telescope_to_native_equatorial_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert telescope offsets to native equatorial offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The telescope (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        native_equatorial_offsets : Coordinate2D
        """
        if indices is None:
            telescope_vpa = self.telescope_vpa
        else:
            telescope_vpa = self.telescope_vpa[indices]

        angle = (np.pi * units.Unit('radian')) - telescope_vpa
        if not in_place:
            offset = offset.copy()
        Coordinate2D.rotate_offsets(offset, angle)
        return offset

    def native_equatorial_to_telescope_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert native equatorial offsets to telescope offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The native equatorial (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        telescope_offsets : Coordinate2D
        """
        if indices is None:
            telescope_vpa = self.telescope_vpa
        else:
            telescope_vpa = self.telescope_vpa[indices]

        angle = (np.pi * units.Unit('radian')) + telescope_vpa
        if not in_place:
            offset = offset.copy()
        Coordinate2D.rotate_offsets(offset, angle)
        return offset

    def telescope_to_equatorial_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert telescope offsets to equatorial offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The telescope (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        equatorial_offsets : Coordinate2D
        """
        offset = self.telescope_to_native_equatorial_offset(
            offset, indices=indices, in_place=in_place)
        offset.scale_x(-1.0)
        return offset

    def equatorial_to_telescope_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert equatorial offsets to telescope offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The equatorial (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        telescope_offsets : Coordinate2D
        """
        if not in_place:
            offset = offset.copy()
        offset.scale_x(-1.0)
        self.native_equatorial_to_telescope_offset(
            offset, indices=indices, in_place=True)
        return offset

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.projector.projector_2d import \
    Projector2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.celestial_coordinates import \
    CelestialCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.focal_plane_coordinates import \
    FocalPlaneCoordinates

__all__ = ['AstroProjector']


class AstroProjector(Projector2D):

    def __init__(self, projection):
        """
        Initialize an astronomical projector.

        The astronomical projector is an extension of the 2-dimensional
        projector, used to store coordinates and projection offsets.  It is
        designed to better deal with astronomical coordinates (celestial
        coordinates), and other spherical coordinate types.

        Such coordinates will be taken from the projection reference
        coordinate. If those reference coordinates are celestial in nature,
        they will be converted to equatorial coordinates (if not already)
        and stored for subsequent use.  In these cases, one will always have
        access to the projected/deprojected coordinates in both the original
        celestial frame, and an equatorial frame.

        Parameters
        ----------
        projection : Projection2D
        """
        self.equatorial = EquatorialCoordinates()
        self.celestial = None
        super().__init__(projection)
        if isinstance(self.coordinates, EquatorialCoordinates):
            self.equatorial = self.coordinates
        elif isinstance(self.coordinates, CelestialCoordinates):
            self.celestial = self.coordinates
            self.equatorial = EquatorialCoordinates()

    def copy(self):
        """
        Return a full copy of the Projector2D

        Returns
        -------
        AstroProjector
        """
        return super().copy()

    def __eq__(self, other):
        """
        Check whether this projection is equal to another.

        Parameters
        ----------
        other : AstroProjection

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if not isinstance(other, AstroProjector):
            return False
        if self.__class__ != other.__class__:  # pragma: no cover
            return False
        if self.celestial != other.celestial:
            return False
        if self.equatorial != other.equatorial:
            return False
        return True

    def is_horizontal(self):
        """
        Return whether the stored coordinates are horizontal.

        Returns
        -------
        bool
        """
        return isinstance(self.coordinates, HorizontalCoordinates)

    def is_focal_plane(self):
        """
        Return whether the stored coordinates are focal plane coordinates.

        Returns
        -------
        bool
        """
        return isinstance(self.coordinates, FocalPlaneCoordinates)

    def set_reference_coordinates(self):
        """
        Set the reference coordinates from the stored projection reference.

        If a celestial coordinate frame has been defined and equatorial
        coordinates exist, the celestial coordinates will be updated to the
        equivalent equatorial coordinates in the celestial frame.

        Returns
        -------
        None
        """
        super().set_reference_coordinates()
        if self.celestial is not None:
            self.celestial.to_equatorial(self.equatorial)

    def project_from_equatorial(self, offsets=None):
        """
        Project the stored equatorial coordinates to offsets.

        During this process, celestial coordinates will be updated to be
        equivalent to the equatorial coordinates in the celestial frame.

        Parameters
        ----------
        offsets : Coordinate2D, optional
            The coordinates used to store the results.  If not supplied,
            defaults to the projector stored offsets.

        Returns
        -------
        offsets : Coordinate2D
        """
        if self.celestial is not None:
            self.celestial.from_equatorial(self.equatorial)
        return self.project(self.coordinates, offsets=offsets)

    def deproject(self, offsets=None, coordinates=None):
        """
        Deproject offsets onto coordinates.

        If a celestial coordinate frame has been defined, it will be updated
        using the newly calculated equatorial coordinates if present.

        Parameters
        ----------
        offsets : Coordinate2D, optional
            The offsets to deproject.  If not supplied, defaults to the offsets
            stored in the projector.
        coordinates : Coordinate2D, optional
            The coordinates to hold the results.  If not supplied, defaults to
            the coordinates stored in the projector.

        Returns
        -------
        coordinates : Coordinate2D
        """
        coordinates = super().deproject(
            offsets=offsets, coordinates=coordinates)
        if self.celestial is not None:
            self.celestial.to_equatorial(self.equatorial)
        return coordinates

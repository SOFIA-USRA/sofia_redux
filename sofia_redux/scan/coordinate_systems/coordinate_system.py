# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC

from sofia_redux.scan.coordinate_systems.coordinate_axis import CoordinateAxis

__all__ = ['CoordinateSystem']


class CoordinateSystem(ABC):

    default_axes_labels = ['x', 'y', 'z', 'u', 'v', 'w']

    def __init__(self, name='Default Coordinate System', dimensions=None):
        """
        Initialize a coordinate system.

        A standard coordinate system is simply a collection of axes used to
        define that system.  Each axis contains a name and label.  By default,
        axes will be named x, y, z, y, v, w, yx, yy, yz, yu, yv, yw, zx, ....
        with increasing dimension.

        Parameters
        ----------
        name : str, optional
            The name of the coordinate system.
        dimensions : int, optional
            The number of axes in the coordinate system.
        """
        self.name = name
        self.axes = None
        if dimensions is not None:
            for dimension in range(dimensions):
                self.add_axis(
                    CoordinateAxis(label=self.dimension_name(dimension)))

    def __len__(self):
        """
        Return the number of axes in the coordinate system.

        Returns
        -------
        n_axis : int
        """
        if self.axes is None:
            return 0
        return len(self.axes)

    def __getitem__(self, axis_name):
        """
        Retrieve an axis of the given name.

        Parameters
        ----------
        axis_name : str
            The long or short name of the axis.

        Returns
        -------
        CoordinateAxis
        """
        if self.axes is None:
            raise KeyError("No available axes.")
        for axis in self.axes:
            if axis.label == axis_name:
                return axis
            elif (axis.short_label is not None
                  and axis.short_label == axis_name):
                return axis
        else:
            raise KeyError(f"Axis not found: {axis_name}")

    def __contains__(self, axis_name):
        """
        Return whether an axis already exists.

        Parameters
        ----------
        axis_name : str

        Returns
        -------
        bool
        """
        try:
            _ = self[axis_name]
            return True
        except KeyError:
            return False

    @property
    def size(self):
        """
        Return the number of axes in the coordinate system.

        Returns
        -------
        n_axis : int
        """
        return self.__len__()

    def dimension_name(self, dimension):
        """
        Return a default axis dimension name for a dimension number.

        Parameters
        ----------
        dimension : int

        Returns
        -------
        str
        """
        n_default = len(self.default_axes_labels)
        name = self.default_axes_labels[dimension % n_default]
        if dimension >= n_default:
            name += self.default_axes_labels[dimension // n_default]
        return name

    def add_axis(self, axis):
        """
        Add an axis to the coordinate system.

        Parameters
        ----------
        axis : CoordinateAxis

        Returns
        -------
        None
        """
        if axis.label in self:
            if axis.short_label is None:
                axis_name = axis.label
            else:
                axis_name = axis.short_label
            raise ValueError(f"{self.name} already has axis {axis_name}.")

        if self.axes is None:
            self.axes = [axis]
        else:
            self.axes.append(axis)

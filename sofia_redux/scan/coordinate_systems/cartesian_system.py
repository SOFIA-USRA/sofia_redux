# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem
from sofia_redux.scan.coordinate_systems.coordinate_axis import CoordinateAxis

__all__ = ['CartesianSystem']


class CartesianSystem(CoordinateSystem):

    labels = ['x', 'y', 'z', 'u', 'v', 'w', 't']

    def __init__(self, axes=2):
        """
        Initialize a cartesian system of coordinate axes.

        A cartesian system is defined by orthogonal axes that intersect at an
        origin.  Axes will be named x, y, z, u, v, w, and t for increasing
        dimensions.  If more than 7 dimensions are required, the corresponding
        axes will be named t1, t2, t3, etc.

        Parameters
        ----------
        axes : int, optional
        """
        super().__init__(name='Cartesian Coordinates')
        n_add = min(axes, len(self.labels))
        n_labels = len(self.labels)
        for i in range(n_add):
            self.add_axis(CoordinateAxis(label=self.labels[i]))
        if axes > n_labels:
            for i in range(n_labels, axes):
                self.add_axis(CoordinateAxis(label=f't{i - n_labels + 1}'))

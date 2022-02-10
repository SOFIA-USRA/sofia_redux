# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import enum
import numpy as np

from sofia_redux.scan.flags.flags import Flags

__all__ = ['MotionFlags']


class MotionFlags(Flags):

    class MotionFlagTypes(enum.Flag):

        X = enum.auto()
        Y = enum.auto()
        Z = enum.auto()
        X2 = enum.auto()
        Y2 = enum.auto()
        Z2 = enum.auto()
        X_MAGNITUDE = enum.auto()
        Y_MAGNITUDE = enum.auto()
        Z_MAGNITUDE = enum.auto()
        MAGNITUDE = enum.auto()
        NORM = enum.auto()
        TELESCOPE = enum.auto()
        SCANNING = enum.auto()
        CHOPPER = enum.auto()
        PROJECT_GLS = enum.auto()

    flags = MotionFlagTypes

    descriptions = {
        flags.X: 'x',
        flags.Y: 'y',
        flags.Z: 'z',
        flags.X2: 'x^2',
        flags.Y2: 'y^2',
        flags.Z2: 'z^2',
        flags.X_MAGNITUDE: '|x|',
        flags.Y_MAGNITUDE: '|y|',
        flags.Z_MAGNITUDE: '|z|',
        flags.MAGNITUDE: 'Magnitude',
        flags.NORM: 'Norm',
        flags.TELESCOPE: 'Telescope',
        flags.SCANNING: 'Scanning',
        flags.CHOPPER: 'Chopper',
        flags.PROJECT_GLS: 'Project GLS'
    }

    letters = {
        'x': flags.X,
        'y': flags.Y,
        'z': flags.Z,
        'i': flags.X2,
        'j': flags.Y2,
        'k': flags.Z2,
        'X': flags.X_MAGNITUDE,
        'Y': flags.Y_MAGNITUDE,
        'Z': flags.Z_MAGNITUDE,
        'M': flags.MAGNITUDE,
        'n': flags.NORM,
        't': flags.TELESCOPE,
        's': flags.SCANNING,
        'c': flags.CHOPPER,
        'p': flags.PROJECT_GLS
    }

    position_functions = {
        flags.X: lambda position: position.x.copy(),
        flags.Y: lambda position: position.y.copy(),
        flags.Z: lambda position: position.z.copy(),
        flags.X2: lambda position: position.x ** 2,
        flags.Y2: lambda position: position.y ** 2,
        flags.Z2: lambda position: position.z ** 2,
        flags.MAGNITUDE: lambda position: position.length,

        flags.X_MAGNITUDE: lambda position: np.abs(position.x),
        flags.Y_MAGNITUDE: lambda position: np.abs(position.y),
        flags.Z_MAGNITUDE: lambda position: np.abs(position.z),
        flags.NORM: lambda position: np.linalg.norm(
            position.coordinates, axis=0),
        flags(0): lambda position: position.x * np.nan,
    }

    @classmethod
    def convert_flag(cls, flag):
        """
        Convert a given user flag to a standard flag Enum.

        Parameters
        ----------
        flag : enum.Enum or None or int or str
            `None` will return flag(0).  str values will look for that given
            flag name.  Note that unlike the Flags class, the '|' character
            indicates magnitude rather than a separator for multiple flags.

        Returns
        -------
        enum.Enum
        """
        if isinstance(flag, enum.Flag):
            return flag
        elif isinstance(flag, int):
            return cls.flags(flag)
        elif isinstance(flag, str):
            attr = getattr(cls.flags, flag.upper().strip(), None)
            if attr is not None:
                return attr
            s = flag.strip().upper()
            if s.startswith('|'):
                return getattr(cls.flags, f'{s[1]}_MAGNITUDE')
            elif '^' in s:
                return getattr(cls.flags, ''.join(s.split('^')))
            elif s.startswith('M'):
                return getattr(cls.flags, 'MAGNITUDE')
            elif s.startswith('N'):
                return getattr(cls.flags, 'NORM')
            else:
                raise ValueError(f"Unknown flag: {flag}")
        else:
            raise ValueError(f"Invalid flag type: {flag}")

    def __init__(self, direction):
        """
        Initialize a MotionFlags object.

        Unlike most other flag classes, the MotionFlags class can be
        initialized for use in extracting motion signals from a given
        position.  The given position object (passed into
        :func:`MotionFlags.__call__` or :func:`MotionFlags.get_value`)
        must have retrievable values in the 'x', 'y', or 'z' attribute.

        Parameters
        ----------
        direction : int or str or enum.Enum
            The direction from which the MotionFlags object will extract
            position signals.  For example, `direction='x^2'` will return the
            value of the 'x' attribute squared from any given object.
        """
        self.direction = self.convert_flag(direction)
        self.position_function = self.position_functions.get(self.direction)
        if self.position_function is None:
            log.warning(f"No position function exists for "
                        f"{direction} direction.")
            log.warning("Resulting output will be NaN.")
            self.direction = self.flags(0)
            self.position_function = self.position_functions.get(
                self.direction)

    def get_value(self, position):
        """
        Return a value for the given position.

        Parameters
        ----------
        position : Coordinate

        Returns
        -------
        value : float or numpy.ndarray or units.Quantity
        """
        return self.position_function(position)

    def __call__(self, position):
        """
        Get the value for a given position.

        Parameters
        ----------
        position : Coordinate

        Returns
        -------
        value : float or numpy.ndarray or units.Quantity
        """
        return self.get_value(position)

    def __str__(self):
        """
        Return a string representation of the motion flag.

        Returns
        -------
        str
        """
        return self.__class__.__name__ + f': {self.direction}'

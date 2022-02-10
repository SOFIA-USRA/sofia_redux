# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
import re
import warnings

from astropy import units
from astropy.units import UnitConversionError
import numpy as np

__all__ = ['Range']


class Range(ABC):

    def __init__(self, min_val=-np.inf, max_val=np.inf, include_min=True,
                 include_max=True):
        """
        Initialize a Range object.

        A Range object defines a range of values that are considered "valid",
        and provides a few methods to check if given values fall within this
        range.

        Examples
        --------
        >>> r = Range(1, 3)
        >>> print(r(np.arange(5)))
        [False  True  True  True False]

        >>> r = Range(1, 3, include_min=False, include_max=False)
        >>> print(r(np.arange(5)))
        [False False  True False False]

        Parameters
        ----------
        min_val : int or float or units.Quantity, optional
            The minimum value of the range.
        max_val : int or float or units.Quantity, optional
            The maximum value of the range.
        include_min : bool, optional
            Whether the minimum value of the range is considered valid.
        include_max : bool, optional
            Whether the maximum value of the range is considered valid.
        """
        if isinstance(min_val, units.Quantity):
            if not isinstance(max_val, units.Quantity):
                if not np.isfinite(max_val):
                    max_val = max_val * min_val.unit
                else:
                    raise ValueError("Range units are incompatible.")
            else:
                try:
                    max_val.to(min_val.unit)
                except UnitConversionError:
                    raise ValueError("Range units are incompatible.")
        elif isinstance(max_val, units.Quantity):
            if not isinstance(min_val, units.Quantity):
                if not np.isfinite(min_val):
                    min_val = min_val * max_val.unit
                else:
                    raise ValueError("Range units are incompatible.")
        else:
            min_val = float(min_val)
            max_val = float(max_val)

        self.min = min_val
        self.max = max_val

        self.include_min = include_min
        self.include_max = include_max

    def copy(self):
        """
        Return a copy of the Range.

        Returns
        -------
        Range
        """
        return Range(min_val=self.min,
                     max_val=self.max,
                     include_max=self.include_max,
                     include_min=self.include_min)

    @property
    def midpoint(self):
        """
        Return the midpoint of the Range.

        Returns
        -------
        float or units.Quantity
        """
        return 0.5 * (self.min + self.max)

    @property
    def span(self):
        """
        Return the span of the range.

        The span is defined as the difference between the maximum and minimum
        range values.

        Returns
        -------
        float or units.Quantity
        """
        span = self.max - self.min
        if span < 0:
            if isinstance(self.min, units.Quantity):
                return 0.0 * self.min.unit
            else:
                return 0.0
        return span

    @property
    def lower_bounded(self):
        """
        Return whether the range has a lower limit.

        Returns
        -------
        bool
        """
        return not (np.isinf(self.min) and self.min < 0)

    @property
    def upper_bounded(self):
        """
        Return whether the range has an upper limit.

        Returns
        -------
        bool
        """
        return not (np.isinf(self.max) and self.max > 0)

    @property
    def bounded(self):
        """
        Return whether the Range has upper or lower bounds.

        Returns
        -------
        bool
        """
        return self.upper_bounded and self.lower_bounded

    def __eq__(self, other):
        """
        Return whether this Range is equal to another.

        Parameters
        ----------
        other : Range

        Returns
        -------
        equal : bool
        """
        if not isinstance(other, Range):
            return False
        if other is self:
            return True
        if self.min != other.min:
            return False
        if self.max != other.max:
            return False
        if self.include_min is not other.include_min:
            return False
        if self.include_max is not other.include_max:
            return False
        return True

    def __str__(self):
        """
        Return a string representation of the Range.

        Returns
        -------
        str
        """
        return f'({self.min} -> {self.max})'

    def __repr__(self):
        """
        Return a canonical string representation of the Range.

        Returns
        -------
        str
        """
        return object.__repr__(self) + f' {self}'

    def __call__(self, value):
        """
        Check whether given values fall inside the range.

        Parameters
        ----------
        value : int or float or units.Quantity or numpy.ndarray

        Returns
        -------
        in_range : bool or np.ndarray (bool)
        """
        return self.in_range(value)

    def __contains__(self, value):
        """
        Return whether a given value falls inside the valid range.

        Parameters
        ----------
        value : int or float or units.Quantity or Range

        Returns
        -------
        contained : bool
        """
        if isinstance(value, Range):
            return self.in_range(value.min) and self.in_range(value.max)
        return self.in_range(value)

    def in_range(self, value):
        """
        Return whether a given value is inside the valid range.

        Parameters
        ----------
        value : int or float or units.Quantity or numpy.ndarray.

        Returns
        -------
        valid : bool or numpy.ndarray (bool)
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if self.include_min:
                result = value >= self.min
            else:
                result = value > self.min
            if self.include_max:
                result &= value <= self.max
            else:
                result &= value < self.max
        return result

    def intersect_with(self, *args):
        """
        Intersects the current range with the new supplied range.

        Parameters
        ----------
        args : Range or value, value
           Either one or two arguments can be supplied.  If a single argument
           is received, it should be another Range object.  Otherwise, an
           intersection minimum and maximum must be supplied.

        Returns
        -------
        None
        """
        if len(args) == 1:
            min_value = args[0].min
            max_value = args[0].max
        elif len(args) > 2:
            raise ValueError(
                "Intersection requires two arguments (min, max) or a Range.")
        else:
            min_value, max_value = args

        if min_value > self.min:
            self.min = min_value
        if max_value < self.max:
            self.max = max_value

    def include_value(self, value):
        """
        Extend the range if necessary to include the provided value.

        Note that a NaN value will set the Range to be unbounded (-inf, inf).

        Parameters
        ----------
        value : int or float or units.Quantity.

        Returns
        -------
        None
        """
        if np.isnan(value):
            self.full()
            return
        if value < self.min:
            self.min = value
        elif value > self.max:
            self.max = value

    def include(self, *args):
        """
        Extend the range if necessary.

        Parameters
        ----------
        args : Range or value, value
           Either one or two arguments can be supplied.  If a single argument
           is received, it should be another Range object.  Otherwise, an
           inclusion minimum and maximum must be supplied.

        Returns
        -------
        None
        """
        if len(args) == 1 and isinstance(args[0], Range):
            args = args[0].min, args[0].max
        for arg in args:
            self.include_value(arg)

    def scale(self, value):
        """
        Scale the range by a given factor.

        Parameters
        ----------
        value : int or float or units.Quantity or units.Unit

        Returns
        -------
        None
        """
        self.min *= value
        self.max *= value

    def flip(self):
        """
        Swap the minimum and maximum values of the Range.

        Returns
        -------
        None
        """
        temp = self.min
        self.min = self.max
        self.max = temp

    def empty(self):
        """
        Remove the range values.

        Sets the (min,max) to (+inf,-inf).

        Returns
        -------
        None
        """
        self.min = np.inf
        self.max = -np.inf

    def is_empty(self):
        """
        Return whether the range is empty.

        A Range is considered empty when the minimum is greater than the
        maximum.

        Returns
        -------
        bool
        """
        return self.min > self.max

    def full(self):
        """
        Set the range to -inf -> +inf

        Returns
        -------
        None
        """
        self.min = -np.inf
        self.max = np.inf

    def is_intersecting(self, other):
        """
        Return whether this Range is intersecting with another.

        Parameters
        ----------
        other : Range

        Returns
        -------
        intersecting : bool
        """
        if other.is_empty():
            return False
        if self.is_empty():
            return False
        return (other.min in self) or (other.max in self) or (self in other)

    def grow(self, factor):
        """
        Grow a bounded range.

        The span is increased by the specified factor while keeping the
        midpoint fixed.

        Parameters
        ----------
        factor : int or float

        Returns
        -------
        None
        """
        if not self.bounded:
            return
        grow = 0.5 * (factor - 1) * self.span
        self.min -= grow
        self.max += grow

    @staticmethod
    def from_spec(spec, is_positive=False):
        """
        Return a Range object from a string specification.

        Specifications should be of the form min:max or min-max if
        `is_positive` is `True` where min and max should be replaced by a
        number.  min and max can also be replaced by '*' indicating no minimum
        or maximum range limit should be placed.  i.e., the spec 5:* would
        range from 5 to infinity.

        A single value can also be supplied by just supplying a single number
        without any ':' or '-' delimiter.  Note that the returned range will
        be inclusive and must not contain any units.  i.e. min <= valid <= max.
        You can always set units later by using the scale method.  e.g.,

        - An infinite range may be set with '*'
        - Unbounded ranges may be set with '>=#', '<=#', '>#', '<#' where #
          indicates a number.

        >>> r = Range.from_spec('1:3')
        >>> r.scale(units.Unit('minute'))
        >>> print(r)
        (1.0 min -> 3.0 min)

        Parameters
        ----------
        spec : str or None
            The range specification.  Usually read from a configuration file.
        is_positive : bool, optional
            If `True`, all values in the range are considered positive and any
            '-' character in `spec` will be treated as a delimiter rather than
            a minus sign.

        Returns
        -------
        Range
        """
        if spec is None:
            return None

        spec = str(spec).strip()
        if spec == '*':
            return Range(-np.inf, np.inf)
        if spec.startswith('>='):
            return Range(float(spec[2:]), np.inf, include_min=True)
        if spec.startswith('<='):
            return Range(-np.inf, float(spec[2:]), include_max=True)
        if spec.startswith('>'):
            return Range(float(spec[1:]), np.inf, include_min=False)
        if spec.startswith('<'):
            return Range(-np.inf, float(spec[1:]), include_max=False)

        splitter = r'[:-]' if is_positive else r'[:]'
        ranges = re.split(splitter, spec)
        n_range = len(ranges)
        if n_range not in [1, 2]:
            raise ValueError(f"Incorrect range spec format: {spec}")
        if n_range == 1:
            min_val = float(ranges[0])
            max_val = float(ranges[0])
        else:
            if ranges[0] == '*':
                min_val = -np.inf
            else:
                min_val = float(ranges[0])
            if ranges[1] == '*':
                max_val = np.inf
            else:
                max_val = float(ranges[1])
        return Range(min_val, max_val)

    @staticmethod
    def full_range():
        """
        Return a completely unbounded Range.

        The range will extend from -infinity -> +infinity.

        Returns
        -------
        Range
        """
        r = Range()
        r.full()
        return r

    @staticmethod
    def positive_range():
        """
        Return a range valid for positive values.

        Returns
        -------
        Range
        """
        return Range(0, np.inf)

    @staticmethod
    def negative_range():
        """
        Return a range valid for negative values.

        Returns
        -------
        Range
        """
        return Range(-np.inf, 0)

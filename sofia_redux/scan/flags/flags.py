# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from abc import ABC
import enum

from sofia_redux.scan.flags import flag_numba_functions

__all__ = ['Flags']


class Flags(ABC):
    """
    The Flags class contains methods to manipulate flagging.
    """
    flags = None
    descriptions = {}
    letters = {}

    @classmethod
    def all_flags(cls):
        """
        Return the flag containing all flags.

        Returns
        -------
        all_flags : enum.Enum
        """
        all_flags = cls.flags(0)
        for flag in cls.flags:
            all_flags = all_flags | flag
        return all_flags

    @classmethod
    def letter_to_flag(cls, letter):
        """
        Return the associated flag for a given string letter identifier.

        Parameters
        ----------
        letter : str
            A length 1 string.

        Returns
        -------
        flag : enum.Enum
        """
        return cls.letters.get(letter, cls.flags(0))

    @classmethod
    def flag_to_letter(cls, flag):
        """
        Return a letter representation of a given flag.

        Parameters
        ----------
        flag : enum.Enum or None or int or str
            `None` will return flag(0).  str values will look for that given
            flag name and may also use the '|' character to provide a
            combination of flags.

        Returns
        -------
        str
        """
        flag_value = cls.convert_flag(flag)
        result = ''
        for letter, test_flag in cls.letters.items():
            if flag_value & test_flag:
                result += letter
        if result == '':
            return '-'
        return result

    @classmethod
    def flag_to_description(cls, flag):
        """
        Return a description of a given flag.

        Parameters
        ----------
        flag : enum.Enum or None or int or str
            `None` will return flag(0).  str values will look for that given
            flag name and may also use the '|' character to provide a
            combination of flags.

        Returns
        -------
        str
        """
        flag_value = cls.convert_flag(flag)
        descriptions = []
        for check_flag, description in cls.descriptions.items():
            if check_flag & flag_value:
                descriptions.append(description)
        if len(descriptions) == 0:
            return ''
        if len(descriptions) == 1:
            return descriptions[0]
        return ' & '.join(descriptions)

    @classmethod
    def parse_string(cls, text):
        """
        Return the flag for a string of letter identifiers.

        Parameters
        ----------
        text : str
            A string containing single letter flag identifiers.

        Returns
        -------
        flag : enum.Enum
        """
        flag = cls.flags(0)
        for letter in str(text).strip():
            flag = flag | cls.letter_to_flag(letter)
        return flag

    @classmethod
    def convert_flag(cls, flag):
        """
        Convert a flag in various forms to a standard enum format.

        Parameters
        ----------
        flag : enum.Enum or None or int or str
            `None` will return flag(0).  str values will look for that given
            flag name and may also use the '|' character to provide a
            combination of flags.

        Returns
        -------
        enum.Enum
        """
        if flag is None:
            return cls.flags(0)
        if isinstance(flag, enum.Flag):
            return flag
        elif isinstance(flag, (int, np.integer)):
            flag = int(flag)
            if flag == -1:
                return cls.all_flags()
            else:
                return cls.flags(flag)
        elif isinstance(flag, str):
            if '|' in flag:
                new_flag = cls.flags(0)
                for flag_name in flag.split('|'):
                    if flag_name in cls.letters:
                        new_flag |= cls.letter_to_flag(flag_name)
                    else:
                        new_flag |= cls.convert_flag(flag_name)
                return new_flag
            else:
                if flag in cls.letters:
                    return cls.letter_to_flag(flag)
                else:
                    return getattr(cls.flags, flag.upper().strip())
        else:
            raise ValueError(f"Invalid flag type: {flag}")

    @classmethod
    def is_flagged(cls, thing, flag=None, indices=False, exact=False):
        """
        Return whether a given argument is flagged.

        Parameters
        ----------
        thing : numpy.ndarray (int) or str or enum.Enum or int
            An array of flags or a flag identifier.
        flag : enum.Enum or int or str, optional
            The flag to check.
        indices : bool, optional
            If `True` return an array of integer indices as returned by
            :func:`np.nonzero`.  Otherwise, return a boolean mask.
        exact : bool, optional
            If `True`, a value will only be considered flagged if it matches
            the given flag exactly.

        Returns
        -------
        flagged : numpy.ndarray (int or bool) or bool or tuple (int)
            If `indices` is `True` (only applicable when `thing` is an array),
            returns a numpy array of ints if the number of dimensions is 1.
            For N-D arrays the output will be similar to :func:`np.nonzero`.
            If `thing` is an array and `indices` is `False`, a boolean mask
            will be returned.  If `thing` contains a single value then `True`
            or `False` will be returned.
        """
        if flag is not None and not isinstance(flag, int):
            flag = cls.convert_flag(flag).value

        # For the single-value case
        if not hasattr(thing, '__len__'):
            if not isinstance(thing, int):
                thing = cls.convert_flag(thing).value
            if flag is None:
                return thing != 0
            elif flag == 0:
                return thing == 0
            elif exact:
                return thing == flag
            else:
                return (thing & flag) != 0

        if thing.size == 0:
            return np.empty(0, dtype=int if indices else bool)

        mask = flag_numba_functions.is_flagged(thing, flag=flag, exact=exact)
        if not indices:
            return mask

        result = np.nonzero(mask)
        return result[0] if mask.ndim == 1 else result

    @classmethod
    def is_unflagged(cls, thing, flag=None, indices=False, exact=False):
        """
        Return whether a given argument is flagged.

        Parameters
        ----------
        thing : numpy.ndarray (int) or str or enum.Enum or int
            An array of flags or a flag identifier.
        flag : enum.Enum or int or str, optional
            The flag to check.
        indices : bool, optional
            If `True` return an array of integer indices as returned by
            :func:`np.nonzero`.  Otherwise, return a boolean mask.
        exact : bool, optional
            If `True`, a value will only be considered unflagged if it is not
            exactly equal to the given flag.

        Returns
        -------
        flagged : numpy.ndarray (int or bool) or bool or tuple (int)
            If `indices` is `True` (only applicable when `thing` is an array),
            returns a numpy array of ints if the number of dimensions is 1.
            For N-D arrays the output will be similar to :func:`np.nonzero`.
            If `thing` is an array and `indices` is `False`, a boolean mask
            will be returned.  If `thing` contains a single value then `True`
            or `False` will be returned.
        """
        if flag is not None and not isinstance(flag, int):
            flag = cls.convert_flag(flag).value

        # For the single value case.
        if not hasattr(thing, '__len__'):
            if not isinstance(thing, int):
                thing = cls.convert_flag(thing).value
            if flag is None:
                return thing == 0
            elif flag == 0:
                return thing != 0
            elif exact:
                return thing != flag
            else:
                return (thing & flag) == 0

        if thing.size == 0:
            return np.empty(0, dtype=int if indices else bool)

        mask = flag_numba_functions.is_unflagged(thing, flag=flag, exact=exact)
        if not indices:
            return mask

        result = np.nonzero(mask)
        return result[0] if mask.ndim == 1 else result

    @classmethod
    def and_operation(cls, values, flag):
        """
        Return the result of an "and" operation with a given flag.

        Parameters
        ----------
        values : int or str or enum.Enum or numpy.ndarray (int)
            The values to "and".
        flag : int or str or enum.Enum
            The flag to "and" with.

        Returns
        -------
        and_result : int or numpy.ndarray (int)
        """
        if flag is not None and not isinstance(flag, int):
            flag = cls.convert_flag(flag).value

        if not hasattr(values, '__len__'):
            if not isinstance(values, int):
                values = cls.convert_flag(values).value

        return values & flag

    @classmethod
    def or_operation(cls, values, flag):
        """
        Return the result of an "or" operation with a given flag.

        Parameters
        ----------
        values : int or str or enum.Enum or numpy.ndarray (int)
            The values to "or".
        flag : int or str or enum.Enum
            The flag to "or" with.

        Returns
        -------
        or_result : int or numpy.ndarray (int)
        """
        if flag is not None and not isinstance(flag, int):
            flag = cls.convert_flag(flag).value

        if not hasattr(values, '__len__'):
            if not isinstance(values, int):
                values = cls.convert_flag(values).value

        return values | flag

    @classmethod
    def discard_mask(cls, flag_array, flag=None, criterion=None):
        r"""
        Return a mask indicating which flags do not match certain conditions.

        Parameters
        ----------
        flag_array : numpy.ndarray (int)
            An array of integer flags to check.
        flag : int or str or enum.Enum, optional
            The flag to check against.  If not supplied and non-zero flag is
            considered fair game in the relevant `criterion` schema.
        criterion : str, optional
            May be one of {'DISCARD_ANY', 'DISCARD_ALL', 'DISCARD_MATCH',
            'KEEP_ANY', 'KEEP_ALL', 'KEEP_MATCH'}.  If not supplied,
            'DISCARD_ANY' will be used if a flag is not supplied, and
            'DISCARD_ALL' will be used if a flag is supplied.  The '_ANY'
            suffix means `flag` is irrelevant and any non-zero value will be
            considered "flagged".  '_ALL' means that flagged values will
            contain 'flag', and '_MATCH' means that flagged values will
            exactly equal 'flag'.  'KEEP\_' inverts the True/False meaning
            of the output.

        Returns
        -------
        mask : numpy.ndarray (bool)
            An array the same shape as `flag_array` where `True` indicated that
            element met the given criterion.
        """
        flag_array = np.asarray(flag_array)

        if criterion is None:
            criterion = 'DISCARD_ANY' if flag is None else 'DISCARD_ALL'

        criterion = criterion.upper().strip()

        if criterion == 'DISCARD_ANY':
            return cls.is_flagged(flag_array, flag=None)
        elif criterion == 'DISCARD_ALL':
            return cls.is_flagged(flag_array, flag=flag)
        elif criterion == 'DISCARD_MATCH':
            return cls.is_flagged(flag_array, flag=flag, exact=True)
        elif criterion == 'KEEP_ANY':
            return cls.is_unflagged(flag_array, flag=None)
        elif criterion == 'KEEP_ALL':
            return cls.is_unflagged(flag_array, flag=flag)
        elif criterion == 'KEEP_MATCH':
            return cls.is_unflagged(flag_array, flag=flag, exact=True)
        else:
            raise ValueError(f"Invalid criterion flag: {criterion}")

    @classmethod
    def flag_mask(cls, flag_array, flag=None, criterion=None):
        r"""
        Return a mask indicating which flags that meet certain conditions.

        This is basically the same as `discard_mask`, but the meanings of
        KEEP and DISCARD are swapped.

        Parameters
        ----------
        flag_array : numpy.ndarray (int)
            An array of integer flags to check.
        flag : int or str or enum.Enum, optional
            The flag to check against.  If not supplied and non-zero flag is
            considered fair game in the relevant `criterion` schema.
        criterion : str, optional
            May be one of {'DISCARD_ANY', 'DISCARD_ALL', 'DISCARD_MATCH',
            'KEEP_ANY', 'KEEP_ALL', 'KEEP_MATCH'}.  If not supplied,
            'KEEP_ANY' will be used if a flag is not supplied, and
            'KEEP_ALL' will be used if a flag is supplied.  The '_ANY'
            suffix means `flag` is irrelevant and any non-zero value will be
            considered "flagged".  '_ALL' means that flagged values will
            contain 'flag', and '_MATCH' means that flagged values will
            exactly equal 'flag'.  'KEEP\_' inverts the True/False meaning
            of the output.

        Returns
        -------
        mask : numpy.ndarray (bool)
            An array the same shape as `flag_array` where `True` indicated that
            element met the given criterion.
        """
        flag_array = np.asarray(flag_array)

        if criterion is None:
            criterion = 'KEEP_ANY' if flag is None else 'KEEP_ALL'
        criterion = criterion.upper().strip()

        if criterion == 'KEEP_ANY':
            return cls.is_flagged(flag_array, flag=None)
        elif criterion == 'KEEP_ALL':
            return cls.is_flagged(flag_array, flag=flag)
        elif criterion == 'KEEP_MATCH':
            return cls.is_flagged(flag_array, flag=flag, exact=True)
        elif criterion == 'DISCARD_ANY':
            return cls.is_unflagged(flag_array, flag=None)
        elif criterion == 'DISCARD_ALL':
            return cls.is_unflagged(flag_array, flag=flag)
        elif criterion == 'DISCARD_MATCH':
            return cls.is_unflagged(flag_array, flag=flag, exact=True)
        else:
            raise ValueError(f"Invalid criterion flag: {criterion}")

    @classmethod
    def discard_indices(cls, flag_array, flag=None, criterion=None):
        r"""
        Return indices to discard for a given criterion/flag.

        Parameters
        ----------
        flag_array : numpy.ndarray (int)
            An array of integer flags to check.
        flag : int or str or enum.Enum, optional
            The flag to check against.  If not supplied and non-zero flag is
            considered fair game in the relevant `criterion` schema.
        criterion : str, optional
            May be one of {'DISCARD_ANY', 'DISCARD_ALL', 'DISCARD_MATCH',
            'KEEP_ANY', 'KEEP_ALL', 'KEEP_MATCH'}.  If not supplied,
            'DISCARD_ANY' will be used if a flag is not supplied, and
            'DISCARD_ALL' will be used if a flag is supplied.  The '_ANY'
            suffix means `flag` is irrelevant and any non-zero value will be
            considered "flagged".  '_ALL' means that flagged values will
            contain 'flag', and '_MATCH' means that flagged values will
            exactly equal 'flag'.  'KEEP\_' inverts the True/False meaning of
            the output.

        Returns
        -------
        indices : numpy.ndarray (int) or tuple (int)
            The indices to discard.  If `flag_array` has multiple dimensions,
            the result will be a tuple of integer arrays as would be returned
            by :func:`np.nonzero`.  Otherwise a 1-D integer array will be
            returned.
        """
        indices = np.nonzero(cls.discard_mask(flag_array, flag=flag,
                                              criterion=criterion))
        return indices[0] if len(indices) == 1 else indices

    @classmethod
    def flagged_indices(cls, flag_array, flag=None, criterion=None):
        r"""
        Return indices to for a given criterion/flag.

        This is the same as `discard_indices` with switched meanings of
        'DISCARD' and 'KEEP'.

        Parameters
        ----------
        flag_array : numpy.ndarray (int)
            An array of integer flags to check.
        flag : int or str or enum.Enum, optional
            The flag to check against.  If not supplied and non-zero flag is
            considered fair game in the relevant `criterion` schema.
        criterion : str, optional
            May be one of {'DISCARD_ANY', 'DISCARD_ALL', 'DISCARD_MATCH',
            'KEEP_ANY', 'KEEP_ALL', 'KEEP_MATCH'}.  If not supplied,
            'KEEP_ANY' will be used if a flag is not supplied, and
            'KEEP_ALL' will be used if a flag is supplied.  The '_ANY'
            suffix means `flag` is irrelevant and any non-zero value will be
            considered "flagged".  '_ALL' means that flagged values will
            contain 'flag', and '_MATCH' means that flagged values will
            exactly equal 'flag'.  'KEEP\_' inverts the True/False meaning
            of the output.

        Returns
        -------
        indices : numpy.ndarray (int) or tuple (int)
            The indices to discard.  If `flag_array` has multiple dimensions,
            the result will be a tuple of integer arrays as would be returned
            by :func:`np.nonzero`.  Otherwise a 1-D integer array will be
            returned.
        """
        indices = np.nonzero(cls.flag_mask(flag_array, flag=flag,
                                           criterion=criterion))
        return indices[0] if len(indices) == 1 else indices

    @classmethod
    def all_excluding(cls, flag):
        """
        Return all available flags with the exception of the one given here.

        Parameters
        ----------
        flag : str or int or enum.Enum
            The flag to not include.

        Returns
        -------
        flag : enum.Enum
        """
        return cls.unflag(cls.all_flags(), flag)

    @classmethod
    def unflag(cls, flag, remove_flag):
        """
        Return the result of unflagging one flag by another.

        Parameters
        ----------
        flag : int or str or enum.Enum
            The base flag.
        remove_flag : int or str or enum.Enum
            The flag to remove.

        Returns
        -------
        enum.Enum
        """
        flag = cls.convert_flag(flag)
        remove_flag = cls.convert_flag(remove_flag)

        if (flag.value & remove_flag.value) != 0:
            return flag ^ remove_flag
        else:
            return flag

    @classmethod
    def edit_header(cls, header, prefix=''):
        """
        Add the flags to a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        prefix : str, optional
            An optional prefix to add to the header key name.

        Returns
        -------
        None
        """
        if cls.flags is None:
            return
        if prefix is None:
            prefix = ''
        flags = cls.flags
        flag_class_name = flags.__name__
        for flag_name, flag_value in flags.__dict__.items():
            if isinstance(flag_value, enum.Enum):
                bit = flag_value.value
                key = f'{prefix}FLAG{bit}'
                value = flag_name
                description = cls.flag_to_description(flag_value)
                letter = cls.to_letters(flag_value)
                comment = f'{flag_class_name}'
                if description != '':
                    comment += f' {description}'
                if letter not in [None, '', '-']:
                    comment += f' ({letter})'
                header[key] = value, comment

    @classmethod
    def to_letters(cls, flag):
        """
        Convert a flag or flags to a string representation.

        Parameters
        ----------
        flag : str or int or enum.Enum or iterable
            The flag(s) to convert.

        Returns
        -------
        str or numpy.ndarray (str)
        """
        if not hasattr(flag, '__len__'):
            return cls.flag_to_letter(flag)

        flag_array = np.asarray(flag).copy()
        result = np.full(flag_array.shape, '?' * len(cls.letters))
        unique_flags = np.unique(flag_array)
        for flag in unique_flags:
            try:
                letter_representation = cls.flag_to_letter(flag)
            except ValueError:
                letter_representation = '-'
            result[flag_array == flag] = letter_representation
        return result.astype(str)

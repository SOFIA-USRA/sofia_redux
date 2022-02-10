# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.coordinates import (
    BaseCoordinateFrame, SkyCoord, SkyOffsetFrame)
import numpy as np
from scipy.sparse.csr import csr_matrix

from sofia_redux.scan.flags.flagged_data import FlaggedData
from sofia_redux.scan.flags import flag_numba_functions
from sofia_redux.scan.coordinate_systems.coordinate import Coordinate

__all__ = ['FlaggedDataGroup']


class FlaggedDataGroup(FlaggedData):

    def __init__(self, data, indices=None, name=None):
        """
        Create a subgroup of flagged data.

        The FlaggedDataGroup object is a wrapper around the FlaggedData class,
        accessing only certain indices and may be used to access or modify
        those elements of the original data.  However, due to the nature of
        indexing in numpy arrays, for value writing operations, the entire
        field must be modified at one.  For example self.gain[0]=1.234 will
        not result in any values being set.  To modify values, use
        self.gain=<new_array> or self.gain = <value> where <value> can be
        broadcast to the parent array.

        Parameters
        ----------
        data : FlaggedData
            The data to reference.
        indices : numpy.ndarray (int), optional
            The indices of FlaggedData that will belong to the group. If no
            indices are supplied, the entire FlaggedData will be referenced.
        name : str, optional
            The name of the group.
        """
        self.name = None
        self.data = None
        self._indices = None
        self._fixed_indices = None
        super().__init__()
        self.name = name
        self.apply_data(data)
        self.indices = indices  # property

    def copy(self, full=False):
        """
        Return a copy of the ChannelGroup.

        Parameters
        ----------
        full : bool, optional
            If `True`, reference a hard copy of the channel data rather than
            a reference to allow for local rather than global alteration.

        Returns
        -------
        ChannelGroup
        """
        self.reindex()
        if full:
            new = self.__class__(self.data.copy(),
                                 indices=self.indices,
                                 name=self.name)
        else:
            new = self.__class__(self.data,
                                 indices=self.indices,
                                 name=self.name)
        return new

    def __getitem__(self, indices):
        """
        Return a selection of the data.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)

        Returns
        -------
        FlaggedData
        """
        return self.create_data_group(indices=indices, name=None)

    def __getattr__(self, attribute):
        """
        Retrieves selected indices of the parent attribute.

        Parameters
        ----------
        attribute : str
            Name of the data field.

        Returns
        -------
        numpy.ndarray
        """
        if attribute == 'data':
            return object.__getattribute__(self, 'data')

        if attribute not in vars(self.data):
            raise AttributeError(f"'{self.__class__.__name__}' object has "
                                 f"no attribute '{attribute}'")

        value = getattr(self.data, attribute, None)
        if value is None:
            if not hasattr(self.data, attribute):  # pragma: no cover
                raise AttributeError(f"'{self.__class__.__name__}' object has "
                                     f"no attribute '{attribute}'")
            else:
                return value

        if self.indices is None:
            return value

        if isinstance(value, csr_matrix):
            return value[self.indices][:, self.indices]
        else:
            return value[self.indices]

    def __setattr__(self, attribute, value):
        """
        Sets selected indices of the parent data.

        Parameters
        ----------
        attribute : str
            Name of the data field.
        value : numpy.ndarray
            Value to set.  value.shape[0] must be equal to the size of the
            group data.

        Returns
        -------
        None
        """
        if attribute == 'data':
            object.__setattr__(self, 'data', value)

        if attribute in self.protected_attributes:
            super().__setattr__(attribute, value)
            return

        parent_value = getattr(self.data, attribute, None)
        if parent_value is None:
            return

        if value is None:
            super().__setattr__(attribute, value)
            return

        if self.indices is None:
            parent_value[...] = value

        elif isinstance(parent_value, csr_matrix):
            full_matrix = parent_value.toarray()
            if isinstance(value, csr_matrix):
                value = value.toarray()
            elif not isinstance(value, np.ndarray) or value.shape == ():
                value = np.full((self.size, self.size), value)

            r, c = np.indices(value.shape)
            r, c = self.indices[r], self.indices[c]
            full_matrix[r, c] = value
            setattr(self.data, attribute, csr_matrix(full_matrix))

        elif isinstance(value, (BaseCoordinateFrame, SkyCoord,
                                SkyOffsetFrame)):
            # Only data values may be set
            parent_value.cache.clear()
            parent_value.data.lat[self.indices] = value.data.lat
            parent_value.data.lon[self.indices] = value.data.lon
        elif isinstance(value, Coordinate):
            parent_value.paste(value, self.indices)
        else:
            parent_value[self.indices] = value

    @property
    def protected_attributes(self):
        """
        Protected attributes belong to the group, not the parent.

        These are used to distinguish between internal and referenced
        attributes.

        Returns
        -------
        set (str)
        """
        return {'data', 'name', '_indices', '_fixed_indices', 'indices',
                'fixed_indices'}

    @property
    def indices(self):
        """
        Returns the parent reference indices.

        Returns
        -------
        numpy.ndarray (int)
        """
        return self._indices

    @indices.setter
    def indices(self, values):
        """
        Define the referenced indices of the parent.

        Also, updates fixed indices which are stored in case those elements
        are deleted from the parent.

        Parameters
        ----------
        values : numpy.ndarray (int)
            The referenced indices.

        Returns
        -------
        None
        """
        if values is None:
            values = np.arange(self.data.size)
        else:
            values = np.atleast_1d(np.asarray(values, dtype=int))
        self._indices = values
        self._fixed_indices = self.data.fixed_index[values]

    @property
    def fixed_indices(self):
        """
        Return the fixed indices of parent class (copied to self).

        Returns
        -------
        numpy.ndarray (int)
        """
        return self._fixed_indices

    @fixed_indices.setter
    def fixed_indices(self, values):
        """
        Set the fixed indices of the group to reference the parent.

        The underlying indices are updated behind the scenes.

        Parameters
        ----------
        values : numpy.ndarray (int)
            The fixed indices to reference.

        Returns
        -------
        None
        """
        if values is None:
            self.indices = None  # Applies property setter for indices. (all)
            return
        values = np.atleast_1d(np.asarray(values, dtype=int))
        self._indices = self.data.find_fixed_indices(values)
        self._fixed_indices = self.data.fixed_index[self._indices]

    @property
    def fields(self):
        """
        Return the available parent data fields.

        Returns
        -------
        set (str)
        """
        if self.data is None:
            return set([])
        else:
            return self.data.fields

    @property
    def size(self):
        """
        Return the size of the data group.

        Returns
        -------
        size : int
            The number of referenced indices of the parent data.
        """
        if self.indices is None:
            return super().size
        if self.fixed_index is None:
            return 0
        return self.fixed_index.size

    @property
    def flagspace(self):
        """
        Return the flag space for the underlying FlaggedData object.

        Returns
        -------
        Flags
        """
        if self.data is None:
            return None
        return self.data.flagspace

    def apply_data(self, data):
        """
        Set the parent flagged data for the data group.

        Parameters
        ----------
        data : FlaggedData or FlaggedDataGroup or object
            If an object is supplied, it should have a 'data' attribute that
            is either FlaggedData or FlaggedDataGroup type.

        Returns
        -------
        None
        """
        if isinstance(data, FlaggedDataGroup):
            self.data = data.data
        elif isinstance(data, FlaggedData):
            self.data = data
        elif hasattr(data, 'data'):
            self.apply_data(getattr(data, 'data'))
        else:
            raise ValueError(f"Flagged data must be {FlaggedDataGroup} or "
                             f"{FlaggedData}, or contain such an "
                             f"object in the 'data' attribute.")

    def create_data_group(self, indices=None, name=None, keep_flag=None,
                          discard_flag=None, match_flag=None):
        """
        Creates and returns a data group.

        A data group is a referenced subset of flagged data.  Operations
        performed on a data group will be applied to the original data.

        Parameters
        ----------
        indices : numpy.ndarray (int), optional
            The indices to reference.  If not supplied, defaults to all
            indices in the parent FlaggedData object.
        name : str, optional
            The name of the group.  If not supplied, defaults to the
            name of this FlaggedDataGroup.
        discard_flag : int or str or FlagTypes, optional
            Flags to discard_flag from the new group.
        keep_flag : int or str or FlagTypes, optional
            Keep data with these matching flags.
        match_flag : int or str or FlagTypes, optional
            Keep only data with a flag exactly matching this flag.

        Returns
        -------
        FlaggedDataGroup
            A newly created data group.
        """
        if indices is None:
            indices = self.indices
        else:
            indices = self.to_indices(indices)
            if self.indices is not None:
                indices = self.indices[indices]

        for flag in [keep_flag, discard_flag, match_flag]:
            if flag is not None:
                flag_indices = self.get_flagged_indices(
                    keep_flag=keep_flag, discard_flag=discard_flag,
                    match_flag=match_flag)
                flag_indices = self.indices[flag_indices]
                indices = np.intersect1d(indices, flag_indices)
                break

        if name is None:
            name = self.name
        return self.__class__(self.data, indices=indices, name=name)

    def delete_indices(self, indices_or_mask):
        """
        Remove channels from the DataGroup.

        Actual indices should be passed in.  To delete based on fixed index
        values, please convert first using `find_fixed_indices`.

        Note that since this a FlaggedDataGroup which is a reference to the
        FlaggedData, only the reference indices are removed, not the data in
        the parent.

        Parameters
        ----------
        indices_or_mask : numpy.ndarray of (bool or int)
            The indices to delete, or a boolean mask where `True` marks an
            element for deletion.

        Returns
        -------
        None
        """
        keep_indices = self.to_indices(indices_or_mask, discard=True)
        self.indices = self.indices[keep_indices]

    def discard_flag(self, flag, criterion=None):
        r"""
        Given a flag, remove indices from the group data.

        Since this is a FlaggedDataGroup object, only the reference indices are
        removed, not data in the parent FlaggedData object.

        Parameters
        ----------
        flag : int or str or FlagTypes
            The flag to discard_flag.
        criterion : str, optional
            One of {'DISCARD_ANY', 'DISCARD_ALL', 'DISCARD_MATCH',
            'KEEP_ANY', 'KEEP_ALL', 'KEEP_MATCH'}.  \*_ANY refers to any flag
            that is not zero (unflagged).  \*_ALL refers to any flag that
            contains `flag`, and \*_MATCH refers to any flag that exactly
            matches `flag`.  The default (`None`), uses DISCARD_ANY if
            `flag` is None, and DISCARD_ALL otherwise.

        Returns
        -------
        None
        """
        self.delete_indices(self.flagspace.discard_indices(
            self.flag, flag, criterion=criterion))

    def set_flags(self, flag, indices=None):
        """
        Flag data with the supplied flag.

        Modified to access parent flags.

        Parameters
        ----------
        flag : int or str or Flags.flags
            The flag to set.
        indices : numpy.ndarray (int), optional
            The indices to flag.  If not supplied, all data are flagged.

        Returns
        -------
        None
        """
        if indices is None:
            indices = slice(None)
        elif indices.size == 0:
            return

        parent_flags = self.data.flag
        parent_indices = self.indices[indices]
        if not isinstance(flag, int):
            flag = self.flagspace.convert_flag(flag).value

        flag_numba_functions.set_flags(
            parent_flags, flag, indices=parent_indices)

    def unflag(self, flag=None, indices=None):
        """
        Remove data flags.

        Modified to access parent flags.

        Parameters
        ----------
        flag : int or str or Flags.flags
            The flag to remove.  If not supplied, all data are unflagged.
        indices : numpy.ndarray (int), optional
            The indices to flag.  If not supplied, all data are flagged.

        Returns
        -------
        None
        """
        if indices is None:
            indices = slice(None)
        elif indices.size == 0:
            return

        parent_flags = self.data.flag
        parent_indices = self.indices[indices]

        if flag is not None and not isinstance(flag, int):
            flag = self.flagspace.convert_flag(flag).value

        flag_numba_functions.unflag(
            parent_flags, flag=flag, indices=parent_indices)

    def reindex(self):
        """
        Validate channel indices with parent indices.
        """
        self.indices = self.data.find_fixed_indices(self._fixed_indices)

    def new_indices_in_old(self):
        """
        If reindexing occurs, returns the new indices on the old array.

        Returns
        -------
        new_indices : numpy.ndarray (int)
        """
        old_indices = self._fixed_indices
        new_indices = self.data.fixed_index
        new_on_old = np.nonzero(new_indices[:, None] == old_indices)[1]
        return new_on_old

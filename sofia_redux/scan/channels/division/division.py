# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC

__all__ = ['ChannelDivision']


class ChannelDivision(ABC):

    def __init__(self, name, groups=None):
        """
        Instantiates a channel division.

        A Channel division is a collection of channel data groups.  Each group
        name will be set to <name>-<i>, where <i> marks the index of a supplied
        group.  A data group consists of only the data arrays belonging to the
        master Channel object.

        Parameters
        ----------
        name : str
            The name of the division.
        groups : list of ChannelGroup
            A list of channel data groups to include in the division.
        """
        self.name = name
        if groups is None:
            self.groups = []
            return

        if not hasattr(groups, '__len__'):
            groups = [groups]
        self.groups = []
        for group in groups:
            if group is not None:
                self.groups.append(group)
        self.set_default_names()

    @property
    def size(self):
        """
        Return the number of channel groups in the channel division.

        Returns
        -------
        int
        """
        if self.groups is None:
            return 0
        return len(self.groups)

    @property
    def fields(self):
        """
        Returns the available fields for the division channel groups

        Returns
        -------
        set (str)
        """
        if self.size == 0:
            return set([])
        else:
            return self[0].fields

    def set_default_names(self):
        """
        Set the default names for each group in the channel division.

        Returns
        -------
        None
        """
        for i, group in enumerate(self.groups):
            group.name = f'{self.name}-{i + 1}'

    def validate_group_index(self, index_or_group_name):
        """
        Return the valid index of a given group.

        Raises an error if invalid.

        Parameters
        ----------
        index_or_group_name : int or str
           The name of the group, or the group index.

        Returns
        -------
        int
        """
        if self.size == 0:
            raise KeyError("No channel groups available in channel division.")

        if isinstance(index_or_group_name, int):
            index = index_or_group_name
        elif isinstance(index_or_group_name, str):
            index = self.get_group_name_index(index_or_group_name)
            if index is None:
                raise KeyError(f"Group {index_or_group_name} "
                               f"does not exist in division.")
        else:
            raise ValueError(f"Invalid index type: "
                             f"{type(index_or_group_name)}")

        if index < 0:
            reverse_index = self.size + index
            if reverse_index < 0:
                raise IndexError(f"Cannot use index {index} "
                                 f"with groups size {self.size}.")
            index = reverse_index

        if index >= self.size:
            raise IndexError(f"Group {index_or_group_name} out of range. "
                             f"Groups size = {self.size}.")
        return index

    def get_group_name_index(self, group_name):
        """
        Given a group name, return its groups index.

        Parameters
        ----------
        group_name : str
            The name of the group.

        Returns
        -------
        int
        """
        for index, group in enumerate(self.groups):
            if group.name == group_name:
                return index
        else:
            return None

    def __getitem__(self, index_or_group_name):
        """
        Return a group from the channel division.

        Parameters
        ----------
        index_or_group_name : int or str
            The index or name of the group.

        Returns
        -------
        ChannelGroup
        """
        index = self.validate_group_index(index_or_group_name)
        return self.groups[index]

    def __setitem__(self, index_or_group_name, group):
        """
        Set a channel group member of the division.

        Parameters
        ----------
        index_or_group_name : int or str
            The index or name of the group.
        group : ChannelGroup

        Returns
        -------
        None
        """
        index = self.validate_group_index(index_or_group_name)
        group_class = self.groups[index].__class__
        if not isinstance(group, group_class):
            raise ValueError(f"Group must be of {group_class} type.")

        self.groups[index] = group

    def __str__(self):
        """
        Return a string representation of the channel division.

        Returns
        -------
        str
        """
        class_name = self.__class__.__name__
        result = f"{class_name} ({self.name}): {self.size} group(s)"
        if self.size == 0:
            return result
        result += '\n' + '\n'.join([group.__str__() for group in self.groups])
        return result

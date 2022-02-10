# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.modality.correlated_modality import (
    CorrelatedModality)
from sofia_redux.scan.channels.mode.coupled_mode import CoupledMode

__all__ = ['CoupledModality']


class CoupledModality(CorrelatedModality):

    def __init__(self, modality, name=None, identity=None, gain_provider=None):
        """
        Create a coupled modality.

        Unlike the standard modality, a coupled modality must be supplied with
        a parent modality.  The modes of the parent are added to the coupled
        modality as coupled modes.

        Parameters
        ----------
        modality : Modality
        name : str, optional
            The name of the coupled modality.
        identity : str, optional
            A shorthand abbreviation for the coupled modality.
        gain_provider : GainProvider or str, optional
            If a string is provided, a FieldGainProvider will be created that
            defined the gain as that field of the channel group data.
            If a gain provider is explicitly provided, it will be used instead.
        """
        super().__init__(name=name, identity=identity)
        self.parent_modality = modality
        self.mode_class = CoupledMode
        if self.parent_modality.modes is not None:
            self.modes = [self.mode_class(mode, gain_provider=gain_provider)
                          for mode in self.parent_modality.modes]
        else:
            self.modes = None

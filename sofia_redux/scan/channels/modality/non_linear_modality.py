# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.channels.mode.non_linear_response \
    import NonLinearResponse

__all__ = ['NonlinearModality']


class NonlinearModality(Modality):

    def __init__(self, modality, name=None, identity=None, gain_provider=None):
        """
        Create a non-linear modality.

        Unlike the standard modality, a non-linear modality must be supplied
        with a parent modality.  The modes of the parent are used to create
        non-linear response modes in the non-linear modality.

        Parameters
        ----------
        modality : Modality
        name : str, optional
            The name of the non-linear modality.
        identity : str, optional
            A shorthand abbreviation for the non-linear modality.
        gain_provider : GainProvider or str, optional
            If a string is provided, a FieldGainProvider will be created that
            defined the gain as that field of the channel group data.
            If a gain provider is explicitly provided, it will be used instead.
        """
        super().__init__(name=name, identity=identity)
        self.parent_modality = modality
        self.mode_class = NonLinearResponse
        if self.parent_modality.modes is not None:
            self.modes = [self.mode_class(mode, gain_provider=gain_provider)
                          for mode in self.parent_modality.modes]
        else:
            self.modes = None

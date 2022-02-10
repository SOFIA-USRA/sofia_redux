# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.sofia.info.observation import SofiaObservationInfo

__all__ = ['HawcPlusObservationInfo']


class HawcPlusObservationInfo(SofiaObservationInfo):

    def is_aor_valid(self):
        """
        Checks whether the observation AOR ID is valid.

        Returns
        -------
        valid : bool
        """
        return super().is_aor_valid() & self.aor_id != '0'

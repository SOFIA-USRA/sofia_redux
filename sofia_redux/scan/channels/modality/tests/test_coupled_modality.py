# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.channels.modality.coupled_modality import CoupledModality
from sofia_redux.scan.channels.mode.coupled_mode import CoupledMode


class TestCoupledModality(object):
    def test_init(self, example_modality):
        modality = Modality()
        nonlin = CoupledModality(modality)

        # no parent modes
        assert nonlin.parent_modality is modality
        assert nonlin.mode_class is CoupledMode
        assert nonlin.modes is None

        # parent modes
        nonlin = CoupledModality(example_modality)
        assert nonlin.parent_modality is example_modality
        assert nonlin.mode_class is CoupledMode
        assert len(nonlin.modes) == 2
        for m in nonlin.modes:
            assert isinstance(m, CoupledMode)

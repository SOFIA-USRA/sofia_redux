# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.channels.modality.non_linear_modality \
    import NonlinearModality
from sofia_redux.scan.channels.mode.non_linear_response \
    import NonLinearResponse


class TestNonLinearModality(object):
    def test_init(self, example_modality):
        modality = Modality()
        nonlin = NonlinearModality(modality)

        # no parent modes
        assert nonlin.parent_modality is modality
        assert nonlin.mode_class is NonLinearResponse
        assert nonlin.modes is None

        # parent modes
        nonlin = NonlinearModality(example_modality)
        assert nonlin.parent_modality is example_modality
        assert nonlin.mode_class is NonLinearResponse
        assert len(nonlin.modes) == 2
        for m in nonlin.modes:
            assert isinstance(m, NonLinearResponse)

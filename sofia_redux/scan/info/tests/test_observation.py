# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.configuration.objects import ObjectOptions
from sofia_redux.scan.info.observation import ObservationInfo


class TestObservationInfo(object):
    def test_init(self):
        info = ObservationInfo()
        assert info.source_name is None
        assert info.project is None
        assert info.log_id == 'obs'

    def test_set_source(self):
        info = ObservationInfo()
        info.set_source('Mars')
        assert info.source_name == 'Mars'

        info.configuration = Configuration()
        info.configuration.objects = ObjectOptions()
        info.configuration.objects.set('Jupiter', {'test': True})
        info.set_source('Jupiter')
        assert info.source_name == 'Jupiter'
        assert 'Jupiter' in info.configuration.objects.applied_objects

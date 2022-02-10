# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.pipeline.pipeline import Pipeline
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.source_models.astro_intensity_map \
    import AstroIntensityMap


class TestPipeline(object):

    def test_parallel_scans(self):
        # 1 if no reduction
        pipe = Pipeline(None)
        assert pipe.parallel_scans == 1

        # 1 if no reduction.parallel_scans
        reduction = Reduction(None)
        reduction.parallel_scans = None
        pipe = Pipeline(reduction)
        assert pipe.parallel_scans == 1

        # otherwise return reduction value
        reduction.parallel_scans = 4
        assert pipe.parallel_scans == 4

    def test_parallel_tasks(self):
        # 1 if no reduction
        pipe = Pipeline(None)
        assert pipe.parallel_tasks == 1

        # 1 if no reduction.parallel_tasks
        reduction = Reduction(None)
        reduction.parallel_tasks = None
        pipe = Pipeline(reduction)
        assert pipe.parallel_tasks == 1

        # otherwise return reduction value
        reduction.parallel_tasks = 4
        assert pipe.parallel_tasks == 4

    def test_available_jobs(self):
        reduction = Reduction(None)
        pipe = Pipeline(reduction)

        reduction.parallel_scans = None
        reduction.parallel_tasks = 4
        assert pipe.available_jobs == 4

        reduction.parallel_scans = 2
        reduction.parallel_tasks = 4
        assert pipe.available_jobs == 8

    def test_configuration(self):
        pipe = Pipeline(None)
        assert pipe.configuration is None

        # unconfigured reduction has info = None
        reduction = Reduction(None)
        pipe = Pipeline(reduction)
        assert pipe.configuration is None

        reduction = Reduction('example')
        pipe = Pipeline(reduction)
        assert isinstance(pipe.configuration, Configuration)
        assert pipe.configuration is reduction.info.configuration

    def test_pipeline_id(self):
        pipe = Pipeline(None)
        assert pipe.pipeline_id.startswith('pipeline')

        reduction = Reduction(None)
        pipe = Pipeline(reduction)
        assert pipe.pipeline_id.startswith(str(id(reduction)))

    def test_set_source_model(self, mocker):
        reduction = Reduction('example')
        pipe = Pipeline(reduction)
        pipe.reduction.parallel_tasks = 4

        # without source
        pipe.set_source_model(None)
        assert pipe.scan_source is None

        # with source
        source = AstroIntensityMap(reduction.info)
        pipe.set_source_model(source)

        # sets a copy of the source object
        assert pipe.scan_source is not source
        assert isinstance(pipe.scan_source, AstroIntensityMap)

        # parallel_tasks is passed to source data
        assert pipe.scan_source.map.parallelism == 4

    def test_add_scan(self):
        pipe = Pipeline(None)
        assert pipe.scans is None
        pipe.add_scan('test')
        assert pipe.scans == ['test']
        pipe.add_scan(2)
        assert pipe.scans == ['test', 2]

    def test_set_ordering(self):
        pipe = Pipeline(None)
        pipe.set_ordering(['test', 'value'])
        assert pipe.ordering == ['test', 'value']

    def test_update_source(self, scan_file):
        # no op if no reduction
        pipe = Pipeline(None)
        pipe.update_source(None)

        # same if no source
        reduction = Reduction('example')
        pipe = Pipeline(reduction)
        pipe.update_source(None)

        # add a source and a scan
        reduction = Reduction('example')
        reduction.read_scans(scan_file)
        reduction.validate()
        pipe = Pipeline(reduction)
        pipe.set_source_model(reduction.source)

        scan = reduction.scans[0]
        pipe.update_source(scan)
        assert pipe.scan_source.enable_level

        # adds a comment to integration under some conditions
        scan.integrations[0].gain = -1
        pipe.update_source(scan)
        assert '-' in scan.integrations[0].comments

        scan.integrations[0].configuration.set_option('jackknife', True)
        scan.integrations[0].gain = 1
        pipe.update_source(scan)
        assert '+' in scan.integrations[0].comments

        # enable level is set False under some conditions
        scan.integrations[0].source_generation = 1
        pipe.update_source(scan)
        assert not pipe.scan_source.enable_level

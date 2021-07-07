# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Pipe class."""

import os

from sofia_redux.pipeline.pipe import Pipe
from sofia_redux.pipeline.configuration import Configuration


class TestPipe(object):
    def test_run(self, capsys):
        pipe = Pipe()
        data = "any data"
        pipe.run(data)

    def test_config(self, tmpdir):
        input = str(tmpdir.join('test_infiles.txt'))
        param = str(tmpdir.join('test_param.txt'))
        conf_dict = {'input_manifest': input,
                     'parameter_file': param}
        config = Configuration(conf_dict)
        pipe = Pipe(config)

        data = "any data"
        pipe.run(data)

        assert os.path.isfile(input)
        assert os.path.isfile(param)

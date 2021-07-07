# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Run Redux reduction objects from the command line."""

import argparse

from sofia_redux.pipeline.interface import Interface
from sofia_redux.pipeline.configuration import Configuration


class Pipe(Interface):
    """Command-line interface to data reduction steps."""

    def __init__(self, configuration=None):
        """
        Initialize the pipeline with an optional configuration file.

        Parameters
        ----------
        configuration : Configuration, optional
            Configuration items to be used for all reductions.
        """
        super().__init__(configuration)

    def run(self, data):
        """
        Run the pipeline on an input data set.

        Calls the `Interface` start and reduce methods.
        If specified in the configuration, an as-run input manifest,
        an output manifest, and an as-run parameter file will be
        saved to disk after the reduction is complete.

        Parameters
        ----------
        data : `list` of str or str
            Input data file paths, or a text file containing
            data file paths.
        """
        # start up the pipe
        self.start(data)

        # run the reduction
        try:
            self.reduce()
            # save the as-run parameters and the input/output manifest
            if self.configuration.parameter_file is not None:
                self.save_parameters(self.configuration.parameter_file)
            self.save_input_manifest()
            self.save_output_manifest()
        finally:
            if self.reduction is not None:
                self.clear_reduction()


def main():
    """Run a pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description='Reduce a set of data through a pipeline.')
    parser.add_argument('filename', metavar='filename', nargs='+',
                        help='Path to one or more input files.')
    parser.add_argument('-c', '--configuration', dest='config', type=str,
                        action='store', default=None,
                        help='Path to configuration file.')
    parser.add_argument('-l', '--loglevel', dest='loglevel', type=str,
                        action='store', default='INFO',
                        help='Log level.')
    args = parser.parse_args()

    Pipe.tidy_log(args.loglevel.upper())

    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration()

    pipe = Pipe(config)
    pipe.run(args.filename)

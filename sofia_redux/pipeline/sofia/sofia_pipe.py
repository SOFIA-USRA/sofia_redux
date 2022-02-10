# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Run SOFIA Redux reduction objects from the command line."""

import argparse
import sys
import warnings

from astropy import log

from sofia_redux.pipeline.pipe import Pipe
from sofia_redux.pipeline.sofia.sofia_configuration import SOFIAConfiguration


def main():
    """
    Run Redux in batch mode.

    The Redux pipeline may be started from the command line with
    the command::

        redux_pipe

    Input files are specified on
    the command line, either as a list of filenames or in an
    input manifest.  An input manifest should be a plain text file,
    containing a list of file paths to reduce, one per line.

    Redux will determine the appropriate reduction object for the
    input data, load all necessary pipeline steps, and run them in
    sequence with default parameter values.  At the end of the
    reduction, Redux will save an output manifest (containing all
    files produced by the pipeline), an as-run input manifest
    (containing all files actually processed by the pipeline),
    and an as-run parameter configuration file (containing all
    parameters used by the pipeline).

    Pipeline runs may be customized with command-line options to
    redux_pipe.

    The output directory for any files produced by the pipeline
    may be set with the '-o' command line option.  If the directory
    does not exist, it will be created.

    The log level for the terminal may be set with the
    '-l' command line option.  To quiet all log output, specify
    '-l critical'; for fully verbose log messages, specify '-l debug'.

    Other custom values may be specified in a configuration file in
    INI format on the command line. This file may contain:

    * an output directory for the pipeline
    * a file name for an as-run input manifest
    * a file name for the output manifest
    * a file name for the as-run parameter file
    * a log file name template (may contain `time.strftime` formatting keys)
    * the log level for the log file
    * the format for the log file

    For any file name, if it is set to a blank string, no file will be
    produced.

    In addition, configuration files may contain custom parameters
    for any pipeline step.  These should be specified with the pipeline
    step name as a section header (with an optional step index number),
    then a keyword = value pair for the parameter to be modified.
    Any parameters not specified are left at their default values.

    Parameter files may be generated interactively via the Redux GUI
    (see `sofia_redux.pipeline.sofia.redux_app`), then saved and fed to
    redux_pipe for batch-mode processing with custom parameters.

    See Also
    --------
    sofia_redux.pipeline.sofia.redux_app : Interactive processing

    Examples
    --------
    The following values, if placed in a configuration file
    (e.g. 'custom.cfg'), would replicate the current default settings::

        output_directory = .
        input_manifest = redux_infiles.txt
        output_manifest = outfiles.txt
        parameter_file = redux_param.cfg
        log_file = "redux_%Y%m%d_%H%M%S.log"
        log_level = DEBUG
        log_format = "%(asctime)s - %(origin)s - %(levelname)s - %(message)s"

    The following additional section would set the initial 'save'
    parameter to True for the pipeline step named 'calibrate'::

        [calibrate]
            save = True

    To run the Redux pipeline with these settings on input files specified
    in an input manifest called 'infiles.txt'::

        redux_pipe -c custom.cfg infiles.txt

    To run the Redux pipeline with a verbose log on a set of FITS files
    in the current directory::

        redux_pipe -l debug *.fits

    To run the Redux pipeline with no terminal or log window output::

        redux_pipe -l critical *.fits

    To redirect the output to a different directory::

        redux_pipe -o output *.fits
    """
    # suppress all runtime warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(
        description='Reduce a set of data through a SOFIA pipeline.')
    parser.add_argument('filename', metavar='filename', nargs='+',
                        help='Path to one or more input files.')
    parser.add_argument('-c', '--configuration', dest='config', type=str,
                        action='store', default=None,
                        help='Path to Redux configuration file.')
    parser.add_argument('-o', '--out', dest='outdir', type=str,
                        action='store', default=None,
                        help='Path to output directory.')
    parser.add_argument('-l', '--loglevel', dest='loglevel', type=str,
                        action='store', default='INFO',
                        help='Log level.')

    args = parser.parse_args()

    # format the log for pretty-printing to the terminal
    Pipe.tidy_log(args.loglevel.upper())

    # allow "None" to be passed for the configuration,
    # and interpreted as None
    if str(args.config).strip().upper() == 'NONE':
        args.config = None

    if args.config is not None:
        config = SOFIAConfiguration(args.config)
    else:
        config = SOFIAConfiguration()

    # specify an output directory if necessary
    if args.outdir is not None:
        config.output_directory = args.outdir

    # make a new pipeline from the SOFIA configuration
    pipe = Pipe(config)

    # run the pipeline on the input files
    pipe.run(args.filename)

    # one last whitespace
    log.info("")

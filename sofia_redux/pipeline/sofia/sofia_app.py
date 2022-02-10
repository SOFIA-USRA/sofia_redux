# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Run SOFIA Redux reduction objects interactively."""

import argparse
import sys
import warnings

from sofia_redux.pipeline.application import Application
from sofia_redux.pipeline.sofia.sofia_configuration import SOFIAConfiguration


def main():
    """
    Run the Redux GUI.

    The Redux application may be started from the command line with
    the command::

        redux

    Input files may then be loaded via
    the File menu (*File -> Open New Reduction*).  Redux will
    determine the appropriate reduction object for the input data, and
    load all necessary pipeline steps.

    Each reduction step has a number of parameters that can be edited
    before running the step. To examine or edit these parameters,
    click the *Edit* button next to the step name to bring up the
    parameter editor for that step. Within the parameter editor,
    all values may be edited.  Click *OK* to save the edited values
    and close the window. Click *Reset* to restore any
    edited values to their last saved values.  Click *Restore Defaults*
    to reset all values to their stored defaults.
    Click *Cancel* to discard all changes to the parameters and
    close the editor window.

    The current set of parameters can be displayed, saved to a file,
    or reset all at once using the *Parameters* menu. A previously
    saved set of parameters can also be restored for use with the
    current reduction (*Parameters -> Load Parameters*).

    After all parameters for a step have been examined and set to the
    user's satisfaction, a processing step can be run on all loaded
    files either by clicking *Step*, or the *Run* button next to the
    step name. Each processing step must be run in order, but if a
    processing step is selected in the *Step through:* widget,
    then clicking *Step* will treat all steps up through the selected
    step as a single step and run them all at once. When a step has
    been completed, its buttons will be grayed out and inaccessible.
    It is possible to undo one previous step by clicking *Undo*.
    All remaining steps can be run at once by clicking *Reduce*.
    After each step, the results of the processing may be displayed
    in a data viewer. After running a pipeline step or reduction,
    click *Reset* to restore the reduction to the initial state,
    without resetting parameter values.

    Files can be added to the reduction set (*File -> Add Files*) or
    removed from the reduction set (*File -> Remove Files*), but
    either action will reset the reduction for all loaded files.
    Select the *File Information* tab to display a table of information
    about the currently loaded files.

    It is possible to set some custom values before Redux starts up.

    The log level for the terminal and GUI log window may be set with the
    '-l' command line option.  To quiet all log output, specify
    '-l critical'; for fully verbose log messages, specify '-l debug'.

    Other custom values may be specified in a configuration file in
    INI format on the command line. This file may contain:

    * a default output directory for the pipeline
    * a default name for an as-run input manifest
    * a default name for the output manifest
    * a default name for the as-run parameter file
    * a log file name template (may contain `time.strftime` formatting keys)
    * the log level for the log file
    * the format for the log file

    All file names may be changed from within the GUI.

    In addition, configuration files may contain new default parameters
    for any pipeline step.  These should be specified with the pipeline
    step name as a section header (with an optional step index number),
    then a keyword = value pair for the parameter to be modified.
    Any parameters not specified are left at their default values.
    All parameters will still be editable at runtime within the GUI.
    For a complete example of the pipeline parameters for a particular
    reduction type, use the GUI to display or save the parameters
    (*Parameters -> Display All Parameters*, or
    *Parameters -> Save Parameters*).

    See Also
    --------
    sofia_redux.pipeline.sofia.redux_pipe : Batch mode processing

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

    To run Redux with these settings::

        redux -c custom.cfg

    To run Redux with a verbose log::

        redux -l debug

    To run Redux with no terminal or log window output::

        redux -l critical
    """
    # suppress all runtime warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Interactively reduce data through SOFIA pipelines.')
    parser.add_argument('-c', '--configuration', dest='config', type=str,
                        action='store', default=None,
                        help='Path to configuration file.')
    parser.add_argument('-l', '--loglevel', dest='loglevel', type=str,
                        action='store', default='INFO',
                        help='Log level.')

    args = parser.parse_args()

    # format the log for pretty-printing to the terminal
    Application.tidy_log(args.loglevel.upper())

    if args.config is not None:
        config = SOFIAConfiguration(args.config)
    else:
        config = SOFIAConfiguration()

    app = Application(config)
    app.run()

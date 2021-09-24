# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Interface to Redux reduction objects."""

import logging
import mimetypes
import os
import shutil
import time

import astropy
from astropy import log
from configobj import ConfigObj

import sofia_redux.pipeline
from sofia_redux.pipeline.configuration import Configuration


class TidyLogHandler(logging.StreamHandler):
    """Simple log handler for printing INFO messages."""
    def emit(self, record):
        """
        Print the log message.

        Parameters
        ----------
        record : `logging.LogRecord`
           The log record.
        """
        print(record.msg)


class Interface(object):
    """
    Interface to Redux reduction objects.

    Parent object to both command-line and graphical interfaces.

    Attributes
    ----------
    reduction: Reduction
        A Redux reduction object for the currently loaded data
    chooser: Chooser
        A Redux chooser object that selects reduction objects for new data
    configuration: Configuration
        A Redux configuration object that selects a chooser and
        sets any necessary parameters for the reduction.
    viewers: list of Viewer
        List of Redux Viewer objects associated with the reduction.
    """
    def __init__(self, configuration=None):
        """
        Initialize the interface, with an optional configuration.

        Parameters
        ----------
        configuration : `Configuration`, optional
            Configuration items to be used for all reductions.
        """

        # associations
        self.reduction = None
        self.chooser = None
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = Configuration()
        self.viewers = []

    def clear_reduction(self):
        """
        Clear reduction variables.

        Resets the reduction and viewers attributes.
        Removes any existing log file handlers.
        """
        try:
            self.reduction.cleanup()
            del self.reduction
        except Exception:
            pass
        self.reduction = None
        self.viewers = []

        # remove any old file handlers
        self.unset_log_file()

    def close_viewers(self):
        """Close any open viewers."""
        for viewer in self.viewers:
            viewer.close()

    def has_embedded_viewers(self):
        """
        Check for associated embedded viewers.

        Returns
        -------
        bool
            True if associated viewers are intended to be
            embedded in the main GUI; False otherwise
        """
        if self.reduction is not None and self.viewers:
            for viewer in self.viewers:
                if viewer.embedded:
                    return True
        return False

    def is_input_manifest(self, data):
        """
        Test if input data list is actually an input manifest file.

        Parameters
        ----------
        data : `list` of str
            List of input file names.

        Returns
        -------
        bool
            True if `data` is a single plain text file; False otherwise.
        """

        # currently, the test is just whether it is a plain text file
        if len(data) == 1 and os.path.isfile(data[0]):
            mtype = mimetypes.guess_type(data[0])
            if mtype[0] == 'text/plain':
                return True
            else:
                return False

    def load_configuration(self):
        """
        Set reduction chooser from configuration.

        All other configuration items are accessed as needed.
        """
        self.chooser = self.configuration.chooser

    def load_data_id(self):
        """Load the input file description from the reduction."""
        if self.reduction is None:
            return
        self.reduction.load_data_id()

    def load_files(self, data):
        """
        Load input files from a list of file names.

        The reduction object is assigned by the chooser, then
        an output directory and log file are set, and the data
        is loaded into the reduction object.

        Parameters
        ----------
        data : `list` of str or str
           Lists the input file names to be reduced.
        """
        if self.chooser is None:
            return
        self.reduction = self.chooser.choose_reduction(
            data, config=self.configuration.config)
        if self.reduction is None:
            return

        # set log file and output directory
        self.set_output_directory()
        self.set_log_file()

        self.reduction.load(data)

        # override the recipe from configuration if necessary
        self.set_recipe()

    def load_parameters(self, param_file=None):
        """
        Load reduction parameters.

        Default reduction parameters may optionally be overridden
        by a parameter file in INI format with reduction step names
        as section headers, and parameters specified in
        key, value pairs.

        Parameters
        ----------
        param_file : str, optional
            Parameter file name
        """
        if self.reduction is None:
            return
        if param_file is None:
            param_file = self.configuration.config
        self.reduction.param_file = param_file
        self.reduction.load_parameters()

    def read_input_manifest(self, data):
        """
        Read an input manifest and return input filenames.

        Input manifests are assumed to be plain text files that
        list one input file per line. Only file names that exist on
        disk as files are returned.

        Parameters
        ----------
        data : str or `list` of str
            File name for the input manifest.

        Returns
        -------
        list of str
            List of input file names.
        """
        # assume one input file per line
        infiles = []
        if type(data) is list:
            data = data[0]
        with open(data) as fh:
            for line in fh.readlines():
                line = line.strip()
                # keep only if it is a file that exists
                if os.path.isfile(line):
                    infiles.append(line)
        return infiles

    def reduce(self):
        """Call the reduce method of the current reduction."""
        if self.reduction is None:
            return
        self.reduction.reduce()

    def register_viewers(self, parent=None):
        """
        Retrieve and start viewers defined by the reduction object.

        Calls the :meth:`Reduction.register_viewers` method, to
        retrieve :class:`Viewer` objects, then calls the
        :meth:`Viewer.start` method to initialize them.  If
        configuration.update_display is False, no viewers will
        be registered.

        Parameters
        ----------
        parent: object, optional
            A parent widget for the viewer.  May be any
            type that the viewer understands
            (e.g. a Qt frame widget).
        """
        if self.reduction is None:
            return
        if self.configuration.update_display is False:
            self.viewers = []
            return
        self.viewers = self.reduction.register_viewers()
        for viewer in self.viewers:
            viewer.start(parent)

    def reset_reduction(self, data):
        """
        Reset the current reduction to its initial state.

        Calls the :meth:`Reduction.load` method on the input
        data.  Also resets a non-standard recipe if necessary.

        Parameters
        ----------
        data : `list` of str or str
            Input data filenames.
        """
        if self.reduction is None:
            return
        self.reduction.load(data)
        self.set_recipe()

    def reset_viewers(self):
        """
        Reset viewers to default state.

        Calls the :meth:`Viewer.reset` method
        """
        if self.reduction is None:
            return
        for viewer in self.viewers:
            viewer.reset()

    def save_input_manifest(self, filename=None, absolute_paths=None):
        """
        Save input manifest to disk.

        Writes a plain text file containing one file name per line.
        Input files are specified in the ``reduction.raw_files``
        attribute.

        Parameters
        ----------
        filename : str, optional
            Output file path to write to.  Default may be set in
            the ``configuration.input_manifest`` attribute.
            If ``configuration.input_manifest`` is not an absolute path,
            it will be joined with the ``reduction.output_directory``
            attribute.
        absolute_paths : bool, optional
            If True, input file names will be written as absolute
            paths.  If False, they will be written as relative paths.
            If None, the default value specified in
            ``configuration.absolute_paths`` will be used if present;
            otherwise, `absolute_paths` defaults to True.
        """
        if self.reduction is None:
            return
        if filename is None:
            filename = self.configuration.input_manifest
        # if still None, do nothing
        if filename is None or str(filename).strip() == '':
            return
        elif self.reduction.output_directory is not None \
                and not os.path.isabs(filename):
            filename = os.path.join(self.reduction.output_directory,
                                    filename)

        if absolute_paths is None:
            absolute_paths = self.configuration.absolute_paths
        # if still None, default to True
        if absolute_paths is None:
            absolute_paths = True

        infiles = self.reduction.raw_files
        if type(infiles) is not list:
            infiles = [infiles]
        if infiles:
            with open(filename, 'w') as manifest:
                for fname in infiles:
                    if absolute_paths:
                        fname = os.path.abspath(fname)
                    else:
                        fname = os.path.relpath(
                            os.path.abspath(fname),
                            os.path.dirname(os.path.abspath(filename)))
                    manifest.write("{}\n".format(fname))
            log.info("Wrote input manifest to {}".format(filename))
        else:
            log.warning("No input files; not saving input manifest.")

    def save_configuration(self, filename=None):
        """
        Save configuration values.

        If a file name is provided, this function writes a plain
        text file in INI format, containing top-level configuration
        values.

        Parameters
        ----------
        filename : str, optional
           File path to write to.

        Returns
        -------
        None or list of str
           The return value depends on the `filename` parameter.
        """
        conf = ConfigObj(self.configuration.config)
        if str(filename).strip() == '':
            filename = None
        if filename is not None:
            if self.configuration.output_directory is not None \
                    and not os.path.isabs(filename):
                filename = os.path.join(self.configuration.output_directory,
                                        filename)
            conf.filename = filename

        conf.initial_comment = [
            "Redux v{} Configuration".format(sofia_redux.pipeline.__version__)]
        text = conf.write()
        if filename is None:
            return text
        else:
            log.info("Wrote configuration file to {}".format(filename))

    def save_parameters(self, filename=None):
        """
        Save current reduction parameters.

        If a file name is provided, this function writes a plain
        text file in INI format.  Reduction step index and names
        are used as section headers, followed by key-value pairs
        for all currently defined parameters.

        If no filename is provided, this function will return
        a list of strings containing the parameters pretty-printed
        to INI format.

        Parameters
        ----------
        filename : str, optional
           File path to write to.

        Returns
        -------
        None or list of str
           The return value depends on the `filename` parameter.
        """
        if self.reduction is None:
            return
        conf = self.reduction.parameters.to_config()
        if str(filename).strip() == '':
            filename = None
        if filename is not None:
            if self.reduction.output_directory is not None \
                    and not os.path.isabs(filename):
                filename = os.path.join(self.reduction.output_directory,
                                        filename)
            conf.filename = filename

        conf.initial_comment = [
            "Redux parameters for {} instrument in {} mode".format(
                self.reduction.instrument,
                self.reduction.mode),
            "Pipeline: {} v{}".format(
                self.reduction.pipe_name,
                self.reduction.pipe_version),
        ]
        text = conf.write()
        if filename is None:
            return text
        else:
            log.info("Wrote parameter file to {}".format(filename))

    def save_output_manifest(self, filename=None, absolute_paths=None):
        """
        Save the output manifest.

        Writes a plain text file containing one file name per line.
        Output files are specified in the ``reduction.out_files``
        attribute.

        Parameters
        ----------
        filename : str, optional
            Output file path to write to.  Default may be set in
            the ``configuration.output_manifest`` attribute.
            If ``configuration.output_manifest`` is not an absolute path,
            it will be joined with the ``reduction.output_directory``
            attribute.
        absolute_paths : bool or None
            If True, output file names will be written as absolute
            paths.  If False, they will be written as relative paths.
            If None, the default value specified in
            ``configuration.absolute_paths`` will be used if present;
            otherwise, `absolute_paths` defaults to True
        """
        if self.reduction is None:
            return
        if filename is None:
            filename = self.configuration.output_manifest
        # if still None, do nothing
        if filename is None or str(filename).strip() == '':
            return
        elif self.reduction.output_directory is not None \
                and not os.path.isabs(filename):
            filename = os.path.join(self.reduction.output_directory,
                                    filename)

        if absolute_paths is None:
            absolute_paths = self.configuration.absolute_paths
        # if still None, default to True
        if absolute_paths is None:
            absolute_paths = True

        outfiles = self.reduction.out_files
        if type(outfiles) is not list:
            outfiles = [outfiles]
        if outfiles:
            with open(filename, 'w') as manifest:
                for fname in outfiles:
                    if absolute_paths:
                        fname = os.path.abspath(fname)
                    else:
                        fname = os.path.relpath(
                            os.path.abspath(fname),
                            os.path.dirname(os.path.abspath(filename)))
                    manifest.write("{}\n".format(fname))
            log.info("Wrote output manifest to {}".format(filename))
        else:
            log.warning("No output files; not saving output manifest.")

    def set_log_file(self, filename=None):
        """
        Set a log file handler for the reduction.

        If the provided file name is not an absolute path,
        it will be joined with ``reduction.output_directory``.
        When called, any existing file handlers will be removed.
        If a log file has already been written, it will be moved
        to the new file name; any further log messages will be
        appended to the existing log file.

        Parameters
        ----------
        filename : str, optional
           Log file path name.  If not provided, the
           ``configuration.log_file`` attribute will be used.
           If this value is None, no action will be taken.
        """
        if self.reduction is None:
            return
        if filename is None:
            filename = self.configuration.log_file
        if filename is not None and str(filename).strip() != '':
            # add time stamp to filename
            filename = time.strftime(filename)

            # add output directory to filename
            if self.reduction.output_directory is not None \
                    and not os.path.isabs(filename):
                filename = os.path.join(self.reduction.output_directory,
                                        filename)

            # remove any old file handlers
            for hand in log.handlers:
                if isinstance(hand, logging.FileHandler):
                    old_log = hand.baseFilename
                    log.removeHandler(hand)
                    del hand
                    # TODO - possible race condition here
                    if os.path.isfile(old_log):
                        shutil.move(old_log, filename)

            # add the new file handler
            fhand = logging.FileHandler(filename, 'at')
            if self.configuration.log_level is not None:
                fhand.setLevel(self.configuration.log_level)
            if self.configuration.log_format is not None:
                fhand.setFormatter(
                    logging.Formatter(self.configuration.log_format))
            log.addHandler(fhand)

            # log the file name
            log.info("Log file: {}".format(filename))

    def set_output_directory(self, dirname=None):
        """
        Set the output directory for the reduction.

        Makes the directory if it does not yet exist.  The directory
        name is stored in the ``reduction.output_directory`` attribute.

        Parameters
        ----------
        dirname : str, optional
            Path to the output directory.  If not provided, the
            ``configuration.output_directory`` attribute will
            be used.  If this value is None, no action is taken.
        """
        if self.reduction is None:
            return
        if dirname is None:
            dirname = self.configuration.output_directory
        if dirname is not None:
            # make the directory if it does not exist.
            # Allow OS/type errors to be raised here.
            dirname = os.path.abspath(dirname)
            os.makedirs(dirname, exist_ok=True)
            self.reduction.output_directory = dirname

    def set_recipe(self, recipe=None):
        """
        Set a non-default processing recipe for the reduction.

        All specified processing steps must be known to the
        current reduction object.  The reduction is not reset
        after setting the new recipe.

        Parameters
        ----------
        recipe : list of str, optional
            List of processing step names to use in place of
            the default reduction.  If not provided, the
            ``configuration.recipe`` attribute will
            be used.  If this value is also None, no action
            will be taken.
        """
        if self.reduction is None:
            return
        if recipe is None:
            recipe = self.configuration.recipe
        if recipe is not None:
            log.warning('Setting a new recipe: {}'.format(recipe))
            self.reduction.recipe = recipe

    def start(self, data):
        """
        Start a new reduction from an input data set.

        Parameters
        ----------
        data : `list` of str or str
            Input data.
        """
        self.load_configuration()

        # check for input manifest file
        if type(data) is not list:
            data = [data]
        if self.is_input_manifest(data):
            data = self.read_input_manifest(data)

        # load files from input data (usually filenames)
        self.load_files(data)

        # load parameters
        self.load_parameters()

    def step(self):
        """
        Run a reduction step.

        Calls the current reduction's `Reduction.step` method.

        Returns
        -------
        str
            Empty string if step was successful; an error message otherwise.
        """
        if self.reduction is None:
            return ''

        # output status is empty string
        # or error message
        status = self.reduction.step()
        return status

    def update_configuration(self, config_file):
        """
        Update top-level configuration.

        Default configuration parameters may optionally be overridden
        by a parameter file in INI format, specified in key, value pairs.

        Parameters
        ----------
        config_file : str, dict, or ConfigObj
            Configuration to update with.
        """
        self.configuration.update(config_file)

    def update_viewers(self):
        """
        Update all viewers associated with the current reduction.

        Calls the current viewers' `Viewer.update` method with the
        current reduction's ``reduction.display_data``.  If
        configuration.update_display is False, the update will not be
        performed.
        """
        if self.reduction is None:
            return
        if self.configuration.update_display is False:
            return
        for viewer in self.viewers:
            if viewer.name in self.reduction.display_data:
                viewer.update(self.reduction.display_data[viewer.name])

    def unset_log_file(self):
        """Remove any existing log file handlers."""
        # remove any old file handlers
        for hand in log.handlers:
            if isinstance(hand, logging.FileHandler):
                log.removeHandler(hand)

    @staticmethod
    def _info_only(x):
        return x.levelno == logging.INFO

    @staticmethod
    def _not_info(x):
        return x.levelno != logging.INFO

    @staticmethod
    def tidy_log(loglevel="INFO"):
        """
        Tidy up log printing for easier human readability at INFO level.

        INFO messages to the terminal will be printed directly.
        Messages at all other levels will use ``astropy.log`` format.
        Log file formatting is unaffected.

        Parameters
        ----------
        loglevel : str or int, optional
            Logging level to use for the terminal log.  May be any of
            the values accepted by the logging module
            (CRITICAL, ERROR, WARNING, INFO, DEBUG).
        """

        # set overall level as low as possible, so that messages are
        # accessible if necessary
        log.setLevel("DEBUG")

        # check to see if log has already been tidied
        found = False
        for hand in log.handlers:
            if isinstance(hand, TidyLogHandler):
                # just set the level
                hand.setLevel(loglevel)
                found = True
            elif isinstance(hand, astropy.logger.StreamHandler):
                # configure astropy terminal log to ignore info
                # messages
                hand.addFilter(Interface._not_info)
                hand.setLevel(loglevel)

        # if not already found, add a new handler
        if not found:
            stream_handler = TidyLogHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            stream_handler.addFilter(Interface._info_only)
            stream_handler.setLevel(loglevel)
            log.addHandler(stream_handler)

    @staticmethod
    def reset_log(loglevel=None):
        """
        Reset astropy log to standard settings.

        Removes any TidyLogHandlers, and resets filters for
        any astropy LogHandlers.

        Parameters
        ----------
        loglevel : str or int, optional
            Logging level to use for the terminal log.  May be any of
            the values accepted by the logging module
            (CRITICAL, ERROR, WARNING, INFO, DEBUG).  If not specified,
            log levels will not be modified.
        """
        for hand in log.handlers:
            if isinstance(hand, TidyLogHandler):
                log.removeHandler(hand)
            elif isinstance(hand, astropy.logger.StreamHandler):
                hand.removeFilter(Interface._not_info)
                if loglevel is not None:
                    hand.setLevel(loglevel)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pipeline step that processes a single input object."""

import time

from astropy import log
from configobj import ConfigObj

from sofia_redux.instruments.hawc.dataparent import DataParent

__all__ = ['StepParent']


class StepParent(object):
    """
    Pipeline step parent class.

    This class defines a pipeline step. Pipeline steps are
    the modules responsible for all data reduction tasks. Input
    and output data are passed as pipeline data objects (`DataFits`).
    This class expects that data is passed as a single data object,
    and the output is also a single data object (single-in,
    single-out (SISO) mode).

    All pipeline steps inheriting from this class must define a
    `setup` function that initializes data reduction parameters
    and metadata, and a `run` function that performs the data reduction.

    This class is callable: given a data object input and keyword
    arguments corresponding to the step parameters, it calls the run
    function and returns a data object as output.
    """
    def __init__(self):
        # initialize input and output
        self.datain = DataParent()
        self.dataout = DataParent()

        # placeholder for any extra output file names produced
        # by the pipeline (region files, PNG images, etc.)
        self.auxout = []

        # set names
        self.name = None
        self.description = None
        self.procname = None

        # set parameters
        # Dictionary with current arguments
        self.arglist = {}
        # List with possible parameters
        self.paramlist = []

        # set configuration
        self.config = None

        # specify whether this step runs on a single
        # PipeData object with a single output PipeData
        # object (SISO), multiple input PipeData objects
        # with multiple output PipeData objects (MIMO),
        # or multiple input Pipefile objects with a single
        # output PipeData object (MISO).
        self.iomode = 'SISO'

        # do local setup
        self.setup()

    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        This function is called at the end of __init__ to establish
        parameters and metadata specific to a pipeline step.

        The name of the step and a short description should be set,
        as well as a three-letter abbreviation for the step. The first
        two values are used for header history and pipeline display;
        the abbreviation is used in the output filenames.

        Parameters are stored in a list, where each element is a list
        containing the following information:

            - name: The name for the parameter. This name is used when
              calling the pipe step from a python shell. It is also
              used to identify the parameter in the pipeline
              configuration file.
            - default: A default value for the parameter. If nothing, set
              '' for strings, 0 for integers and 0.0 for floats.
            - help: A short description of the parameter.

        """
        # Name of the pipeline reduction step
        self.name = 'parent'
        self.description = 'Step Parent'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'unk'

        # Clear Parameter list
        self.paramlist = []

    def run(self):
        """
        Run the data reduction algorithm.

        Input is read from self.datain. The result is
        set in self.dataout.
        """
        # Copy datain to dataout
        self.dataout = self.datain

    def __call__(self, datain, **kwargs):
        """
        Run the pipeline step.

        Parameters
        ----------
        datain : DataFits or DataText
            Input data.
        **kwargs
            Parameter name, value pairs to pass to the pipeline step.

        Returns
        -------
        DataFits or DataText
        """
        # Get input data
        self.datain = datain

        # Start Setup
        self.runstart(self.datain, kwargs)

        # Call the run function
        self.run()

        # Finish - call end
        self.runend(self.dataout)

        # return result
        return self.dataout

    def runstart(self, data, arglist):
        """
        Initialize the pipeline step.

        This method should be called after setting self.datain,
        and before calling self.run.

        Sends an initial log message, checks the validity of the
        input data, and gets the configuration from input data.

        Parameters
        ----------
        data : DataFits or DataText
            Input data to validate.
        arglist : dict
            Parameters to pass to the step.
        """
        # Start Message
        log.info('Start Reduction: Pipe Step %s' % self.name)

        # Set input arguments
        for k in arglist.keys():
            self.arglist[k.lower()] = arglist[k]

        # Check input data type and set data config
        if issubclass(data.__class__, DataParent):
            self.config = data.config
        else:
            msg = 'Invalid input data type: DataParent ' \
                  'child object is required'
            log.error(msg)
            raise TypeError('Runstart: ' + msg)

        # Set Configuration
        if self.config is None:
            # no config specified, make an empty one
            self.config = ConfigObj()
            self.config[self.name] = {}

        # Check configuration
        if not isinstance(self.config, ConfigObj):
            msg = 'Invalid configuration information - aborting'
            log.error(msg)
            raise RuntimeError('Runstart: ' + msg)

    def runend(self, data):
        """
        Clean up after a pipeline step.

        This method should be called after calling self.run.

        Sends a final log message, updates the header in
        the output data, and clears input parameter arguments.

        Parameters
        ----------
        data : DataFits or DataText
            Output data to update.
        """
        # update header (status and history)
        self.updateheader(data)

        # clear input arguments
        self.arglist = {}
        log.info('Finished Reduction: Pipe Step %s' % self.name)

    def updateheader(self, data):
        """
        Update the header for a data object.

        This function:

            - Updates the filename with the self.procname value
            - Sets the PROCSTAT and PRODTYPE keywords in the data header
            - Adds a history entry to the data header

        Data is modified in place.

        Parameters
        ----------
        data : DataFits or DataText
            Output data to update.
        """
        # Update PRODuct TYPE keyword with step name, add history keyword
        data.setheadval('PRODTYPE', self.name, 'Product Type')
        histmsg = 'Reduced: {} {}'.format(
            self.name, time.strftime('%Y-%m-%d_%H:%M:%S'))
        data.setheadval('HISTORY', histmsg)

        # Add input parameters to history
        for p in [par[0] for par in self.paramlist]:
            histmsg = ' %s: %s=%s' % (self.name, p, self.getarg(p))
            data.setheadval('HISTORY', histmsg)

        # Update file name with procname
        data.filename = data.filenamebegin + \
            self.procname.upper() + data.filenameend

        # Add config file name if available and not already present
        # in HISTORY
        for fname in data.config_files:
            # New history message
            basename = fname.split(str(data.data_path))[-1]
            histmsg = 'CONFIG: %s' % basename

            # Check history for presence of the full message or possibly
            # a truncated version (eg. for long filenames in FITS headers)
            full_history = data.getheadval('HISTORY')
            if len(histmsg) > 72:
                shortmsg = histmsg[0:72]
            else:
                shortmsg = histmsg
            if histmsg not in full_history and shortmsg not in full_history:
                log.debug('Recording config file name %s' % fname)
                data.setheadval('HISTORY', histmsg)

    def getarg(self, parname):
        """
        Return the value of a parameter.

        The parameter is first searched for in self.arglist['parname'],
        then in config['stepname']['parname']. If the parameter is not found,
        the default value from parameter list is returned.
        Should the parameter name not have an entry in the parameter list,
        a error is returned and a KeyError is raised.

        All name comparisons are made in lower case.

        Parameters
        ----------
        parname : str
            The parameter name.

        Returns
        -------
        bool, int, float, str, or list
            The parameter value.

        Raises
        ------
        KeyError
            If the parameter name is not found.
        """
        # list of strings that should parse to boolean true
        # we need to handle booleans separately, because bool("False")
        # evaluates to True
        booltrue = ['yes', 'true', '1', 't']

        # so we don't have to worry about case
        parname = parname.lower()

        # Get paramlist index and check if parameter is valid
        try:
            ind = [par[0].lower() for par in self.paramlist].index(parname)
        except ValueError:
            msg = 'GetArg: There is no parameter named %s' % parname
            log.error(msg)
            raise KeyError(msg)
        # ParName in original Case
        parnameraw = self.paramlist[ind][0]
        default = self.paramlist[ind][1]

        # get from arguments if possible
        if parname in self.arglist:
            try:
                ret = self.arglist[parnameraw]
            except KeyError:
                ret = self.arglist[parname]
            log.debug('GetArg: from arg list, done (%s=%s)' %
                      (parnameraw, repr(ret)))
            return ret

        # make temporary config entry with lowercase key names
        conftmp = {}
        if self.name in self.config:
            # skip if no step entry in config
            for keyname in self.config[self.name].keys():
                conftmp[keyname.lower()] = self.config[self.name][keyname]

        # get from config if possible
        if parname in conftmp:
            value = conftmp[parname]
            # If default is a sequence:
            if isinstance(default, (tuple, list)):
                # Get type for list elements
                # (if default is empty, convert to string)
                if len(default) > 0:
                    outtype = type(default[0])
                else:
                    outtype = str
                ret = []
                # Convert elements in list
                # Note: if the keyword only has one item in the list and there
                # is no trailing comma, configobj will read it as a string
                # instead of a 1-element list. We force to list here.
                if isinstance(value, str):
                    value = [value]
                for i in range(len(value)):
                    # Check if it's boolean
                    if outtype == bool:
                        if value[i].lower() in booltrue:
                            ret.append(True)
                        else:
                            # default to False
                            ret.append(False)
                    # Not boolean - just convert to type
                    else:
                        ret.append(outtype(value[i]))
                # convert to tuple
                log.debug('GetArg: from config file '
                          '(%s=%s)' %
                          (parname, repr(type(default)(ret))))
                return type(default)(ret)
            else:
                # Default is not a sequence
                # Check if it's boolean
                if isinstance(default, bool) and not \
                        isinstance(value, bool):
                    if value.lower() in booltrue:
                        log.debug('GetArg: from config file '
                                  '(%s=True)' % parname)
                        return True
                    else:
                        log.debug('GetArg: from config file '
                                  '(%s=False)' % parname)
                        return False
                else:
                    # Not boolean - just convert to type
                    log.debug('GetArg: from config file '
                              '(%s=%s)' %
                              (parname, repr(type(default)(value))))
                    return type(default)(value)

        # get default from parameter list
        ret = self.paramlist[ind][1]

        # return parameter
        log.debug('GetArg: from param list '
                  '(%s=%s)' % (parname, repr(ret)))
        return ret

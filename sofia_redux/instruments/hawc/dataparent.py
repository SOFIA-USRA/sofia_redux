# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Data storage parent class."""

from datetime import datetime
import os
import re

from astropy import log
import configobj

__all__ = ['DataParent']


class DataParent(object):
    """
    Pipeline data object.

    This object stores a config file, header, and data.
    """
    # Pipeline version
    pipever = '3.0.0'
    """str : Pipeline version."""

    def __init__(self, config=None):
        """
        Initialize data object and variables.

        Parameters
        ----------
        config : `configobj.ConfigObj`, dict, str, or list of str, optional
            If specified, the configuration will be loaded.
        """
        # set up internal variables
        self.filename = ''
        self.rawname = ''
        self.loaded = False

        # Data Variable:
        self.data = None

        # Header: A dictionary. Lists for HISTORY and COMMENT entries.
        self.header = {}

        # retrieve config directory location from current file path:
        # assumes config is in sofia_redux/instruments/hawc/data/config
        # and this file is in sofia_redux/instruments/hawc
        pipe_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(pipe_path, 'data', '')
        self.config_path = os.path.join(pipe_path, 'data', 'config', '')

        # place holder for pipeline mode
        self.mode = None

        self.config = None
        self.config_files = []
        self.setconfig(config)

    def __getattr__(self, name):
        """
        Get attribute.

        Allows access to these attributes:

            - filenamebegin: filename start
            - filenameend: filename end
            - filenum: file number, extracted from file name

        Each attribute is derived using the regex in the configuration
        parameter of the same name (in the [data] section of the
        configuration file). See config/pipeconf.cfg for an example.

        Parameters
        ----------
        name : str
            The attribute to retrieve.
        """
        # return filenamebegin if it's requested
        if name == 'filenamebegin':
            (fpath, fname) = os.path.split(self.filename)
            # if filenamebegin is specified, use it
            try:
                filenamebegin = self.config['data']['filenamebegin']
                match = re.search(filenamebegin, fname)
            except (KeyError, TypeError):
                match = False
                filenamebegin = "None"
            if match:
                # found file name beginning -> return it
                return os.path.join(fpath, match.group())
            else:
                # assume filename format is name.filestep.fits or ....fits.gz
                msg = "Filename=%s doesn't match pattern=%s" % \
                      (fname, filenamebegin)
                log.warning(msg)
                extloc = fname.rfind('.f')

                if (extloc < 0
                        or fname[extloc:]
                        not in ['.fts', '.fits', '.fits.gz']):
                    log.warning('Filename has non-fits extension')
                    extloc = fname.rfind('.')
                    if extloc < 0:
                        extloc = len(fname)
                    else:
                        # to add the '.'
                        extloc += 1
                else:
                    # to add the '.'
                    extloc += 1

                typeloc = fname[0:extloc - 1].rfind('.')
                if typeloc < 0:
                    typeloc = extloc
                else:
                    typeloc += 1
                return os.path.join(fpath, fname[:typeloc])

        # return filenameend if it's requested
        if name == 'filenameend':
            (fpath, fname) = os.path.split(self.filename)
            # if filenameend is specified, use it
            try:
                filenameend = self.config['data']['filenameend']
                match = re.search(filenameend, fname)
            except (KeyError, TypeError):
                match = False

            if match:
                # found file name end -> return it
                return match.group()
            else:
                extloc = fname.rfind('.f')
                if (extloc < 0
                        or fname[extloc:]
                        not in ['.fts', '.fits', '.fits.gz']):
                    log.warning('Filename has non-fits extension')
                    extloc = fname.rfind('.')
                    if extloc < 0:
                        extloc = len(fname)
                return fname[extloc:]

        # return file number if it's requested
        if name == 'filenum':
            (fpath, fname) = os.path.split(self.filename)
            # if filenum is specified in pipeconf, use it
            try:
                filenum = self.config['data']['filenum']
                match = re.search(filenum, fname)
            except (KeyError, TypeError):
                match = False
            if match:
                # found file num -> return first match
                for m in match.groups():
                    if m is not None:
                        return m
                # single match -- return it
                return match.group()
            else:
                return None

        # raise error if attribute is unknown
        msg = "'%s' object has no attribute '%s'" % \
              (type(self).__name__, name)
        raise AttributeError(msg)

    def default_config(self):
        """
        Set default configuration from config/pipeconf.cfg.

        Returns
        -------
        `configobj.ConfigObj`
        """
        # get master default config file
        default = os.path.join(self.config_path, 'pipeconf.cfg')
        config = configobj.ConfigObj(default)

        self.config_files.append(default)
        return config

    def date_override_config(self, date):
        """
        Retrieve an override configuration for a particular date.

        Parameters
        ----------
        date : datetime
            The date of the observation.

        Returns
        -------
        `configobj.ConfigObj`
            Any non-default parameters associated with the observation date.
        """
        # get overrides for given date if provided
        override_conf = None
        if date is not None:
            date_config = os.path.join(self.config_path, 'date_config.cfg')
            dates = dict((dobj['datetime'], dobj['file'])
                         for dobj in configobj.ConfigObj(date_config).values())
            override = None
            for datekey in sorted(dates.keys()):
                dateobj = datetime.strptime(datekey, "%Y-%m-%dT%H:%M:%S")
                if date < dateobj:
                    override = dates[datekey]
                    break

            if override is not None:
                override = os.path.join(self.config_path, override)
                override_conf = configobj.ConfigObj(override)
                self.config_files.append(override)
                log.debug('Override config file for date %s: %s' %
                          (date, override))
        return override_conf

    def mode_override_config(self, mode):
        r"""
        Retrieve an override configuration for an observation mode.

        Parameters
        ----------
        mode : str
            The pipeline mode to retrieve. Should be specified in the
            config file with 'mode\_' prepended.

        Returns
        -------
        `configobj.ConfigObj`
            Any non-default parameters associated with the observation date.
        """
        # get overrides for given date if provided
        override_conf = None
        if mode is not None and self.config is not None:
            mode_key = "mode_%s" % mode
            if mode_key in self.config:
                override_conf = configobj.ConfigObj(self.config[mode_key])
        return override_conf

    def setconfig(self, config=None, date=None):
        """
        Set configuration for the pipe data.

        The configuration object is returned. The config
        parameter can be one of these:

        - A ConfigObj object
        - A path string containing the filename of a valid config file
        - A list of path strings to valid config files. In this case,
          each file is merged in order.

        A default configuration will be loaded first, then the config
        parameter will be merged into it.

        Parameters
        ----------
        config : `configobj.ConfigObj`, str, or list of str, optional
            Configuration to merge into the default.
        date : datetime, optional
            If specified, additional override configurations will
            be loaded for this date, if they exist, prior to loading
            configurations from `config`.

        Returns
        -------
        `configobj.ConfigObj`
            The merged configuration.
        """
        # first set default config.
        # Anything else provided will be merged onto this config.
        self.config = self.default_config()
        self.mergeconfig(config=config, date=date)

        return self.config

    def mergeconfig(self, config=None, date=None, mode=None):
        r"""
        Merge configuration into the existing configuration.

        All values from the new configuration are used,
        overwriting old values if they are already in the old
        configuration.

        The order is:

        - load the default configuration (data/config/pipeconf.cfg)
        - load any overrides for the observation date
        - load any overrides for the pipeline mode
        - load any user overrides from the config parameter

        Parameters
        ----------
        config : `configobj.ConfigObj`, dict, str or `list` of str, optional
            Configuration to merge into the default.
        date : datetime, optional
            If specified, additional override configurations will
            be loaded for this date, if they exist, prior to loading
            configurations from `config`.
        mode : str, optional
            The pipeline mode to retrieve. Should be specified in the
            config file with 'mode\_' prepended.
        """
        # If there is no existing config, call setconfig first
        if self.config is None:
            self.setconfig(config, date)
            return

        # Then set any date-specific overrides
        if date is not None:
            override = self.date_override_config(date)
            if override is not None:
                self.config.merge(override)
            else:
                log.debug('No date config file for %s' % date)

        # Then set any mode-specific overrides
        if mode is not None:
            override = self.mode_override_config(mode)
            if override is not None:
                self.config.merge(override)

        # return if nothing else to do
        if config is None:
            return

        if isinstance(config, configobj.ConfigObj) or isinstance(config, dict):
            # if config is a ConfObj or dict, merge it
            self.config.merge(config)
            if hasattr(config, 'filename'):
                self.config_files.append(config.filename)
                log.debug('User config file: %s' % config.filename)

        elif isinstance(config, str):
            # if config is a string - check for file existence -> load it
            if not os.path.isfile(config):
                config = os.path.join(self.config_path, config)
            if os.path.isfile(config):
                config = os.path.abspath(config)
                try:
                    user_config = configobj.ConfigObj(config)
                    self.config.merge(user_config)
                    self.config_files.append(config)
                    log.debug('User config file: %s' % config)

                except configobj.ConfigObjError as error:
                    msg = 'Error while loading configuration file'
                    log.error('SetConfig: ' + msg)
                    raise error
            else:
                msg = '<%s> is invalid file name for configuration' % config
                log.error('SetConfig: ' + msg)
                raise IOError(msg)

        elif isinstance(config, list):
            # merge each one in the order provided
            for conf in config:
                self.mergeconfig(conf)
        else:
            raise TypeError('Unexpected type for new configuration file.')

    def get_pipe_mode(self):
        """
        Get the pipeline mode.

        Searches for an appropriate pipeline mode in the config file, given
        the header values in the passed data. Tries to mach all key=value
        pairs in the datakeys value of the mode entries in the config
        file. Returns name of the first pipeline mode that matches
        the data. Returns None if no matching pipeline mode found.

        Returns
        -------
        str or None
            The pipeline mode name, or None if not found.
        """
        if self.config is None:
            return None

        for section in self.config.sections:
            if section.startswith('mode_'):
                # Get the datakeys and make list of lists with
                # format [ [key, val], [key, val], [key,val] ]
                try:
                    datakeys = self.config[section]['datakeys'].split('|')
                except KeyError:
                    log.warning("In configuration, missing"
                                " datakeys for mode=%s" % section)
                    continue
                datakeys = [dk.split('=') for dk in datakeys]

                # Check all keywords in the file
                check = True
                for dk in datakeys:
                    try:
                        value = self.getheadval(dk[0].strip(), errmsg=False)
                        if str(value).upper().strip() != dk[1].upper().strip():
                            check = False
                    except KeyError:
                        check = False
                if check:
                    log.debug('GetPipeMode: Found mode=%s' % section[5:])
                    # return mode name w/o 'mode_'
                    return section[5:]
        return None

    def load(self, filename=''):
        """
        Load the data from the file.

        This function is not implemented for the parent class. It should
        be overridden by child classes.
        """
        # raise error -- this should not be called
        raise NotImplementedError("No default load function for data parent.")

    def save(self, filename=''):
        """
        Save the data in the object to the specified file.

        This function is not implemented for the parent class. It should
        be overridden by child classes.
        """
        # raise error -- this should not be called
        raise NotImplementedError("No default save function for data parent.")

    def copy(self):
        """
        Return a copy of the current object.

        Returns
        -------
        DataParent
        """
        # create new object
        out = DataParent(config=self.config)

        # copy filename and header
        out.filename = self.filename
        out.rawname = self.rawname
        out.loaded = self.loaded
        out.header = self.header.copy()

        # Copy data - backup if no copy() available
        try:
            out.data = self.data.copy()
        except AttributeError:
            out.data = self.data

        # return message and new object
        return out

    def mergehead(self, other):
        """
        Merge a data object header into the current object's header.

        Parameters
        ----------
        other : DataParent
            The other object.
        """
        # get selfhist and otherhist lists
        if 'HISTORY' in self.header:
            selfhist = self.header['HISTORY']
        else:
            selfhist = []

        if 'HISTORY' in other.header:
            otherhist = other.header['HISTORY']
        else:
            otherhist = []

        # add history keywords (no duplicates)
        selfhist += [hist for hist in otherhist if hist not in selfhist]

        # if there is something add write back to header
        if len(selfhist):
            self.header['HISTORY'] = selfhist

        # get selfcomm and othercomm lists
        if 'COMMENT' in self.header:
            selfcomm = self.header['COMMENT']
        else:
            selfcomm = []

        if 'COMMENT' in other.header:
            othercomm = other.header['COMMENT']
        else:
            othercomm = []

        # add comment keywords (no duplicates)
        selfcomm += [comm for comm in othercomm
                     if comm not in selfcomm]

        # if there is something add write back to header
        if len(selfcomm):
            self.header['COMMENT'] = selfcomm

        # Go through keywords listed in headmerge: assume self is first
        headmerge = self.config['headmerge']
        for key in headmerge.keys():
            if key in self.header and key in other.header:
                selfval = self.header[key]
                otherval = other.header[key]
                operation = headmerge[key].upper()
                if operation == 'LAST':
                    selfval = otherval
                elif operation == 'MIN':
                    selfval = min(selfval, otherval)
                elif operation == 'MAX':
                    selfval = max(selfval, otherval)
                elif operation == 'SUM':
                    selfval += otherval
                elif operation == 'OR':
                    selfval = selfval | otherval
                elif operation == 'AND':
                    selfval = selfval & otherval
                elif operation == 'CONCATENATE':
                    if ',' in str(selfval):
                        vlist = str(selfval).split(',')
                    else:
                        vlist = [str(selfval)]
                    if ',' in str(otherval):
                        olist = str(otherval).split(',')
                    else:
                        olist = [str(otherval)]
                    for oval in olist:
                        if oval not in vlist:
                            vlist.append(oval)
                    selfval = ','.join(sorted(vlist))
                elif operation == 'DEFAULT':
                    if type(selfval) is str:
                        selfval = 'UNKNOWN'
                    elif type(selfval) is int:
                        selfval = -9999
                    elif type(selfval) is float:
                        selfval = -9999.0
                self.header[key] = selfval

    def getheadval(self, key, errmsg=True):
        """
        Get header value.

        Returns the value of the requested key from the header.

        If the key is present in the [header] section
        of the configuration, that value is returned instead.
        The following entries are possible in the configuration file:

        - KEY = VALUE : VALUE is returned. The system checks
          if value is an int or a float, else a string is returned.
        - KEY = NEWKEY : The value under header[NEWKEY] is returned.
        - KEY = ?_ALTKEY : If the keyword KEY is present, header[KEY] is
          returned, else header[ALTKEY] is returned.

        If the key can not be found in either the header or the configuration,
        a KeyError is produced and a warning is issued.

        Parameters
        ----------
        key : str
            The keyword value to return.
        errmsg : bool, optional
            Flag indicating if a log error message should be
            issued if the keyword is not found. A KeyError will still
            be raised if errmsg is False.

        Returns
        -------
        int, float, or str
            The header value.

        Raises
        ------
        KeyError
            If the keyword is not found.
        """
        val = None
        # Look in the config
        try:
            # get the value
            val = self.config['header'][key]
            # Check if it's optional header replacement i.e. starts with '?_'
            if val[:2] in ['?_', '? ', '?-']:
                # if key is not in the header ->
                # use key name under value instead
                if key not in self.header:
                    key = val[2:].upper()
                val = None
            # Check if it's a Header replacement (but not T/F)
            elif val[0].isalpha() and \
                    val[:2] not in ['T ', 'F '] and \
                    val not in ['T', 'F']:
                log.info('Getheadval: Using %s value for %s' %
                         (val.upper(), key))
                key = val.upper()
                val = None
            # Else: read value
            else:
                # Try as T / F
                found = True
                if val == 'T' or val[:2] == 'T ':
                    val = True
                elif val == 'F' or val[:2] == 'F ':
                    val = False
                else:
                    found = False
                # Try as int
                if not found:
                    try:
                        val = int(val)
                        found = True
                    except ValueError:
                        pass
                # Try as float
                if not found:
                    try:
                        val = float(val)
                    except ValueError:
                        pass

                # If not found - just leave value as string
                # update value in header
                self.setheadval(key, val)

        except KeyError:
            # if key is not in config - continue
            pass
        except TypeError:
            # if config is not yet loaded - issue message only
            log.debug('GetHeadVal: Missing Configuration')

        # Look in the header
        if val is None:
            # get value from header
            try:
                val = self.header[key]
            except KeyError:
                # if keyword is not found
                msg = 'Missing %s keyword in header' % key
                if errmsg:
                    log.error('GetHeadVal: %s' % msg)
                raise KeyError(msg)

        return val

    def setheadval(self, key, value, comment=''):
        """
        Set a keyword value in the header.

        Parameters
        ----------
        key : str
            The keyword to set.
        value : str, int, float, or bool
            The value to set.
        comment : str, optional
            If provided, will be set in the value of the COMMENT
            keyword in the header.
        """
        # If key is HISTORY or COMMENT: add to list
        if key == 'HISTORY' or key == 'COMMENT':
            if key in self.header:
                self.header[key].append(value)
            else:
                self.header[key] = [value, ]
        else:
            # otherwise add as normal keyword
            self.header[key] = value
            if len(comment) > 0:
                self.setheadval('COMMENT', '%s, %s' % (key, comment))

    def delheadval(self, key):
        """
        Delete one or more keywords from the header.

        Keywords are deleted from self.header, which defaults to the
        first header in the data object.

        If the keyword is HISTORY or COMMENT, then all HISTORY or COMMENT
        entries will be removed.

        Parameters
        ----------
        key : str or list of str
            The header keyword(s) to delete.
        """
        # If key is a list, remove all entries
        if isinstance(key, (list, tuple)):
            for k in key:
                self.delheadval(k)
        # Else if it's a string delete the key - ignore any KeyError
        else:
            if key in self.header:
                del(self.header[key])

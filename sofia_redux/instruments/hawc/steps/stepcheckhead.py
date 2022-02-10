# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Header validation pipeline step."""

import os
import re

from astropy import log
import configobj

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepCheckhead', 'HeaderValidationError']


# Define a custom error value so that wrapper scripts can catch
# them and abort with a readable message
class HeaderValidationError(RuntimeError):
    """Error raised when a FITS header does not meet requirements."""
    pass


class StepCheckhead(StepParent):
    """
    Validate headers for HAWC+ raw data files.

    This step checks the primary header of the input file for
    keywords required for data reduction. It also reformats the
    filename stored in the DataFits object, to conform to SOFIA
    requirements.

    This step should be called before any other steps, on raw
    HAWC data. Output from this step is identical to the input
    except for the filename; it should not be saved to disk.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'checkhead', and are named with
        the step abbreviation 'CHK'.

        Parameters defined for this step are:

        abort : bool
            If set, this step will raise a `HeaderValidationError`
            if the input header does not meet requirements. Otherwise,
            it will only issue warnings.
        headerdef : str
            Path to the header keyword definition file, usually
            stored in data/config/header_req_config.cfg.
        """

        # Name of the pipeline reduction step
        self.name = 'checkhead'
        self.description = 'Check Headers'

        # Identifier for saved file names.
        self.procname = 'chk'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['abort', True,
                               'Abort pipeline if headers do not meet '
                               'requirements'])
        self.paramlist.append(['headerdef', 'headerdef.txt',
                               'Header keyword definition file'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Read required keyword types and limits from a configuration file
           (usually in pipeline/config/header_req_config.cfg).
        2. Check the primary header of the input file for compliance and
           output error messages if necessary.
        3. Rename the output file to SOFIA standard format.

        Raises
        ------
        HeaderValidationError
            If any errors are found and the parameter 'abort' is True.
        """

        # Read input parameters
        abort = self.getarg('abort')
        deffile = os.path.expandvars(self.getarg('headerdef'))

        # Read header definition file into config object
        if os.path.isfile(deffile):
            try:
                reqconf = configobj.ConfigObj(deffile)
            except configobj.ConfigObjError as error:
                msg = 'Error while loading header configuration file'
                log.error('HeaderCheck: ' + msg)
                raise error
        else:
            msg = '<%s> is invalid file name for header configuration' % \
                  deffile
            log.error('HeaderCheck: ' + msg)
            raise IOError(msg)

        # Check a few important mode keywords
        chopping = self._getsafeval('CHOPPING')
        nodding = self._getsafeval('NODDING')
        dithering = self._getsafeval('DITHER')
        scanning = self._getsafeval('SCANNING')

        # Add any that are True to the requirement set
        req_set = ['*']
        if chopping is not None and chopping:
            req_set.append('chopping')
        if nodding is not None and nodding:
            req_set.append('nodding')
        if dithering is not None and dithering:
            req_set.append('dithering')
        if scanning is not None and scanning:
            req_set.append('scanning')

        # Flag to throw error if requirements are not met
        abort_flag = False

        # Loop through keywords, checking against requirements
        reqdict = reqconf.dict()
        for key, req in reqdict.items():
            # Retrieve requirements
            try:
                req_category = str(req['requirement']).strip()
            except KeyError:
                req_category = '*'
            try:
                req_dtype = str(req['dtype']).strip()
            except KeyError:
                req_dtype = 'str'
            try:
                req_drange = req['drange']
            except KeyError:
                req_drange = None

            # Get type class corresponding to string
            if req_dtype == 'bool':
                req_dtype_class = bool
            elif req_dtype == 'int':
                req_dtype_class = int
            elif req_dtype == 'long':
                req_dtype_class = int
            elif req_dtype == 'float':
                req_dtype_class = float
            else:
                req_dtype_class = str

            # Check if key is required for this data type
            if req_category not in req_set:
                continue

            # Retrieve value from header and/or config file
            val = self._getsafeval(key)
            valtype = type(val)
            stype = valtype.__name__

            # Check if required key is present
            if val is None:
                abort_flag = True
                msg = 'Required keyword <%s> not found' % key
                if abort:
                    log.error(msg)
                else:
                    log.warning(msg)
                continue

            # Check if key matches required type
            if req_dtype == 'str' or req_dtype == 'bool':
                # Use exact type for str, bool
                if stype != req_dtype:
                    abort_flag = True
                    msg = 'Required keyword <%s> has wrong ' \
                          'type <%s>; should be <%s>' % \
                          (key, stype, req_dtype)
                    if abort:
                        log.error(msg)
                    else:
                        log.warning(msg)
                    continue
            elif req_dtype == 'float':
                # Allow any number type for float types
                if stype not in ['float', 'int', 'long']:
                    abort_flag = True
                    msg = 'Required keyword <%s> has wrong ' \
                          'type <%s>; should be <%s>' % \
                          (key, stype, req_dtype)
                    if abort:
                        log.error(msg)
                    else:
                        log.warning(msg)
                    continue
            elif req_dtype == 'int' or req_dtype == 'long':
                # Allow ints to be longs and vice versa
                if stype not in ['int', 'long']:
                    abort_flag = True
                    msg = 'Required keyword <%s> has wrong ' \
                          'type <%s>; should be <%s>' % \
                          (key, stype, req_dtype)
                    if abort:
                        log.error(msg)
                    else:
                        log.warning(msg)
                    continue

            # Check if value meets range requirements
            if req_drange is not None:

                # Check for enum first -- ignore any others if
                # present. May be used for strings, bools, or numerical
                # equality.
                if 'enum' in req_drange:
                    enum = req_drange['enum']

                    # Make into list if enum is a single value
                    if type(enum) is not list:
                        enum = [enum]

                    # Cast to data type
                    if req_dtype == 'bool':
                        enum = [True if str(e).strip().lower() == 'true'
                                else False
                                for e in enum]
                    else:
                        try:
                            enum = [req_dtype_class(e) for e in enum]
                        except ValueError as error:
                            msg = 'Error in header configuration file for ' \
                                  'key <%s>' % key
                            log.error('HeaderCheck: ' + msg)
                            raise error

                    # Case-insensitive comparison for strings
                    if stype == 'str':
                        enum = [str(e).upper() for e in enum]
                        if val.upper() not in enum:
                            abort_flag = True
                            msg = 'Required keyword <%s> has wrong ' \
                                  'value <%s>; should be in %s' % \
                                  (key, val, enum)
                            if abort:
                                log.error(msg)
                            else:
                                log.warning(msg)
                            continue
                    else:
                        if val not in enum:
                            abort_flag = True
                            msg = 'Required keyword <%s> has wrong ' \
                                  'value <%s>; should be in %s' % \
                                  (key, val, enum)
                            if abort:
                                log.error(msg)
                            else:
                                log.warning(msg)
                            continue

                # Check for a minimum requirement
                # (numerical value must be >= minimum)
                else:
                    if ('min' in req_drange
                            and stype in ['int', 'long', 'float']):
                        try:
                            minval = req_dtype_class(req_drange['min'])
                        except ValueError as error:
                            msg = 'Error in header configuration file for ' \
                                  'key <%s>' % key
                            log.error('HeaderCheck: ' + msg)
                            raise error
                        if val < minval:
                            abort_flag = True
                            msg = 'Required keyword <%s> has wrong ' \
                                  'value <%s>; should be >= %f' % \
                                  (key, val, minval)
                            if abort:
                                log.error(msg)
                            else:
                                log.warning(msg)
                            continue

                    # Check for a maximum requirement
                    # (numerical value must be <= maximum)
                    if ('max' in req_drange
                            and stype in ['int', 'long', 'float']):
                        try:
                            maxval = req_dtype_class(req_drange['max'])
                        except ValueError as error:
                            msg = 'Error in header configuration file for ' \
                                  'key <%s>' % key
                            log.error('HeaderCheck: ' + msg)
                            raise error
                        if val > maxval:
                            abort_flag = True
                            msg = 'Required keyword <%s> has wrong ' \
                                  'value <%s>; should be <= %f' % \
                                  (key, val, maxval)
                            if abort:
                                log.error(msg)
                            else:
                                log.warning(msg)
                            continue

        # Bail if requested, and requirements are not met
        if abort and abort_flag:
            msg = 'Header for <%s> does not meet requirements for ' \
                  'data processing' % \
                  os.path.basename(self.datain.filename)
            log.error('HeaderCheck: ' + msg)
            raise HeaderValidationError(msg)

        # Add a reference to the input data in dataout
        self.dataout = self.datain

        # Rename the output file for SOFIA convention

        # Keep the original filename
        (dname, rawname) = os.path.split(self.dataout.filename)
        self.dataout.setheadval('RAWFNAME', rawname, 'Raw filename')

        # Get flight number from mission
        mid = str(self._getsafeval('MISSN-ID'))
        match = re.search(r'F(\d{3,4})', mid)
        if match is not None and len(match.groups()) > 0:
            fltnum = "%4.4d" % int(match.group(1))
        else:
            # if can't find, use HAWC flight number
            try:
                fltnum = "%4.4d" % self._getsafeval('FLGTNUM')
            except (TypeError, ValueError):
                fltnum = "XXXX"

        # Get spectels
        spec1 = self._getsafeval('SPECTEL1')
        spec2 = self._getsafeval('SPECTEL2')
        if spec1 is None:
            spec1 = 'UNKNOWN'
        if spec2 is None:
            spec2 = 'UNKNOWN'
        if spec1 == 'UNKNOWN' and spec2 == 'UNKNOWN':
            spec = 'UNKNOWN'
        else:
            spec2 = re.sub('^HAW_', '', spec2)
            spec = re.sub('_', '', spec1.strip()) + \
                re.sub('_', '', spec2.strip())

        # Get AOR-ID
        aorid = self._getsafeval('AOR_ID')
        if aorid is None:
            aorid = 'UNKNOWN'
        else:
            aorid = re.sub('_', '', aorid.strip())

        # Get obsmode
        instcfg = str(self._getsafeval('INSTCFG')).upper()
        calmode = str(self._getsafeval('CALMODE')).upper()
        if calmode not in ['NONE', 'UNKNOWN']:
            obs = 'CAL'
        else:
            if 'POL' in instcfg:
                obs = 'POL'
            else:
                obs = 'IMA'

        # Get file number from filename
        fnum = self.dataout.filenum
        try:
            int(fnum)
        except (TypeError, ValueError):
            fnum = 'UNKNOWN'

        # Compose output file name
        outfilename = "F%s_HA_%s_%s_%s_RAW_%s.fits" % \
                      (fltnum, obs, aorid, spec, fnum)
        self.dataout.filename = os.path.join(dname, outfilename)

    def _getsafeval(self, key):
        """
        Helper function to quietly return None if a keyword isn't found.

        Parameters
        ----------
        key : str
            The keyword value to retrieve.
        """
        # This function assumes that self.datain contains exactly
        # one pre-loaded file

        # Set the log level to critical only
        old_level = log.level
        log.setLevel('CRITICAL')

        # Try to get the value from the datain header, allowing
        # the config file to override
        try:
            val = self.datain.getheadval(key)
        except KeyError:
            # Set to None if not found
            val = None

        # Strip any spaces from string values
        if type(val) is str:
            val = val.strip()

        # Restore the old log level
        log.setLevel(old_level)

        # Return the safe value
        return val

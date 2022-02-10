# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pipeline step that loads auxiliary data."""

import os
import glob
import re
import time

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.datatext import DataText
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepLoadAux']


class StepLoadAux(StepParent):
    """
    Pipeline step parent class with auxiliary data support.

    This class provides a pipe step with a simple mechanism to search
    auxiliary files and match them to input data. An initial list of
    potential aux files is determined from a path/filename string
    (a glob); the final file(s) is determined by matching header
    keywords between the auxiliary files and the step input data.

    This object does not have a conventional setup function
    since it is intended to be inherited by other steps. Steps that
    inherit from this class should call the loadauxsetup function explicitly
    inside their setup functions, in order to set up the auxiliary data
    parameters.
    """
    def __init__(self):
        # placeholder for auxiliary parameter name
        self.auxpar = None

        super().__init__()

    def loadauxsetup(self, auxpar='aux'):
        """
        Load parameters for auxiliary file identification.

         This function should be called within the setup function
         for a pipeline step, after self.paramlist has been initialized.

         Parameters
         ----------
         auxpar : str, optional
             The auxiliary file type. This string will appear in the
             names of the parameters added to the step.
        """
        if self.config is None:
            self.config = {}

        # Set name of the auxfile parameter
        self.auxpar = auxpar

        # Append parameters
        self.paramlist.append([auxpar + 'file', '%sfolder/*.fits' % auxpar,
                               'Filename for auxiliary file(s). '
                               'Can contain * and ? '
                               'wildcards to match multiple files '
                               'to be selected using fitkeys '
                               '(default = %sfolder/*.fits)' % auxpar])
        self.paramlist.append(['bkup' + auxpar, 'bkup%sfolder/*.fits' % auxpar,
                               'Back up filename for auxiliary '
                               'file(s). Can contain * and ? '
                               'wildcards to match multiple files '
                               'to be selected using fitkeys '
                               '(default = bkup%sfolder/*.fits)' % auxpar])
        self.paramlist.append([auxpar + 'fitkeys', [],
                               'List of header keys that need to match '
                               'auxiliary data file '
                               '(default = []) - only used if '
                               'multiple files match %sfile' % auxpar])
        self.paramlist.append(['daterange', 1.0,
                               'If DATE-OBS is in fitkeys, files are '
                               'matched within this many days.'])

    def getdataobj(self, filename):
        """
        Retrieve the DataParent child appropriate to the data.

        Checks the filename to determine if it is likely to be a FITS
        file. If not, data is assumed to be text.

        The returned object is initialized with self.config, but data
        from the input file is not loaded.

        Parameters
        ----------
        filename : str
            Path to the file to test.

        Returns
        -------
        DataFits or DataText
            The data object. Data is not loaded from the filename.
        """
        if re.search(DataFits.filenamefit, filename):
            data = DataFits(config=self.config)
        else:
            data = DataText(config=self.config)
        return data

    def loadauxname(self, auxpar='', data=None, multi=False):
        """
        Search for files matching auxfile.

        If only one match is found, that file is returned. Otherwise,
        the header keywords listed in auxfitkeys are matched between the
        data and any auxfiles found. The first auxfile for which these
        keyword values match the data values is selected. The filename
        of the best match is returned.

        Parameters
        ----------
        auxpar : str, optional
            A name for the aux file parameter to use. This
            allows loadauxfiles to be used multiple times
            in a given pipe step (for example for darks and
            flats). Default value is self.auxpar which is set
            by loadauxsetup().
        data : DataFits or DataText, optional
            A data object to match the auxiliary file to.
            If no data is specified, self.datain is used (for
            Multi Input steps self.datain[0]).
        multi : bool, optional
            If set, a list of file names is returned instead of a single
            file name.

        Returns
        -------
        str or list of str
            The matching auxiliary file(s).
        """
        # Set auxpar
        if len(auxpar) == 0:
            auxpar = self.auxpar

        # Get parameters
        auxfile = os.path.expandvars(self.getarg(auxpar + 'file'))
        return self._loadauxnamefile(auxfile, auxpar, data, multi)

    def _loadauxnamefile(self, auxfile, auxpar, data, multi, backup=True):
        # This function is now separated out in order to allow sub-classes
        # to manipulate the auxfile parameter before loading.

        fitkeys = self.getarg(auxpar + 'fitkeys')
        if len(fitkeys) == 1 and len(fitkeys[0]) == 0:
            fitkeys = []

        # Look for files - return in special cases

        # Glob the list of files
        auxlist = sorted(glob.glob(auxfile))

        # Throw exception if no file found
        if len(auxlist) < 1:
            if backup:
                log.warning('No files found under %s - '
                            'looking in backup' % auxfile)
                auxfile = os.path.expandvars(self.getarg('bkup' + auxpar))
                auxlist = sorted(glob.glob(auxfile))
            if len(auxlist) < 1:
                msg = 'No %s files found under %s' % (auxpar, auxfile)
                log.error(msg)
                raise ValueError(msg)

        # Get datain object (depends on step being SingleInput or MultiInput)
        if data is None:
            if isinstance(self.datain, list):
                data = self.datain[0]
            else:
                data = self.datain

        # Return unique file, or all files if fitkeys is empty
        if len(auxlist) == 1 or len(fitkeys) == 0:
            if len(auxlist) == 1:
                log.info(f'LoadAuxName: Found unique '
                         f'file = {auxlist[0]}')
            elif multi:
                log.info(f'LoadAuxName: No fitkeys: return all '
                         f'{auxpar}files = {auxlist}')
            else:
                log.info('LoadAuxName: No fitkeys: Return '
                         'first %sfile match = %s' %
                         (self.auxpar, auxlist[0]))
            data.setheadval('HISTORY', '%s: Best %sfile = %s' %
                            (self.name, self.auxpar,
                             os.path.split(auxlist[0])[1], ))
            if multi:
                return auxlist
            else:
                return auxlist[0]

        # Select files with Fitkeys

        # check format (make first element uppercase)
        try:
            fitkeys[0].upper()
        except AttributeError:
            # AttributeError if it's not a string
            log.error('LoadAuxFile: fitkeys config parameter is '
                      'incorrect format - need list of strings')
            raise TypeError('fitkeys config parameter is incorrect format'
                            ' - need list of strings')

        # Load all headers from auxlist into a auxheadlist (pipedata objects)
        auxheadlist = []
        for auxnam in auxlist:
            auxdata = self.getdataobj(auxnam)
            auxdata.loadhead(auxnam)
            auxheadlist.append(auxdata)

        # Look through keywords, only keep auxfiles which fit keys
        newheadlist = []
        for key in fitkeys:
            newheadlist = []

            # Look through auxfiles, transfer good ones
            if key == 'DATE-OBS':
                # SPECIAL CASE DATE-OBS:
                # get time for data
                datime = time.mktime(time.strptime(data.getheadval('DATE-OBS'),
                                                   '%Y-%m-%dT%H:%M:%S.%f'))
                # get time offset (from data) for each auxfile
                auxtimes = []
                for auxhead in auxheadlist:
                    auxtime = time.mktime(
                        time.strptime(auxhead.getheadval('DATE-OBS'),
                                      '%Y-%m-%dT%H:%M:%S.%f'))
                    auxtimes.append(abs(auxtime - datime))

                # only keep auxfiles which are within
                # daterange of closest auxfile
                mindiff = min(auxtimes)
                timerange = self.getarg('daterange') * 86400
                sort_times = []
                for auxi in range(len(auxheadlist)):
                    if auxtimes[auxi] - mindiff < timerange:
                        newheadlist.append(auxheadlist[auxi])
                        sort_times.append(auxtimes[auxi])
                if len(newheadlist) > 0:
                    newheadlist = np.array(newheadlist)[np.argsort(sort_times)]

            else:
                # Normal Keyword compare
                for auxhead in auxheadlist:
                    # Check if the auxfile fits (compare with data)
                    if auxhead.getheadval(key) == data.getheadval(key):
                        # it fits -> add to newheadlist
                        newheadlist.append(auxhead)

            # break key loop if no files left
            if len(newheadlist) == 0:
                break
            else:
                auxheadlist = newheadlist

        # Select file to return
        if multi:
            # Return all filenames
            auxname = [aux.filename for aux in auxheadlist]

            # Return message
            listnames = ', '.join(auxname)
            if len(newheadlist) > 0:
                log.info('LoadAuxName: Matching %s found are <%s>' %
                         (auxpar, listnames))
            else:
                log.warning('LoadAuxName: NO MATCH finding aux files')
                log.warning('Returning files <%s>' % listnames)
        else:
            # Return first filename
            auxname = auxheadlist[0].filename
            # Select best file
            if len(newheadlist) > 0:
                log.info('LoadAuxName: Matching %s found is <%s>' %
                         (auxpar, auxname))
            else:
                log.warning('LoadAuxName: NO MATCH finding aux file')
                log.warning('Returning first file <%s>' % auxname)
            listnames = auxname
        data.setheadval('HISTORY', '%s: Best %s = %s' %
                        (self.name, auxpar, listnames))
        # Return selected file
        return auxname

    def loadauxfile(self, auxpar='', data=None, multi=False):
        """
        Load an auxiliary file into a pipeline data object.

        Uses `loadauxname` to search for files matching auxfile.

        auxpar : str, optional
            A name for the aux file parameter to use. This
            allows loadauxfiles to be used multiple times
            in a given pipe step (for example for darks and
            flats). Default value is self.auxpar which is set
            by loadauxsetup().
        data : DataFits or DataText, optional
            A data object to match the auxiliary file to.
            If no data is specified, self.datain is used (for
            Multi Input steps self.datain[0]).
        multi : bool, optional
            If set, a list of file names is returned instead of a single
            file name.

        Returns
        -------
        DataFits or DataText, or a list of DataFits or DataText
            A data object with the best match is returned.
        """
        # Get auxname
        auxname = self.loadauxname(auxpar, data, multi)

        # Load auxdata
        if multi:
            auxdata = []
            for auxnam in auxname:
                data = self.getdataobj(auxnam)
                data.load(auxnam)
                auxdata.append(data)
        else:
            auxdata = self.getdataobj(auxname)
            auxdata.load(auxname)

        # Return selected file
        return auxdata

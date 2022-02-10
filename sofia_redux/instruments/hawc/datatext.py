# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Data storage class for text-based data."""

import os
import re
import time

from astropy import log

from sofia_redux.instruments.hawc.dataparent import DataParent

__all__ = ['DataText']


class DataText(DataParent):
    """
    Pipeline data text object.

    The data is stored as a list of strings, with an optional
    header.
    """

    # File Name Fit: Regexp expression that fits valid filenames
    # fill files .txt and end of name
    filenamefit = r'\.txt\Z'
    """str : Regular expression that matches valid text file names."""

    def __init__(self, filename='', config=None):
        """
        Initialize data object and variables.

        Parameters
        ----------
        filename : str, optional
            Path to a text file.  If specified, data will be loaded
            immediately from the file.
        config : `configobj.ConfigObj`, dict, str, or list of str, optional
            If specified, the configuration will be loaded.
        """

        # Run Parent Constructor
        # loads config, set up header
        super().__init__(config=config)

        # Match for header: a '#' at start of line followed
        # by space or end of line
        self.headermatch = r'\A#(\Z|\s)'

        # Match for header keyword: Alphanumeric
        # values (with spaces) followed by
        # ':' or '=' and any number of spaces
        self.keymatch = r'\A#[A-Za-z0-9\s_-]{0,20}(:|=)\s*'

        # if file exists, load it
        if os.path.exists(filename):
            self.load(filename)
        elif filename != '':
            # user specified a non-existent filename
            msg = 'No such file %s' % filename
            log.error(msg)
            raise IOError(msg)

    def sethead(self, line):
        """
        Add a text line to the header.

        Header keywords in the line are identified by the
        self.keymatch regular expression.

        Parameters
        ----------
        line : str
            The header line to add.
        """
        # Identify keywords
        match = re.search(self.keymatch, line)
        if match:
            # Get key
            key = match.group()

            # Get value (rest of line after match)
            value = line[len(key):].strip()

            # remove header match and clean up key
            match = re.search(self.headermatch, key)
            key = key[len(match.group()):].strip()[:-1].strip()

            # Set header keyword
            self.setheadval(key, value)
        else:
            # Not a key, it's a comment
            comment = line

            # remove header match and clean up comment
            match = re.search(self.headermatch, comment)
            if not match:
                # bad header line
                log.warning('Bad header line: {}'.format(comment))
                return
            comment = comment[len(match.group()):].strip()

            # Set header keyword
            self.setheadval('COMMENT', comment)

    def loadhead(self, filename=''):
        """
        Load the header for text file given.

        Header lines are identified by the self.headermatch
        regular expression.

        Parameters
        ----------
        filename : str, optional
            The path to the file to load.  If not specified,
            self.filename is used.
        """
        # set self.filename and filename
        if len(filename) > 0:
            self.filename = filename
            self.rawname = filename
        else:
            filename = self.filename
        # Open file
        with open(filename, 'rt') as inf:
            # Load header
            for line in inf:
                # identify header lines
                match = re.search(self.headermatch, line)
                if match:
                    self.sethead(line)

    def load(self, filename=''):
        """
        Load the data and header from a given text file.

        After loading, self.loaded is set to True.

        Parameters
        ----------
        filename : str, optional
            The path to the file to load.
        """
        # set self.filename and filename
        if len(filename) > 0:
            self.filename = filename
            self.rawname = filename
        else:
            filename = self.filename

        # Open file
        with open(filename, 'rt') as inf:
            # Load header and data
            self.data = []
            for line in inf:
                # identify header lines
                match = re.search(self.headermatch, line)
                if match:
                    self.sethead(line)
                else:
                    self.data.append(line.strip())
        self.loaded = True

    def save(self, filename=''):
        """
        Save the data to the specified file.

        Existing files are overwritten.

        Parameters
        ----------
        filename : str, optional
            The path to the file to load.
        """
        # get file name
        if not filename:
            filename = self.filename

        # update pipeline keywords
        self.setheadval('Pipeline Version',
                        'Pipe v' + DataParent.pipever.replace('.', '_'))
        self.setheadval('File Name', os.path.split(filename)[-1])
        self.setheadval('File Date', time.strftime('%Y-%m-%dT%H:%M:%S'))

        # Open file
        with open(filename, 'wt') as outf:
            # Save header
            for key in self.header:
                if 'COMMENT' in key:
                    for comm in self.header['COMMENT']:
                        outf.write('# %s\n' % comm)
                elif 'HISTORY' in key:
                    for hist in self.header['HISTORY']:
                        outf.write('# HISTORY: %s\n' % hist)
                else:
                    outf.write('# %s: %s\n' % (key, self.header[key]))

            # Save data
            for line in self.data:
                outf.write('%s\n' % line)

        log.debug('Save: saved text file %s' % filename)

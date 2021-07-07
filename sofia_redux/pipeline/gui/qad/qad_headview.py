# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FITS header viewing widget."""

import os
from sofia_redux.pipeline.gui.textview import TextView


class HeaderViewer(TextView):
    """View, find, and filter text from FITS headers."""
    def __init__(self, parent=None):
        """
        Initialize the header viewer widget.

        Parameters
        ----------
        parent : `QWidget`
            Parent widget.
        """
        super().__init__(parent)

        self.tableButton.setEnabled(True)
        self.tableButton.clicked.connect(self.table)

    def load(self, header):
        """
        Load FITS headers into TextEdit widget.

        Parameters
        ----------
        header : dict
            Keys are file paths; values are lists of FITS header objects
            (`astropy.io.fits.Header`) from each relevant FITS extension.
        """
        # header is a dictionary of FITS header objects, keyed
        # by the file path
        self.text = header

        # format to html
        self.html = self.format()

        # set in text window
        self.textEdit.setHtml(self.html)
        self.textEdit.repaint()

    def format(self):
        """Format header text into HTML for display."""
        # some useful strings
        anchor = '<a name="anchor"></a>'
        br = '<br>'
        jstr = anchor + br + anchor + '-' * 80 + br

        # format headers as strings
        header_strs = []
        sortkeys = sorted(self.text.keys())
        for fpath in sortkeys:
            hlist = self.text[fpath]
            if type(hlist) is not list:
                hlist = [hlist]
            for i, hdr in enumerate(hlist):
                if 'EXTNAME' in hdr:
                    extname = "Extension: {}".format(hdr['EXTNAME'])
                else:
                    extname = "Extension: {}".format(i)

                # convert the whole header to a string
                hdr = hdr.tostring(sep=br)

                # add the filename to separate different headers
                # add the anchor tag to keep the separation when filtering
                extname = anchor + br + anchor + extname + \
                    anchor + br + anchor + '-' * len(extname)
                if i == 0:
                    start = jstr + anchor + os.path.basename(fpath) + jstr
                else:
                    start = ''
                hdr = start + extname + br + hdr

                header_strs.append(hdr)

        hdstr = (br + br).join(header_strs)
        hdstr = '<pre>' + anchor + hdstr + br + '</pre>' + anchor

        return hdstr

    def table(self):
        """
        Format selected parameters into a table.

        Uses comma-separated filter values as keys to display
        from each header.

        Requires `pandas` to display.
        """
        # read text to filter
        # may be comma-separated keys (no substrings)
        find_text = self.findText.text().strip()

        if find_text == '':
            # clear previous filter / table
            self.textEdit.setHtml(self.html)
        else:
            # check for pandas
            try:
                import pandas as pd
            except ImportError:
                msg = '(install pandas for table display)'
                self.textEdit.setPlainText(msg)
                return

            # split field on commas for multiple keys
            sep = ['File name']
            sep.extend(find_text.upper().split(','))

            # find keys in headers
            data = {}
            sortkeys = sorted(self.text.keys())
            for key in sep:
                data[key] = []
                for fpath in sortkeys:
                    hlist = self.text[fpath]
                    if type(hlist) is not list:
                        hlist = [hlist]
                    for i, hdr in enumerate(hlist):
                        if key == 'File name':
                            basename = os.path.splitext(
                                os.path.basename(fpath))[0]
                            if 'EXTNAME' in hdr:
                                extname = " [{}]".format(hdr['EXTNAME'])
                            else:
                                extname = " [{}]".format(i)
                            data[key].append(basename + extname)
                        elif key in hdr:
                            data[key].append(hdr[key])
                        else:
                            data[key].append(None)

            # pandas dataframe for table display
            df = pd.DataFrame(data, columns=sep)
            htmltable = df.to_html(max_rows=None, max_cols=None, border=1)
            htmltable = htmltable.replace('<table',
                                          '<table cellpadding="10"', 1)
            self.textEdit.setHtml(htmltable)

        # repaint required for some versions of Qt/OS
        self.textEdit.repaint()

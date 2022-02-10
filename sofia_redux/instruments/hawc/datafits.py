# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Data storage class for FITS images and tables."""

from datetime import datetime
import os
import time
import gc

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.hawc.dataparent import DataParent
from sofia_redux.toolkit.utilities.fits import hdinsert

__all__ = ['DataFits']


class DataFits(DataParent):
    """
    Pipeline data FITS object.

    The data is stored as a list of (multi-dimensional) images and a
    similar list of tables. FITS file name and all headers are
    stored as well.
    """

    # File Name Fit: Regexp expression that fits valid filenames
    # will fit .fits or .fts at end of name
    filenamefit = r'\.(fits|fts)\Z'
    """str : Regular expression that matches valid FITS file names."""

    def __init__(self, filename='', config=None):
        """
        Initialize data object and variables.

        The parent constructor is not called by this function, since
        this class requires different attribute handling than its parent.

        Parameters
        ----------
        filename : str, optional
            Path to a text file. If specified, data will be loaded
            immediately from the file.
        config : `configobj.ConfigObj`, dict, str, or list of str, optional
            If specified, the configuration will be loaded.
        """
        # set up internal variables
        self.filename = ''
        self.rawname = ''
        self.loaded = False

        # retrieve config directory location from current file path
        # assumes config is in sofia_redux/instruments/hawc/data/config
        # and this file is in sofia_redux/instruments/hawc
        pipe_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(pipe_path, 'data', '')
        self.config_path = os.path.join(pipe_path, 'data', 'config', '')

        # set the configuration
        self.mode = None
        self.config = None
        self.config_files = []
        self.setconfig(config)

        # Image Variables:
        #   At all times len(imgdata) == len(imgnames) == len(imgheads)
        #   - If only loadhead() was called or first HDU has no data:
        #     imghead[0]=header, imgdata[0]=None, imgnames[0]='Primary Header'

        # List with image data - data is in array of int/float
        self.imgdata = []
        # List with image names (all uppercase)
        self.imgnames = []
        # List with image headers
        self.imgheads = []

        # Table Variables:
        #   At all times len(tabdata) == len(tabnames) == len(tabheads)

        # List with tables each item is a record array
        self.tabdata = []
        # List with table names (all uppercase)
        self.tabnames = []
        # List with table headers
        self.tabheads = []

        # if file exists, load it
        if os.path.exists(filename):
            self.load(filename)
        elif filename != '':
            # user specified a non-existent filename
            msg = 'No such file %s' % filename
            log.error(msg)
            raise IOError(msg)

    def __getattr__(self, name):
        """
        Get attribute.

        Allows access to these attributes:

            - header: header of primary HDU
            - image: the first image
            - table: the first table

        From DataParent.__getattr__:

            - filenamebegin: filename start
            - filenameend: filename end

        Parameters
        ----------
        name : str
            The attribute to retrieve.
        """
        # return data
        if name == 'data':
            if self.imgdata and self.imgdata[0] is not None:
                # return image if available
                return self.imgdata[0]
            elif self.tabdata:
                # else return a table
                return self.tabdata[0]
            else:
                msg = "'%s' object has no data" % type(self).__name__
                log.warning(msg)
                return None

        # return image if it's requested - raise error if no images present
        if name == 'image':
            if len(self.imgdata) == 0:
                msg = "'%s' object has no image data" % type(self).__name__
                log.warning(msg)
                return None
            elif self.imgdata[0] is None:
                msg = "'%s' object has no image data" % type(self).__name__
                log.warning(msg)
            return self.imgdata[0]
        # return table if it's requested - raise error if no tables present
        if name == 'table':
            if len(self.tabdata) == 0:
                msg = "'%s' object has no table data" % type(self).__name__
                log.warning(msg)
                return None
            return self.tabdata[0]
        # return header if it's requested - raise error if no images present
        if name == 'header':
            if len(self.imgheads) == 0:
                msg = "'%s' object has no header data" % type(self).__name__
                log.warning(msg)
                return None
            return self.imgheads[0]

        # run parent function (filenamebegin and filenameend)
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        """
        Set attribute.

        Allows setting of these attributes:

            - header: header of primary HDU
            - image: the first image
            - table: the first table

        Parameters
        ----------
        name : str
            The attribute to set.
        value : object
            The value to set.
        """
        # set the data to image
        if name == 'data':
            # if it's a table put it in table
            if issubclass(value.__class__, np.recarray):
                self.table = value
            # else assume it's an image
            else:
                self.image = value
        elif name == 'image':
            # set the image if it's requested
            if len(self.imgdata) > 0:
                self.imgdata[0] = value
            else:
                self.imgdata = [value]
                self.imgnames = ['PRIMARY IMAGE']
                self.imgheads = [fits.Header()]
            self.loaded = True
        elif name == 'table':
            # set the table if it's requested
            if len(self.tabdata) > 0:
                self.tabdata[0] = value
            else:
                self.tabdata = [value]
                self.tabnames = ['PRIMARY TABLE']
                self.tabheads = [fits.Header()]
            self.loaded = True
        elif name == 'header':
            # set the header if it's requested
            if len(self.imgdata) > 0:
                self.imgheads[0] = value
            else:
                self.imgdata = [None]
                self.imgnames = ['PRIMARY HEADER']
                self.imgheads = [value]
        else:
            # else pass the command to the parent function
            super().__setattr__(name, value)

    def loadhead(self, filename='', dataname='', hdul=None):
        """
        Load and return the primary header of the FITS file given.

        This also checks for file existence and type. Only the header
        is loaded into the DataFits object; the data is not loaded.

        Parameters
        ----------
        filename : str, optional
            The name of the file to load. If omitted,
            self.filename is used.
        dataname : str or int, optional
            The EXTNAME value or extension number of the header to be loaded
            If such a header is not found, or dataname=='', the first header
            is loaded. This option should be used if the main file
            information is not in the primary header.
        hdul : fits.HDUList, optional
            An in-memory HDUList to load, in place of an on-disk file.
        """
        # check for file existence, type and get primary header

        # set self.filename and filename
        if len(filename) > 0:
            self.filename = filename
            self.rawname = filename
        else:
            filename = self.filename

        # read fits header, checks for existing valid fits file
        if hdul is None:
            try:
                hdus = fits.open(filename)
            except (IOError, OSError, IndexError,
                    TypeError, ValueError) as error:
                log.error('LoadHead: FITS read error ' + filename)
                raise error
        else:
            hdus = hdul

        # No dataname -> return primary header
        if dataname == '':
            header = [hdus[0].header]
        elif dataname == 'all':
            # Retrieve headers from all extensions
            header = []
            for ext in hdus:
                header.append(ext.header)
        else:
            # Look for correct dataname
            try:
                header = [hdus[dataname].header]
            except (KeyError, ValueError, IndexError):
                msg = "loadhead: HDU with EXTNAME=%s not found" % dataname
                log.error(msg)
                raise ValueError(msg)

        # Fill in the header, if necessary fill in data
        self.imgheads = header
        self.imgdata = [None] * len(header)
        try:
            self.imgnames = [h['EXTNAME'].upper() for h in header]
        except KeyError:
            self.imgnames = ['PRIMARY HEADER'] * len(header)

        # Update the config for the current file if possible
        try:
            date = self.getheadval('DATE-OBS', errmsg=False)
            date = datetime.strptime(
                date.split('.')[0], "%Y-%m-%dT%H:%M:%S")
            self.mergeconfig(date=date)
        except (KeyError, ValueError, AttributeError):
            log.debug('No date found for %s' % filename)

        # Set the default pipeline mode from the config
        self.mode = self.get_pipe_mode()
        if self.mode is not None:
            self.mergeconfig(mode=self.mode)

        # Fill in the filename
        self.filename = filename
        if hdul is None:
            hdus.close()

    def load(self, filename='', hdul=None):
        """
        Load a FITS file into the data object.

        Loads headers, images, and tables from the file on disk.
        After loading, self.loaded is set to True.

        Parameters
        ----------
        filename : str, optional
            The name of the file to load. If omitted, self.filename is used.
        hdul : fits.HDUList, optional
            An in-memory HDUList to load, in place of an on-disk file.
        """
        # Clear file data
        self.imgdata, self.imgheads, self.imgnames = [], [], []
        self.tabdata, self.tabheads, self.tabnames = [], [], []

        # Get filename and file checks
        # set self.filename and filename
        if len(filename) > 0:
            self.filename = filename
            self.rawname = filename
        else:
            filename = self.filename

        if hdul is not None:
            # directly load provided HDUL
            self.loadhead(hdul=hdul)
            hdus = hdul
        else:
            # check for file existence, type and get primary header
            #   fills first entry into imgdata, imgnames
            self.loadhead(filename)

            # Read File
            hdus = fits.open(filename, memmap=False)

        # Collect images / Load them
        # Search for ImageHDUs (does not include PrimaryHDU)
        imgind = [i for i in range(len(hdus))
                  if isinstance(hdus[i], fits.ImageHDU)]
        # store number of images
        imgn = len(imgind)

        # get naxis and naxis1 from primary header, check if keywords exist
        naxis = 0
        naxis1 = 0
        try:
            naxis = int(self.getheadval('NAXIS'))
            if naxis > 0:
                naxis1 = self.getheadval('NAXIS1', errmsg=False)
        except (KeyError, ValueError, AttributeError):
            # KeyError: keyword not found (no keywords available)
            log.warning('Load: missing naxis keywords in fits file '
                        + filename)

        # Check if file has no image (i.e. if naxis==1 and naxis1==0)
        # -> No change, primary image stays as it was from loadhead()
        if naxis * naxis1 == 0:
            log.debug('Load: No image data in first HDU')
        else:
            # Load first HDU data if there is image data in it
            imgn += 1
            self.imgdata[0] = hdus[0].data

            # Set image name
            if 'EXTNAME' in self.imgheads[0]:
                self.imgnames[0] = self.imgheads[0]['EXTNAME'].upper()
            else:
                self.imgnames[0] = 'PRIMARY IMAGE'

        # Message with number of images
        log.debug('Load: %d image(s) in file %s' % (imgn, filename))

        # Load subsequent images
        for ind in imgind:
            # get name
            if 'EXTNAME' in hdus[ind].header.keys():
                self.imgnames.append(hdus[ind].header['EXTNAME'].upper())
            else:
                self.imgnames.append('SECONDARY IMAGE %d' % ind)

            # get data and header
            self.imgdata.append(hdus[ind].data)
            self.imgheads.append(hdus[ind].header)

        # Search for BinTableHDUs
        tabind = [i for i in range(len(hdus))
                  if isinstance(hdus[i], fits.BinTableHDU)]
        tabn = len(tabind)

        # Messages on number of tables
        log.debug('Load: %d table(s) in file %s' % (tabn, filename))

        # Load all tables
        for ind in tabind:
            # if table is empty -> warning and skip it
            try:
                if hdus[ind].data is None or len(hdus[ind].data) == 0:
                    msg = 'Load: Table in HDU number %d has no data' % ind
                    msg += ' -> Ignoring this HDU'
                    log.warning(msg)
                    continue
            except (IndexError, AttributeError, TypeError):
                msg = 'Load: Problem loading table in HDU number %d' % ind
                msg += ' -> Ignoring this HDU'
                log.warning(msg)
                continue

            # get name
            if 'EXTNAME' in hdus[ind].header:
                self.tabnames.append(hdus[ind].header['EXTNAME'].upper())
            else:
                # tables can't be in primary extension, so start secondary
                # after index 1
                if ind > 1:
                    self.tabnames.append('SECONDARY TABLE %d' % (ind - 1))
                else:
                    self.tabnames.append('PRIMARY TABLE')

            # get data, header
            self.tabdata.append(np.rec.array(hdus[ind].data))
            self.tabheads.append(hdus[ind].header)

        # Update the config for the current file if possible
        try:
            date = self.getheadval('DATE-OBS', errmsg=False)
            date = datetime.strptime(
                date.split('.')[0], "%Y-%m-%dT%H:%M:%S")
            self.mergeconfig(date=date)
        except (KeyError, ValueError, AttributeError):
            log.debug('No date found for %s' % filename)

        # Set the default pipeline mode from the config
        self.mode = self.get_pipe_mode()
        if self.mode is not None:
            self.mergeconfig(mode=self.mode)

        # Close the file
        if hdul is None:
            hdus.close()
            gc.collect()

        self.loaded = True
        log.debug('Load: loaded fits file')

    def to_hdulist(self, filename=None, save_tables=True):
        """
        Return an astropy HDUList equivalent to the current data object.

        Also sets pipeline keywords in headers: PIPEVERS, FILENAME, DATE.
        The pipeline version for PIPEVERS is read from self.pipever,
        usually set in the DataParent class.

        Parameters
        ----------
        filename : str, optional
            The file name to store in the FILENAME keyword. If not
            provided, self.filename will be used.
        save_tables : bool, optional
            If not set, binary tables stored in the current object
            will not be passed to the output HDUList.

        Returns
        -------
        fits.HDUList
            The HDUList containing all requested data.
        """
        if filename is None:
            filename = self.filename

        # update pipeline keywords
        self.setheadval('PIPEVERS', DataParent.pipever.replace('.', '_'),
                        'Pipeline Version')
        self.setheadval('FILENAME', os.path.split(filename)[-1])
        self.setheadval('DATE', time.strftime('%Y-%m-%dT%H:%M:%S'))

        # make the data primary HDU -> List
        hdul = []
        for i in range(len(self.imgnames)):
            if i == 0:
                hdui = fits.PrimaryHDU(self.imgdata[i], self.imgheads[i])
            else:
                hdui = fits.ImageHDU(self.imgdata[i], self.imgheads[i])
            hdui.header['EXTNAME'] = (self.imgnames[i].upper(),
                                      'ID of the HDU')
            hdul.append(hdui)

        # make hdus for tables
        if save_tables:
            for i in range(len(self.tabnames)):
                hdut = fits.BinTableHDU(self.tabdata[i], self.tabheads[i])
                hdut.header['EXTNAME'] = (self.tabnames[i].upper(),
                                          'ID of the HDU')
                hdul.append(hdut)

        # make an HDU list
        hdulist = fits.HDUList(hdul)
        return hdulist

    def to_header_list(self, filename=None):
        """
        Return a list of all headers in the current data object.

        Headers are updated with pipeline keywords before returning
        (PIPEVERS, FILENAME, DATE).

        Parameters
        ----------
        filename : str, optional
            The file name to store in the FILENAME keyword. If not
            provided, self.filename will be used.

        Returns
        -------
        list of fits.Header
        """
        if filename is None:
            filename = self.filename

        # update pipeline keywords
        self.setheadval('PIPEVERS', DataParent.pipever.replace('.', '_'),
                        'Pipeline Version')
        self.setheadval('FILENAME', os.path.split(filename)[-1])
        self.setheadval('DATE', time.strftime('%Y-%m-%dT%H:%M:%S'))

        # make the data primary HDU -> List
        hlist = []
        for name in self.imgnames:
            hlist.append(self.getheader(name))
        for name in self.tabnames:
            hlist.append(self.getheader(name))
        return hlist

    def save(self, filename=None):
        """
        Save the data in the object to the specified file.

        Existing files are overwritten.

        Parameters
        ----------
        filename : str, optional
            The file name to store in the FILENAME keyword. If not
            provided, self.filename will be used.
        """
        # get file name
        if filename is None:
            filename = self.filename

        # make an HDU list
        hdulist = self.to_hdulist(filename=filename)

        # save the file (produce errors if not successful)
        try:
            hdulist.writeto(filename, output_verify='fix', overwrite=True)
        except (IOError, TypeError) as error:
            log.error('Save: Failed to write fits file to ' + filename)
            raise error
        log.debug('Save: wrote FITS file %s' % filename)

    def copy(self):
        """
        Return a copy of the current object.

        Returns
        -------
        DataFits
        """

        # create new object
        out = DataFits(config=self.config)

        # copy all images
        out.imgnames = self.imgnames[:]
        out.imgdata = []
        out.imgheads = []
        for imgi in range(len(self.imgdata)):
            if self.imgdata[imgi] is not None:
                out.imgdata.append(self.imgdata[imgi].copy())
            else:
                out.imgdata.append(None)
            out.imgheads.append(self.imgheads[imgi].copy())

        # copy tables
        out.tabnames = self.tabnames[:]
        out.tabdata = []
        out.tabheads = []
        for tabi in range(len(self.tabdata)):
            if self.tabdata[tabi] is not None:
                out.tabdata.append(self.tabdata[tabi].copy())
            else:
                out.tabdata.append(None)
            out.tabheads.append(self.tabheads[tabi].copy())

        # copy filename
        out.filename = self.filename
        out.rawname = self.rawname
        out.loaded = self.loaded

        # return new object
        return out

    def mergehead(self, other):
        """
        Merge the header of another data object to the existing header.

        The merge is between primary headers only.

        Header keywords are merged according to the configuration
        values specified for the 'headmerge' section. Options are:
        FIRST, LAST, MIN, MAX, SUM, OR, AND, CONCATENATE, DEFAULT.

        Parameters
        ----------
        other : DataFits
            The DataFits to merge from.
        """
        # get cards
        otherheader = other.header
        othercards = otherheader.cards
        selfcards = self.header.cards

        # add history keywords (no duplicates)
        otherhist = [card.value for card in othercards
                     if card.keyword == 'HISTORY']
        selfhist = [card.value for card in selfcards
                    if card.keyword == 'HISTORY']
        for hist in otherhist:
            if hist not in selfhist:
                hdinsert(self.header, 'HISTORY', hist, after=True)

        # Go through keywords listed in headmerge: assume self is first
        headmerge = self.config['headmerge']
        for key in headmerge.keys():
            if key in self.header and key in otherheader:
                selfval = self.header[key]
                otherval = otherheader[key]
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

    def copyhead(self, other, name=None, overwrite=True):
        """
        Copy a header into the current object.

        This function copies all header keywords, comments,
        and history from other. Will overwrite existing cards, unless
        overwrite flag is set to False. Exceptions: HISTORY and COMMENT
        cards are always appended to the end of the list of such keywords
        present in self.

        If name is None, the header from the first HDU (self.header,
        other.header) will be used.

        Parameters
        ----------
        other : DataFits
            The object to copy from.
        name : str, optional
            The name of the extension header to copy.
        overwrite : bool, optional
            If not set, keywords already present in the current header
            will not be overwritten.
        """

        if name is None:
            head1 = other.header
            head2 = self.header
        else:
            head1 = other.getheader(name)
            head2 = self.getheader(name)
        try:
            nhist = len(head1['history'])
        except KeyError:
            nhist = 0
        try:
            ncomm = len(head1['comment'])
        except KeyError:
            ncomm = 0

        for k in head1.keys():
            if k in ('COMMENT', 'HISTORY'):
                # handle later
                pass
            elif k in head2.keys():
                if overwrite:
                    head2[k] = (head1[k], head1.comments[k])
            else:
                head2[k] = (head1[k], head1.comments[k])
        for i in range(nhist):
            head2.add_history(head1['history'][i])
        for i in range(ncomm):
            head2.add_comment(head1['comment'][i])

    def copydata(self, other, dataname):
        """
        Copy data into the current object.

        Copies data (image or table) from another DataFits object to the
        current object. If an object of that name already exists it is
        overwritten. Both the data and the header are copied.

        Parameters
        ----------
        other : DataFits
            The object to copy from
        dataname : str
            The name of the extension header to copy.
        """
        # Check type and presence of data in other object
        isimg = False
        if dataname.upper() in other.imgnames:
            isimg = True
            othind = other.imageindex(dataname)
        elif dataname.upper() in other.tabnames:
            othind = other.tableindex(dataname)
        else:
            log.error('CopyData: dataname not found')
            raise ValueError('dataname not found')

        # Check if name exists in local data, delete if it's different type
        if dataname.upper() in self.imgnames:
            if not isimg:
                self.imagedel(dataname)
        elif dataname.upper() in self.tabnames:
            if isimg:
                self.tabledel(dataname)

        # Copy / Overwrite object
        if isimg:
            self.imageset(other.imgdata[othind].copy(), dataname,
                          other.imgheads[othind].copy())
        else:
            self.tableset(other.tabdata[othind].copy(), dataname,
                          other.tabheads[othind].copy())

    def imageindex(self, imagename=None):
        """
        Return the index of an image in the current object.

        Given this index, the associated image can be accessed
        via self.imgdata[index].

        Parameters
        ----------
        imagename: str, optional
            The name of the requested image. If not provided,
            the first image will be returned.

        Returns
        -------
        int
            The image index.

        Raises
        ------
        ValueError
            If the image is not found.
        """
        # empty name -> return first image
        if imagename is None:
            return 0

        # check for valid name
        try:
            ind = self.imgnames.index(imagename.upper())
        except ValueError as error:
            msg = 'invalid image name (%s)' % imagename
            log.error('Imageindex: ' + msg)
            raise error

        # return index
        return ind

    def imageget(self, imagename=None):
        """
        Return an image.

        Parameters
        ----------
        imagename : str, optional
            The name of the requested image. The first image is returned
            if not provided.

        Returns
        -------
        array-like
            The requested image.
        """
        # get index
        ind = self.imageindex(imagename)
        # return image
        return self.imgdata[ind]

    def imageset(self, imagedata, imagename=None, imageheader=None, index=-1):
        """
        Set an image.

        This should be used to add or replace an image in the current
        object. The index flag determines the position of the image
        in the image list.

        Parameters
        ----------
        imagedata : array-like or None
            A multi dimensional array containing the image data
        imagename : str or None, optional
            The name of the image to set (None for first image)
        imageheader : fits.Header
            FITS header for the image
        index : int, optional
            Indicates the position of the image in the image list
            Ignored if == -1.
        """
        totalindex = len(self.imgdata)
        # If index is not set, get a valid index
        if index < 0:
            if imagename is None:
                index = 0
                if totalindex > 0:
                    imagename = self.imgnames[0]
                else:
                    imagename = 'PRIMARY'
            elif imagename.upper() in self.imgnames:
                index = self.imgnames.index(imagename.upper())
            else:
                # name given, but not an existing name
                if totalindex > 0 and self.imgdata[0] is None:
                    # add to primary HDU, if primary HDU empty
                    index = 0
                else:
                    # add to end otherwise
                    index = totalindex

        # re-use header, if none specified
        if imageheader is None:
            if index < totalindex:
                imageheader = self.imgheads[index]

        # Set image
        if index == 0:
            hdu = fits.PrimaryHDU(imagedata, header=imageheader)
        else:
            hdu = fits.ImageHDU(imagedata, header=imageheader)
        if index < totalindex:
            # overwriting an existing image
            self.imgnames[index] = imagename.upper()
            self.imgdata[index] = imagedata
            self.imgheads[index] = hdu.header
        else:
            self.imgnames.append(imagename.upper())
            self.imgdata.append(imagedata)
            self.imgheads.append(hdu.header)
        self.loaded = True

    def imagedel(self, imagename=None):
        """
        Remove an image.

        Parameters
        ----------
        imagename : str, optional
            The name of the image to delete. The first image will be
            deleted if not specified.
        """
        # get index
        ind = self.imageindex(imagename)

        # delete image
        del self.imgnames[ind]
        del self.imgdata[ind]
        del self.imgheads[ind]

    def tableindex(self, tablename=None):
        """
        Return the index of an table in the current object.

        Given this index, the associated table can be accessed
        via self.tabdata[index].

        Parameters
        ----------
        tablename: str, optional
            The name of the requested table. If not provided,
            the first table will be returned.

        Returns
        -------
        int
            The table index.

        Raises
        ------
        RuntimeError
            If there are no tables present.
        ValueError
            If the specified table is not found.
        """
        # Check if tables are present
        if len(self.tabnames) == 0:
            msg = 'no tables in data'
            log.error('Tableindex: ' + msg)
            raise RuntimeError(msg)

        # empty name -> return first table
        if tablename is None:
            return 0

        # check for valid name
        try:
            ind = self.tabnames.index(tablename.upper())
        except ValueError as error:
            msg = 'invalid table name (%s)' % tablename
            log.error('Tableindex: ' + msg)
            raise error

        # return index
        return ind

    def tableget(self, tablename=None):
        """
        Return a table.

        Parameters
        ----------
        tablename : str, optional
            The name of the requested table. The first table is returned
            if not provided.

        Returns
        -------
        array-like
            The requested table.
        """
        # get index
        ind = self.tableindex(tablename)

        # return table
        return self.tabdata[ind]

    def tableset(self, tabledata, tablename=None, tableheader=None, index=-1):
        """
        Set a table.

        This should be used to add or replace a table in the current
        object. The index flag determines the position of the table
        in the table list.

        Parameters
        ----------
        tabledata : array-like or None
            A multi dimensional array containing the table data
        tablename : str or None, optional
            The name of the table to set (None for first table)
        tableheader : fits.Header
            FITS header for the table
        index : int, optional
            Indicates the position of the table in the table list
            Ignored if == -1.
        """

        # If index is not set, get valid index
        totalindex = len(self.tabdata)
        if index < 0:
            if tablename is None:
                index = 0
                if totalindex > 0:
                    tablename = self.tabnames[0]
                else:
                    tablename = 'PRIMARY TABLE'
            # if table exists - replace
            elif tablename.upper() in self.tabnames:
                index = self.tabnames.index(tablename.upper())
            else:
                # add to end otherwise
                index = totalindex

        # Set table
        if index < totalindex:
            # overwrite
            self.tabnames[index] = tablename.upper()
            self.tabdata[index] = tabledata
            if tableheader is not None:
                self.tabheads[index] = tableheader
        else:
            self.tabnames.append(tablename.upper())
            self.tabdata.append(tabledata)

            # make sure tableheader is valid
            if tableheader is None:
                tableheader = fits.Header()
            self.tabheads.append(tableheader)

        self.loaded = True

    def tabledel(self, tablename=None):
        """
        Remove a table.

        Parameters
        ----------
        tablename : str, optional
            The name of the table to delete. The first table will be
            deleted if not specified.
        """
        # get index
        ind = self.tableindex(tablename)

        # delete table
        del self.tabnames[ind]
        del self.tabdata[ind]
        del self.tabheads[ind]

    def tableaddcol(self, colname, array, tablename=None, dtype=None):
        """
        Add a column to the table.

        If the table under tablename doesn't exist, it is created.
        This is intended to be used with single-dimension columns only.

        Parameters
        ----------
        colname : str
            The new column name.
        array : array-like
            Values for the new column
        tablename : str
            The name of the table to add to (None for first table)
        dtype : type, optional
            Data type for the new column
        """
        # make a basic array
        array = np.asarray(array)

        # set correct data type
        if dtype is None:
            dtype = array.dtype

        # get additional dimension
        if len(array.shape) > 1:
            newtype = (colname, dtype, array.shape[1])
        else:
            newtype = (colname, dtype)

        # Get table index (-1 if table needs to be created)
        tabind = -1
        if tablename is None:
            if len(self.tabnames) > 0:
                tabind = 0
            else:
                tablename = "Table"
        else:
            if tablename.upper() in self.tabnames:
                tabind = self.tableindex(tablename)

        # Make new table if necessary
        if tabind < 0:
            # If there is no existing table:

            # get new table data type
            newdtype = np.dtype([newtype])

            # Make new table
            newtable = np.empty(array.shape[0], dtype=newdtype)

            # Add to list of tables
            tabind = len(self.tabnames)
            self.tabdata.append(newtable)
            self.tabnames.append(tablename.upper())
            self.tabheads.append(fits.Header())
        else:
            # table exists: base new table on old table
            table = self.tabdata[tabind]

            # check if dimension of new value array is correct
            if len(table) != len(array):
                msg = 'column array len (%d) != table len (%d)' % \
                      (len(array), len(table))
                log.error('TableAddCol: ' + msg)
                raise ValueError(msg)

            # get new data type for table
            newdtype = np.dtype(table.dtype.descr + [newtype])

            # make new table
            newtable = np.empty(table.shape, dtype=newdtype)

            # fill old table values
            for field in table.dtype.fields:
                newtable[field] = table[field]

        # fill new table values
        newtable[colname] = array

        # copy new table to self.table
        self.tabdata[tabind] = newtable

    def tableaddrow(self, tablerow, tablename=None):
        """
        Add a row to a data table.

        Parameters
        ----------
        tablerow : array-like
            The elements of the row to be added
        tablename : str
            The name of the table to add to (None for first table)
        """
        # Get table
        tabind = self.tableindex(tablename)
        table = self.tabdata[tabind]

        # check if tablerow format is correct (by converting it)
        if len(tablerow) != len(table[0]):
            log.error('TableAddRow: table row has invalid length')
            raise ValueError('table row has invalid length')
        try:
            tablerow = np.rec.array(tablerow, dtype=table.dtype)
        except (ValueError, TypeError):
            log.error('TableAddRow: table row has invalid element type')
            raise ValueError('table row has invalid element type')

        # Add to table
        self.tabdata[tabind] = np.insert(table, len(table), tablerow)

    def tabledelcol(self, colname, tablename=None):
        """
        Delete a column of the data table.

        Parameters
        ----------
        colname: str or list of str
           The name(s) of the column(s) to delete
        tablename: str, optional
           The name of the table to delete from (None for first table)
        """
        # Get table
        tabind = self.tableindex(tablename)
        table = self.tabdata[tabind]

        # Check if colname is valid, make sure it's a list
        olddt = table.dtype
        if isinstance(colname, str):
            colname = [colname]
        if isinstance(colname, (list, tuple)):
            for c in colname:
                if c not in olddt.names:
                    msg = 'Invalid column name %s' % c
                    log.error('TableDelCol: ' + msg)
                    raise ValueError(msg)
        else:
            msg = "Invalid colname '%s'. Must be string " \
                  "or list/tuple" % colname
            log.error('TableDelCol: ' + msg)
            raise ValueError(msg)

        # if all columns deleted, clear table
        if len(olddt) - len(colname) <= 0:
            del self.tabdata[tabind]
            del self.tabnames[tabind]
            del self.tabheads[tabind]
        else:
            # remove column(s) from table

            # if it's a pyfits.FITS_rec -> Treat specially
            # (this was pyfits.fitsrec.FITS_rec)
            # test for FITS_rec. ndarray and recarray don't
            # support names method
            try:
                names = table.names
                formats = table.columns.formats
                dims = table.columns.dims
                units = table.columns.units

                cols = []
                for n, f, d, u in zip(names, formats, dims, units):
                    if n not in colname:
                        cols.append(fits.Column(name=n, format=f, dim=d,
                                                unit=u, array=table.field(n)))
                tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
                self.tabdata[tabind] = tbhdu.data
                self.tabheads[tabind] = tbhdu.header
            except AttributeError:
                # assume table is a regular record array
                newnames = [n for n in olddt.names if n not in colname]
                newtable = table[newnames]
                self.tabdata[tabind] = newtable

    def tabledelrow(self, index, tablename=None):
        """
        Delete a row of the data table.

        Parameters
        ----------
        index: int
            The index of the row to delete.
        tablename: str, optional
            The name of the table to delete from (None for first table)
        """
        # Get table
        tabind = self.tableindex(tablename)
        table = self.tabdata[tabind]

        # Check if index is valid
        if index >= len(table):
            log.error('TableDelRow: Invalid row index %d' % index)
            raise ValueError('Invalid row index %d' % index)

        # if table has one row -> clear table
        if len(table) < 2:
            del self.tabdata[tabind]
            del self.tabnames[tabind]
            del self.tabheads[tabind]
        else:
            # remove row from table
            self.tabdata[tabind] = np.delete(table, index).copy()

    def tablemergerows(self, rows):
        """
        Merge several table rows into a single row.

        Each column is merged according to the rules defined in the
        [table] section of the configuration file. Options are:
        FIRST, LAST, MIN, MAX, MED, AVG, SUM, WTAVG.

        Parameters
        ----------
        rows : record
            NumPy record or FITS table rows to merge.

        Returns
        -------
        record
            The merged row.

        Raises
        ------
        AttributeError
            If input rows have incorrect data type.
        """
        # check if rows has same format as table
        try:
            rows.dtype
        except AttributeError as error:
            msg = 'input rows are incorrect data type'
            log.error('TableMergeRows: %s' % msg)
            raise error

        # if _newdtype != table.dtype :
        #    msg = 'input rows have different format than table'
        #    log.error('TableMergeRows: %s' % msg)
        #    raise TypeError(msg)

        # make output table row (copy from first row)
        outrow = (rows[0:1].copy())[0]

        # run through columns and merge values
        for colname in rows.dtype.names:
            # get merge function
            try:
                funct = self.config['table'][colname.lower()].lower()
            except (IndexError, ValueError, KeyError):
                # if keyword is not available
                log.warning('TableMergeRows: Missing table '
                            'merge entry for '
                            'column -%s- returning first '
                            'row value' % colname)
                funct = 'first'

            # Try to run the function
            # Comment: float() is necessary, otherwise values get messed up
            try:
                if funct == 'first':
                    pass
                elif funct == 'last':
                    outrow[colname] = rows[-1][colname]
                elif funct == 'min':
                    outrow[colname] = float(np.nanmin(rows[colname]))
                elif funct == 'max':
                    outrow[colname] = float(np.nanmax(rows[colname]))
                elif funct == 'med':
                    outrow[colname] = float(np.nanmedian(rows[colname]))
                elif funct == 'avg':
                    outrow[colname] = float(np.nanmean(rows[colname]))
                elif funct == 'sum':
                    outrow[colname] = float(np.nansum(rows[colname]))
                elif funct == 'wtavg':
                    tmp = float(np.nansum(rows[colname] * rows['Samples']))
                    tmpsum = float(np.nansum(rows['Samples']))
                    if tmpsum > 0:
                        outrow[colname] = tmp / tmpsum
                    else:
                        outrow[colname] = tmp * 0.0
                else:
                    log.warning('TableMergeRows: Unknown operation -'
                                + funct + '- for column -' + colname
                                + '- returning first row value')
            except (NameError, TypeError, ValueError, KeyError):
                # if unsuccessful return error
                log.warning('TableMergeRows: Error in %s( %s ) - '
                            'returning first row value' %
                            (funct, colname))
        return outrow

    def tablemergetables(self, tables):
        """
        Return a new table containing data merged from the input table(s).

        Columns are merged according to the rules defined in the [table]
        section of the configuration file. Options are:
        FIRST, LAST, MIN, MAX, MED, AVG, SUM, WTAVG.

        Note that each table is assumed to have a single
        row of data. If you need to merge rows of data, use
        tablemergerows().

        Parameters
        ----------
        tables : list of BinTableHDU data
            Tables should be NumPy records or FITS tables.

        Returns
        -------
        BinTableHDU
            The merged table.
        """
        ntable = len(tables)
        try:
            # FITS tables
            names = tables[0].names
            formats = tables[0].columns.formats
            dims = tables[0].columns.dims
            units = tables[0].columns.units
        except AttributeError:
            msg = 'TableMergeTables is only available for FITS tables'
            log.error(msg)
            raise ValueError(msg)

        if ntable == 1:
            msg = 'TableMergeTables: Only 1 table passed'
            log.debug(msg)
            tbhdu = fits.BinTableHDU(tables[0])
            return tbhdu

        # loop through all tables, make sure names, formats, dims, and units
        # are the same among them.
        for i in range(1, ntable):
            if (names != tables[i].names
                    or formats != tables[i].columns.formats
                    or dims != tables[i].columns.dims
                    or units != tables[i].columns.units):
                log.error('TableMergeTables: columns differ for merging')
                raise ValueError

        cols = []
        for n, f, d, u in zip(names, formats, dims, units):
            # get merge function
            try:
                funct = self.config['table'][n.lower()].lower()
            except (KeyError, ValueError):
                # if keyword is not available
                log.warning('TableMergeTables: Missing table '
                            'merge entry for column -%s- '
                            'returning first row value' % n)
                funct = 'first'

            try:
                if funct == 'first':
                    tmp = tables[0][n]
                elif funct == 'last':
                    tmp = tables[-1][n]
                elif funct == 'min':
                    tmp = np.nanmin([a[n] for a in tables], axis=0)
                elif funct == 'max':
                    tmp = np.nanmax([a[n] for a in tables], axis=0)
                elif funct == 'med':
                    tmp = np.nanmedian([a[n] for a in tables], axis=0)
                elif funct == 'avg':
                    tmp = np.nanmean([a[n] for a in tables], axis=0)
                elif funct == 'sum':
                    tmp = np.nansum([a[n] for a in tables], axis=0)
                elif funct == 'wtavg':
                    tmp = np.nansum([a[n] * a['Samples']
                                     for a in tables], axis=0)
                    tmpsum = np.nansum([a['Samples']
                                        for a in tables], axis=0)
                    tmp = tmp / tmpsum
                else:
                    log.warning('TableMergeTables: Unknown '
                                'operation -' + funct + '- for column -'
                                + n + '- returning first row value')
                    tmp = tables[0][n]
            except (NameError, TypeError, ValueError, KeyError):
                # if unsuccessful return error
                log.warning('TableMergeRows: Error in %s( %s ) - '
                            'returning first row value' %
                            (funct, n))

                tmp = tables[0][n]
            cols.append(fits.Column(name=n, format=f, dim=d,
                                    unit=u, array=tmp))

        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        return tbhdu

    def getheader(self, dataname=''):
        """
        Return a stored header.

        Parameters
        ----------
        dataname : str, optional
            The name of image/table to return a header from; otherwise the
            primary header.

        Returns
        -------
        fits.Header
            The requested header.

        Raises
        ------
        ValueError
            If the requested header could not be found.
        """
        # Return primary header
        if dataname == '':
            header = self.header
        elif dataname.upper() in self.imgnames:
            # Return image header
            index = self.imageindex(dataname)
            header = self.imgheads[index]
        elif dataname.upper() in self.tabnames:
            # Return table header
            index = self.tableindex(dataname)
            header = self.tabheads[index]
        else:
            # Dataname not found -> return error message
            msg = 'Invalid data name (%s)' % dataname
            log.error('GetHeader: ' + msg)
            raise ValueError(msg)
        return header

    def setheader(self, header, dataname=''):
        """
        Overwrite a stored header.

        Parameters
        ----------
        header : fits.Header
            The FITS header to set.
        dataname : str, optional
            The name of image/table to return a header from; otherwise the
            primary header.
        """
        # Set primary header
        if dataname == '':
            self.header = header
        elif dataname.upper() in self.imgnames:
            # Set image header
            index = self.imageindex(dataname)
            self.imgheads[index] = header
        elif dataname.upper() in self.tabnames:
            # Set table header
            index = self.tableindex(dataname)
            self.tabheads[index] = header
        else:
            # Dataname not found -> return error message
            msg = 'Invalid data name (%s)' % dataname
            log.error('SetHeader: ' + msg)
            raise ValueError(msg)

    def getheadval(self, key, dataname='', errmsg=True):
        """
        Get a header value.

        Returns the value of the requested key from
        the header. If the keyword is present in the [Header] section
        of the configuration that value is returned instead. In case that
        value from the configuration file is itself a header key, the value
        stored under that key is returned. If the key can not be found an
        KeyError is produced and a warning is issued.

        Parameters
        ----------
        key : str
            The header key to retrieve.
        dataname : str, optional
            The header extension to retrieve from. First, if not specified.
        errmsg : bool, optional
            If set, an error message will be raised if the keyword could
            not be found.

        Returns
        -------
        str, int, float, or bool
            The header value.

        Raises
        ------
        KeyError
            If the key is not found, and errmsg = True.
        """
        val = None
        # Look in the config
        try:
            # get the value
            val = self.config['header'][key.upper()]

            # Check if it's optional header replacement i.e. starts with '?_'
            if val[:2] in ['?_', '? ', '?-']:
                # if key is not in the header -> use key name
                # under value instead
                if key not in self.header:
                    log.info('Getheadval: Using %s keyword for %s' %
                             (val[2:].upper(), key))
                    key = val[2:].upper()
                val = None
            elif val[0].isupper() and val[:2] not in ['T ', 'F ']:
                # it's a Header replacement (but not T/F)
                log.info('Getheadval: Using %s value for %s' %
                         (val.upper(), key))
                key = val.upper()
                val = None
            else:
                # make it a pyfits.Card then get value and comment
                card = fits.Card()
                card = card.fromstring(key.upper() + ' = ' + val)
                log.info('Getheadval: Setting %s to %s' % (key, val))
                val = card.value

                # update value in header
                self.setheadval(key, val, card.comment,
                                dataname=dataname)

        except KeyError:
            # if key is not in config - continue
            pass

        except TypeError:
            # if config is not yet loaded - return error
            log.warning('GetHeadVal: Missing Configuration')

        # Look in the header
        if val is None:
            # get the header from dataname
            header = self.getheader(dataname)
            # get value from header
            try:
                val = header[key]
            except KeyError:
                # if keyword is not found
                msg = 'Missing %s keyword in header %s' % (key, dataname)
                if errmsg:
                    log.error('GetHeadVal: %s' % msg)
                raise KeyError(msg)

        # if Value is a pyfits.core.Undefined i.e. no keyword
        if isinstance(val, fits.Undefined):
            msg = 'Missing value for key = %s - returning empty string' % key
            log.warning('GetHeadVal: %s' % msg)
            val = ''

        return val

    def setheadval(self, key, value, comment=None, dataname=''):
        """
        Set a FITS header keyword.

        If no header exists, it will be created and added as the
        primary header.

        Parameters
        ----------
        key : str
            The header key to set.
        value : str, int, float, or bool
            The keyword value to set.
        comment : str, optional
            The comment to set for the keyword.
        dataname : str, optional
            The header extension to retrieve from. First, if not specified.
        """
        # If no header exists, make a first empty image
        if len(self.imgheads) == 0:
            self.imgheads = [fits.PrimaryHDU().header]
            self.imgdata = [None]
            self.imgnames = ['PRIMARY']

        # Set value into hdr
        hdr = self.getheader(dataname=dataname)
        if key == 'HISTORY':
            hdinsert(hdr, key, value, after=True)
        else:
            hdinsert(hdr, key, value, comment=comment)

    def delheadval(self, key, dataname=''):
        """
        Delete one or more FITS keywords.

        Parameters
        ----------
        key : str or list of str
            The header key to delete.
        dataname : str, optional
            The header extension to retrieve from. First, if not specified.

        Raises
        ------
        ValueError
            If the key is badly specified.
        """
        if isinstance(key, (list, tuple)):
            for k in key:
                self.delheadval(k, dataname)
        elif isinstance(key, str):
            header = self.getheader(dataname=dataname)
            if key in header:
                del(header[key])
        else:
            msg = 'Invalid key (%s). Must be a str, ' \
                  'list, or tuple' % repr(key)
            log.error('DelHeadVal: ' + msg)
            raise ValueError(msg)

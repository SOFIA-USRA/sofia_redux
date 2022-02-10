# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Flat fielding pipeline step."""

import os

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steploadaux import StepLoadAux

__all__ = ['StepFlat']


class StepFlat(StepLoadAux):
    r"""
    Correct for flat response for chop/nod data.

    This pipeline step corrects the flux data for instrumental
    response, using a flat file made from a reference sky calibration
    and an internal calibrator file.  Flats are made by
    `sofia_redux.instruments.hawc.steps.StepMkflat`.  Input for this step
    is the demodulated data, produced by
    `sofia_redux.instruments.hawc.steps.StepDemodulate` and
    `sofia_redux.instruments.hawc.steps.StepDmdCut`.

    In the output from this step, the R array and T array columns
    of the input demodulated data and their corresponding variances are
    converted to images and their columns removed from the table. In
    addition, separate bad pixel masks for the R and T array are
    copied from the flat file and appended to the output object.
    Pixels with a value of zero in the mask are good, any other value
    is bad.

    Notes
    -----
    For the data :math:`d` and flat image :math:`f` with variances
    :math:`\sigma^2` and :math:`\sigma_f^2` respectively, the
    correction is applied as

      .. math:: d' = f d

    and propagated to the variance as

      .. math:: \sigma'^2 = f^2 \sigma^2  + d^2 \sigma_f^2 .


    """
    def __init__(self):
        # call superclass constructor (calls setup)
        super().__init__()

        # list of data and flats
        # used in run() for every new input data file
        self.datalist = []

        # list containing arrays with flat values
        self.flats = []

        # Pipedata object containing the flat file
        self.flatdata = DataFits()

        # flat file info and header keywords to fit
        # name of selected flat file
        self.flatfile = ''

        # FITS keywords that have to fit
        self.fitkeys = []

        # values of the keywords (from the first data file)
        self.keyvalues = []

    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'flat', and are named with
        the step abbreviation 'FLA'.

        Parameters defined for this step are:

        labmode : bool
            If True, flat correction will be skipped.
        flatfile : str
            File name glob to match flat files.  Default is
            'flats/*OFT*.fits' to match flat files in a folder
            named flats, in the same directory as the input file.
        flatfitkeys : list of str
            Keys that need to match between flat and data file.
        bkupflat : str
            File name glob to match a set of backup files.  These
            are used only if the files specified by flatfile are
            not found.
        """
        # Name of the pipeline reduction step
        self.name = 'flat'
        self.description = 'Flat Correct'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'fla'

        # Clear Parameter list
        self.paramlist = []

        self.paramlist.append(['labmode', False,
                               'If labmode = True, will skip flat correction'])

        # Get parameters for StepLoadAux, replace auxfile with flatfile
        self.loadauxsetup('flat')

    def loadauxname(self, auxpar='', data=None, multi=False):
        """
        Search for files matching auxfile.

        Overrides the default function in order to make flat path
        relative to data output directory if necessary.

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
        # override loadauxname to make flat path relative to
        # data output directory if necessary

        # Set auxpar
        if len(auxpar) == 0:
            auxpar = self.auxpar

        # Get parameters
        auxfile = os.path.expandvars(self.getarg(auxpar + 'file'))
        if not os.path.isabs(auxfile):
            # if input folder is not an absolute path, make it
            # relative to the data location
            auxfile = os.path.join(
                os.path.dirname(self.datain.filename),
                auxfile)
        return self._loadauxnamefile(auxfile, auxpar, data, multi)

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Read data from the flat file.
        2. Apply the flat to the R and T arrays.
        3. Propagate the variance on the flux.
        4. Store the data as images in the output object.
        """
        # Load flat files
        self.loadflat()

        # Copy datain to dataout
        self.dataout = self.datain.copy()

        # Apply Flatfield to each item in data list
        datalist = ['R array', 'T array']
        for dataind in range(len(datalist)):
            dataitem = datalist[dataind]

            # Get dataitem from self.table columns
            image = self.datain.table[dataitem]
            flatimage = self.flats[2 * dataind]

            var = self.datain.table[dataitem + ' VAR']
            flatvar = self.flats[2 * dataind + 1]

            # Flatfield
            image, var = self.flatfield(image, var, flatimage, flatvar)

            # Store image in dataout
            self.dataout.imageset(image, dataitem)
            self.dataout.imageset(var, dataitem + ' VAR')

            # Remove table column from dataout
            self.dataout.tabledelcol(dataitem)
            self.dataout.tabledelcol(dataitem + ' VAR')

        # Add additional image frames from flat file
        addfromfile = ['R BAD PIXEL MASK', 'T BAD PIXEL MASK']
        for dataitem in addfromfile:
            self.dataout.imageset(
                self.flatdata.imageget(dataitem),
                imagename=dataitem,
                imageheader=self.flatdata.getheader(dataitem))

        # Remove the instrumental configuration HDU
        if 'CONFIGURATION' in self.dataout.imgnames:
            self.dataout.imagedel('CONFIGURATION')

        # Update DATATYPE
        self.dataout.setheadval('DATATYPE', 'IMAGE')

        # Add flat file to History
        flatbasename = self.flatdata.filename.split(
            str(self.dataout.data_path))[-1]
        self.dataout.setheadval('HISTORY', 'FLAT: %s' % flatbasename)

        # Update PROCSTAT to level 2
        self.dataout.setheadval('PROCSTAT', 'LEVEL_2')

    def loadflat(self):
        """
        Load the flat images.

        The data is stored in self.flats.
        """
        if self.getarg('labmode'):
            df = DataFits()
            df.filename = 'Lab flat (no correction)'
            tabshape = self.datain.table['R Array'].shape
            imshape = (tabshape[1], tabshape[2])
            df.imageset(np.full(imshape, 1.0),
                        imagename='R Array')
            df.imageset(np.full(imshape, 1.0),
                        imagename='T Array')
            df.imageset(np.full(imshape, 0.0),
                        imagename='R Var')
            df.imageset(np.full(imshape, 0.0),
                        imagename='T Var')
            df.imageset(np.full(imshape, 0, dtype=int),
                        imagename='R BAD PIXEL MASK')
            df.imageset(np.full(imshape, 0, dtype=int),
                        imagename='T BAD PIXEL MASK')
            self.flatdata = df
        else:
            # Search for flat and load it into data object
            self.flatdata = self.loadauxfile()

        # find flat field data arrays and store them
        self.flats = []

        # Loop through datalist items:
        # expect 2 images -- R array, T array, followed by R Var, T Var
        for dataind in range(2):
            # data
            self.flats.append(self.flatdata.imgdata[dataind])
            # variance
            self.flats.append(self.flatdata.imgdata[dataind + 2])

    def flatfield(self, imgin, varin, flat, flatvar):
        """
        Flat field an array.

        The flux image is multiplied by the flat data.

        Parameters
        ----------
        imgin : array-like
            Input flux array.
        varin : array-like
            Input variance array.
        flat : array-like
            Flat array matching imgin.
        flatvar : array-like
            Variance array matching imgin.

        Returns
        -------
        imgout : array-like
            Flat fielded image.
        varout : array-like
            Updated variance.
        """
        # Check flatfield dimension
        self.checksize(imgin.shape, flat.shape)

        # Apply flatfield
        imgout = imgin * flat
        varout = (varin * flat**2) + (flatvar * imgin**2)

        return imgout, varout

    def checksize(self, datashape, flatshape):
        """
        Validate data and flux shapes.

        Parameters
        ----------
        datashape : tuple
            Data shape.
        flatshape : tuple
            Flat image shape.

        Raises
        ------
        ValueError
            If the flat does not match the data.

        """
        if len(datashape) >= len(flatshape):
            # Data has >= dimensions than flat -> compare
            begind = len(datashape) - len(flatshape)
            if datashape[begind:] != flatshape:
                msg = 'Flat does not fit data in file %s' % \
                      self.datain.filename
                log.error('FlatField: %s' % msg)
                raise ValueError(msg)
        else:
            # More dimensions in flat data -> report error
            msg = 'Flat does not fit data in file %s' % \
                  self.datain.filename
            log.error('LoadFlat: %s' % msg)
            raise ValueError(msg)

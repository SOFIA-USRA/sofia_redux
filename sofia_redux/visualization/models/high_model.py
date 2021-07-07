# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The High-Level Models for EOS.

These models correspond to a single FITS file which contain
one or more mid-level models. They also define the methods
by which EOS interacts with models.

Currently implemented high-level models:
  - Grism
  - MultiOrder
"""

from typing import List, Dict, Optional, TypeVar, Union
import astropy.io.fits as pf
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.models import mid_model, low_model

__all__ = ['HighModel', 'Grism', 'MultiOrder']

MidModels = TypeVar('MidModels', mid_model.Order, mid_model.Book)
LowModels = TypeVar('LowModels', low_model.Image, low_model.Spectrum)
LowerModels = Union[MidModels, LowModels, np.ndarray]


class HighModel(object):
    """
    Abstract class for high-level models.

    This class does not contain method implementations.  It should
    be subclassed before using.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        FITS HDU list to load.
    """

    def __init__(self, hdul: pf.HDUList) -> None:
        self.hdul = hdul
        try:
            filename = hdul.filename()
        except TypeError:  # pragma: no cover
            filename = hdul[0].header.get('FILENAME', None)
        if not filename:
            filename = hdul[0].header.get('FILENAME', 'UNKNOWN')

        self.filename = filename
        self.id = self.filename
        self.index = 0
        self.enabled = True
        self.default_ndims = 0

    def load_data(self) -> None:
        """Load the data into the model."""
        raise NotImplementedError

    def retrieve(self, **kwargs) -> None:
        """Retrieve data from the model."""
        raise NotImplementedError

    def list_enabled(self) -> None:
        """List enabled data subsets."""
        raise NotImplementedError

    def valid_field(self, field: str) -> None:
        """
        Determine if the field is a valid parameter.

        Parameters
        ----------
        field : str
            Name of field.

        Returns
        -------
        check : bool
            True if the field is valid, False otherwise.
        """
        raise NotImplementedError

    def enable_orders(self, *args, **kwargs) -> None:
        """Enable specified orders."""
        raise NotImplementedError


class Grism(HighModel):
    """
    Object for describing a Grism data file.

    Extends `HighModel`.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDU list from a Grism FITS file.

    Notes
    -----
    Grism data sets are unique for containing both
    images and spectra. Older grism data sets do not
    have the images, only the single extension.
    """

    def __init__(self, hdul: pf.HDUList) -> None:
        super().__init__(hdul)
        self.default_field = 'spectral_flux'
        self.default_ndims = 1
        self.num_orders = 0

        self.books = list()
        self.orders = list()
        self.load_data()
        log.debug(f'Initialized Grism with {len(self.books)} '
                  f'Books and {len(self.orders)} Orders.')

    def load_data(self) -> None:
        """
        Parse the contents of the HDUL into appropriate classes.

        Older Grism files only have one extension, without
        the image data. As such, the `Book` object is not instantiated
        here.
        """
        if self._spectra_only():
            self._load_order()
        elif self._image_only():
            self._load_book()
        else:
            self._load_order()
            self._load_book()

    def _image_only(self) -> bool:
        """Determine if there if only images are in the file."""

        # TODO - this check is not reliable.
        file_codes = ['CLN', 'DRP', 'LNZ', 'STK',
                      'LOC', 'TRC', 'APS', 'BGS']
        for code in file_codes:
            if f'_{code}_' in self.filename:
                return True
        return False

    def _spectra_only(self) -> bool:
        """Determine if there are only spectra in the file."""
        file_codes = ['CMB', 'MRG', 'SPC', 'CAL', 'RSP', 'IRS']
        for code in file_codes:
            if f'_{code}' in self.filename:
                return True
        return False

    def _load_order(self) -> None:
        """Load all orders.

        A grism observation will only have one `Order`
        in it. `num_orders` should be one at the end.
        The order might be split across several
        extensions or combined in one. Both cases
        are handled by the `Order` class.

        """
        # technically, these are apertures, but we'll handle them as orders
        # for display purposes
        self.num_orders = self._aperture_test()
        log.info(f'Loading {self.num_orders} apertures from {self.filename}')
        for i in range(self.num_orders):
            self.orders.append(mid_model.Order(self.hdul, self.filename, i))

    def _load_book(self) -> None:
        """Load all books.

        A grism observation will only have one `Book`
        in it. The book might be split across several
        extensions or combined in one. Both cases
        are handled by the `Book` class.

        """

        self.books = [mid_model.Book(self.hdul, self.filename, 0)]

    def _aperture_test(self) -> None:
        """
        Check for multiple apertures in the spectral data.

        Returns
        -------
        n_ap : int
            Number of apertures found
        """
        n_ap = 1
        if len(self.hdul) == 1:
            data_ = self.hdul[0].data
            ndim = data_.ndim
            if ndim == 3:
                n_ap = data_.shape[0]
        elif self.default_field in self.hdul:
            data_ = self.hdul[self.default_field].data
            ndim = data_.ndim
            if ndim == 2:
                n_ap = data_.shape[0]
        return n_ap

    def retrieve(self, book: Optional[int] = None,
                 order: Optional[int] = None,
                 field: str = '', level: str = 'raw') -> Optional[LowerModels]:
        """
        Access contents of lower models.

        EOS cannot access the lower models directly. Instead it
        must call this method. The arguments specify what the
        caller is expecting to get returned.

        Parameters
        ----------
        book : int, optional
            Book number to be returned.
        order : int, optional
            Order number to be returned.
        field : str, optional
            The field to retrieve. Required for 'low' and 'raw' levels.
        level : ['high', 'low', 'raw'], optional
            The level of the data to return.

        Returns
        -------
        data : Varies
            The requested data set. Can be a `Book`, `Order`,
            `Image`, `Spectrum`, or numpy array. If the
            parameters are not valid (ie no loaded data matches
            them) then None is returned.
        """
        if book is not None and order is None:
            identifier = 'book'
        elif book is None and order is not None:
            identifier = 'order'
        else:
            raise RuntimeError(f'Invalid identifier choices '
                               f'(book, order) = ({book}, {order}). '
                               f'Only one can be provided')
        if level not in ['high', 'low', 'raw']:
            raise RuntimeError(f'Invalid level choice {level}. Options are: '
                               f'"high", "low", "raw"')
        else:
            if identifier == 'book':
                if not self.books:
                    return None
                elif level == 'high':
                    return self.books[0]
                else:  # pragma: no cover
                    return self.books[0].retrieve(field=field, level=level)
            else:
                if not self.orders:
                    return None
                elif level == 'high':
                    return self.orders[order]
                else:
                    return self.orders[order].retrieve(field=field,
                                                       level=level)

    def valid_field(self, field: str) -> bool:
        """
        Determine if the field is a valid parameter.

        Parameters
        ----------
        field : str
            Name of field, such as 'wavepos' or 'flux'

        Returns
        -------
        check : bool
            True if the field is valid, False otherwise.

        """
        book_check = [book.valid_field(field)
                      for book in self.books]
        order_check = [order.valid_field(field)
                       for order in self.orders]
        check = any(book_check + order_check)
        if check:
            log.debug(f'Field {field} is valid')
        else:
            log.debug(f'Field {field} is not valid')
        return check

    def enable_orders(self, *args, **kwargs) -> None:
        """Set order visibility."""
        if self.orders:
            for order in self.orders:
                order.set_visibility(True)

    def list_enabled(self) -> Dict[str, List[int]]:
        """
        List the enabled Books and Orders.

        Returns
        -------
        full_enabled : dict
            Dictionary listing the enabled books and orders.
        """
        enabled_orders = list()
        enabled_books = list()
        for i, order in enumerate(self.orders):
            if order.enabled:
                enabled_orders.append(i)
        for i, book in enumerate(self.books):
            if book.enabled:
                enabled_books.append(i)
        full_enabled = {'orders': enabled_orders,
                        'books': enabled_books}
        return full_enabled


class MultiOrder(HighModel):
    """
    High-level model for FITS files with multiple independent spectra.

    Extends `HighModel`. Does not contain any `Book` objects,
    only `Spectrum` objects.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDU list from a spectral FITS file.
    """

    def __init__(self, hdul: pf.HDUList) -> None:
        super().__init__(hdul)

        self.default_ndims = 1
        self.num_orders = 0
        self.orders = list()

        self.load_data()

    def load_data(self) -> None:
        """Parse the input Orders."""
        self.num_orders = int(self.hdul[0].header.get('NORDERS', 1))
        log.info(f'Loading {self.num_orders} orders from {self.filename}')
        for i in range(self.num_orders):
            self.orders.append(mid_model.Order(self.hdul, self.filename, i))

    def retrieve(self, order: int = 0, field: str = '',
                 level: str = 'raw') -> Optional[LowerModels]:
        """
        Get data from lowest level.

        Parameters
        ----------
        order : int
            Index of the order to pull data from.
        field : str
            Name of the field to pull data for.
        level : ['high', 'low', 'raw']
            Sets the level to return. 'High' will return
            the Order object, 'low', will return the
            Spectrum object, and 'raw' will return the
            raw numerical data.

        Returns
        -------
        data : np.array
            Raw data
        """
        if not isinstance(order, int):
            raise TypeError('Identifier for MultiOrder must be '
                            'an integer')
        if level == 'high':
            return self.orders[order]
        else:
            if field:
                return self.orders[order].retrieve(field, level=level)
            else:
                log.debug('Need to provide field for low or raw retrievals')
                return None

    def enable_orders(self, order_flags: List[int]) -> None:
        """
        Enable specified orders, disable the rest.

        Parameters
        ----------
        order_flags : list
            List of order numbers to enable.
        """
        log.debug(f'Enable {order_flags}')
        for order in self.orders:
            if order.number in order_flags:
                log.debug(f'Setting Order {order.number} to enabled')
                order.set_visibility(True)
            else:
                log.debug(f'Setting Order {order.number} to disabled')
                order.set_visibility(False)

    def list_enabled(self) -> Dict[str, List]:
        """
        Return a list of enabled med-level objects.

        Returns
        -------
        full_enabled : dict
            Dictionary that lists the enabled Orders and Books.
        """
        enabled_orders = list()
        for i, order in enumerate(self.orders):
            if order.enabled:
                enabled_orders.append(order.number)
        full_enabled = {'orders': enabled_orders, 'books': list()}
        log.debug(f'Current enabled fields: {full_enabled}')
        return full_enabled

    def valid_field(self, field: str) -> bool:
        """
        Verify if a field is a valid option.

        Parameters
        ----------
        field : str
            Field to check.

        Returns
        -------
        order_check : bool
            True if the field is valid for any of the contained
            Orders.
        """
        order_check = [order.valid_field(field)
                       for order in self.orders]
        log.debug(f'Valid field check for {field}: {any(order_check)}')
        return any(order_check)


class ATRAN(HighModel):

    def __init__(self, hdul):
        super().__init__(hdul)

        self.default_ndims = 1
        self.default_field = 'transmission'
        self.num_orders = 0
        self.orders = list()

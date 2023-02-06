# Licensed under a 3-clause BSD style license - see LICENSE.rst

import uuid
from typing import List, Dict, Optional, TypeVar, Union
from copy import deepcopy
import astropy.io.fits as pf
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.models import mid_model, low_model
from sofia_redux.visualization.utils.eye_error import EyeError

__all__ = ['HighModel', 'Grism', 'MultiOrder']

MidModels = TypeVar('MidModels', mid_model.Order, mid_model.Book)
LowModels = TypeVar('LowModels', low_model.Image, low_model.Spectrum)
LowerModels = Union[MidModels, LowModels, np.ndarray]


class HighModel(object):
    """
    Abstract class for high-level models.

    These models correspond to a single FITS file which contain
    one or more mid-level models. They also define the methods
    by which the Eye interacts with models.

    This class does not contain method implementations.  It should
    be subclassed before using. Currently implemented subclasses are
    `Grism` and `MultiOrder`.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        FITS HDU list to load.

    Attributes
    ----------
    filename : str
        Filename associated with the HDU list.
    id : uuid.uuid4
        Unique id (uuid4) associated with each HDU list. Note that it is
        different from the id in low_model (filename).
    index : int
        Model index number.
    enabled: bool
        Indicates if a high model is enabled and thus visible in plots or not.
    default_ndims: int
        Default number of dimensions depending on the type of dataset.
    books : list
        List of mid_model.Book objects.
    orders : list
        List of mid_model.Order objects based on the number of apertures
        (orders) in a dataset.

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
        self.id = uuid.uuid4()
        self.index = 0
        self.enabled = True
        self.default_ndims = 0
        self.num_aperture = 0
        self.num_orders = 0
        self.books = list()
        self.orders = list()

        self.spectral_data_prodtypes = {
            'flitecam': ['spectra', 'spectra_1d', 'calibrated_spectrum',
                         'calibrated_spectrum_1d',
                         'coadded_spectrum', 'combined_spectrum',
                         'response_spectrum', 'instrument_response',
                         'combspec', 'calspec', 'mrgspec', 'mrgordspec'
                         ],
            'forcast': ['spectra', 'merged_spectrum', 'calibrated_spectrum',
                        'calibrated_spectrum_1d', 'spectra_1d',
                        'coadded_spectrum', 'combined_spectrum',
                        'response_spectrum', 'instrument_response',
                        'merged_spectrum_1d', 'combspec',
                        'calspec', 'mrgspec', 'mrgordspec'
                        ],
            'exes': ['calibrated_spectrum_1d', 'calspec', 'coadded_spectrum',
                     'combined_spectrum', 'combined_spectrum_1d', 'combspec',
                     'merged_spectrum_1d', 'mrgordspec', 'mrgspec',
                     'orders_merged', 'orders_merged_1d',
                     'sky_calibrated_spectrum_1d', 'sky_coadded_spectrum',
                     'sky_combined_spectrum', 'sky_combined_spectrum_1d',
                     'sky_combspec', 'sky_merged_spectrum_1d',
                     'sky_mrgordspec', 'sky_orders_merged',
                     'sky_orders_merged_1d', 'sky_spec', 'sky_specmap',
                     'sky_spectra', 'sky_spectra_1d', 'sky_wavecal_refined',
                     'sky_wavecal_refined_1d', 'spec', 'specmap', 'spectra',
                     'spectra_1d', 'wavecal_refined', 'wavecal_refined_1d']
        }

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memodict):
        cls = self.__class__
        new = cls.__new__(cls)
        memodict[id(self)] = new
        for k, v in self.__dict__.items():
            if k == 'hdul':
                # pass hdul by reference, since it is
                # not modified from initial read
                setattr(new, k, v)
            else:
                setattr(new, k, deepcopy(v, memodict))
        return new

    def load_data(self) -> None:
        """Load the data into the model."""
        raise NotImplementedError

    def retrieve(self, **kwargs) -> None:
        """Retrieve data from the model."""
        raise NotImplementedError

    def list_enabled(self) -> Dict[str, List]:
        """
        Return a list of enabled med-level objects.

        Returns
        -------
        full_enabled : dict
            Dictionary that lists the enabled Orders and Books.
            Keys are 'orders', 'books', and 'apertures'; values
            are lists of mid-level model objects.
        """
        full_enabled = {'orders': self._enabled_orders(),
                        'books': self._enabled_books(),
                        'apertures': self._enabled_apertures()}
        log.debug(f'Current enabled fields: {full_enabled}')
        return full_enabled

    def enabled_state(self) -> Dict[str, Union[bool, Dict[str, bool]]]:
        """
        Return the enabled status for all loaded data.

        Returns
        -------
        enabled : dict
            Keys are 'high' or the mid-level model names; values
            are either boolean flags or sub-dictionaries with the
            same structure with field name keys.
        """
        enabled = {'high': self.enabled}
        for collection in [self.orders, self.books]:
            for mid in collection:
                enabled[mid.name] = {'enabled': mid.enabled}
                for field, low in mid.data.items():
                    enabled[mid.name][field] = low.enabled
        return enabled

    def set_enabled_state(self, state: Dict[str, Union[bool, Dict[str, bool]]]
                          ) -> None:
        """
        Set a mid-level model to a new enabled state.

        Parameters
        ----------
        state : dict
            Keys are 'high' or the mid-level model names; values
            are either boolean flags or sub-dictionaries with the
            same structure with field name keys.

        Returns
        -------

        """
        self.enabled = state['high']
        for order in self.orders:
            order.set_enabled_state(state[order.name])
        for book in self.books:
            book.set_enabled_state(state[book.name])

    def _enabled_orders(self) -> List[int]:
        """Return order enabled status."""
        enabled = [order.number for order in self.orders
                   if order.enabled]
        return list(set(enabled))

    def _enabled_books(self) -> List[int]:
        """Return book enabled status."""
        enabled = [book.number for book in self.books
                   if book.enabled]
        return enabled

    def _enabled_apertures(self) -> List[int]:
        """Return aperture enabled status."""
        enabled = [order.aperture for order in self.orders
                   if order.enabled]
        return list(set(enabled))

    def valid_field(self, field: str) -> bool:
        """
        Determine if a field is valid for the loaded data.

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

    def enable_orders(self, order_flags: List[int],
                      enable: Optional[bool] = True,
                      aperture: Optional[bool] = False) -> None:
        """
        Enable or disable specified orders.

        Parameters
        ----------
        order_flags : list of int
            Orders to update.
        enable : bool, optional
            If True, specified orders are enabled. If False, they
            are disabled.
        aperture : bool, optional
            If True, apertures are expected. Currently has no effect.
        """
        log.debug(f'{"Enable" if enable else "Disable"} {order_flags}')
        for order in self.orders:
            if self.num_orders > 1 and self.num_aperture == 1:
                flag = order.number
            elif self.num_orders == 1 and self.num_aperture > 1:
                flag = order.aperture
            elif self.num_orders == 1 and self.num_aperture == 1:
                flag = order.number
            else:
                flag = order.number
            # if aperture:
            #     flag = order.aperture
            # else:
            #     flag = order.number
            if flag in order_flags:
                log.debug(f'Setting Order {order.number}, Aperture '
                          f'{order.aperture} to '
                          f'{"enable" if enable else "disable"}')
                order.set_visibility(enable)
            else:
                log.debug(f'Setting Order {order.number}, Aperture '
                          f'{order.aperture} to '
                          f'{"enable" if not enable else "disable"}')
                order.set_visibility(not enable)

    def extensions(self):
        """
        Retrieve all loaded extension names.

        Returns
        -------
        extensions : list of str
            All extension names present in self.hdul.
        """
        ext = list()
        for hdu in self.hdul:
            ext.append(hdu.name)
        return ext

    def ap_order_count(self):
        """
        Retrieve the number of apertures and orders loaded.

        Returns
        -------
        num_aperture, num_order : int, int
            The aperture and order count.
        """
        return self.num_aperture, self.num_orders

    def spectral_data_check(self) -> bool:
        """
        Check if an input data file contains spectral data.

        This test is based on specific product types known to contain
        spectral data.

        Returns
        -------
        bool
            True if spectral data is present; False otherwise.
        """
        instrument = self.hdul[0].header.get('INSTRUME')
        prodtype = self.hdul[0].header.get('PRODTYPE')
        if instrument is None or prodtype is None:
            return True
        else:
            instrument = str(instrument).lower()
            prodtype = str(prodtype).lower()

        if instrument == 'general':
            return True

        try:
            result = prodtype in self.spectral_data_prodtypes[instrument]
        except KeyError:
            return True
        else:
            if not result:
                raise EyeError(f'{prodtype} files for '
                               f'{instrument.upper()} do not contain '
                               f'spectral data')
            else:
                return True

    def spectral_test(self):
        """
        Check if an input data file contains spectral data.

        This test is based on the file and data structure. There
        are two possible conditions for spectral data files:

           - a 'SPECTRAL_FLUX*' extension is present
           - the first extension contains at least 1 row of data
             and not more than 5 rows of data

        Returns
        -------
        bool
            True if spectral data is present; False otherwise.
        """
        for hdu in self.hdul:
            extname = str(hdu.header.get('EXTNAME', 'UNKNOWN')).lower()
            if 'spectral_flux' in extname:
                return True

        header = self.hdul[0].header
        xdim = header.get('NAXIS1', 0)
        ydim = header.get('NAXIS2', 0)
        if xdim > 0 and ydim < 6:
            return True
        else:
            return False


class Grism(HighModel):
    """
    High level model describing a Grism data file.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDU list from a Grism FITS file.

    Attributes
    ----------
    default_field : str
        Default field for the high model, usually 'spectral_flux'.
    default_ndims : int
        Default expected data dimensionality.
    num_orders : int
        Number of orders loaded.
    """

    def __init__(self, hdul: pf.HDUList) -> None:
        super().__init__(hdul)
        self.default_field = 'spectral_flux'
        self.default_ndims = 1
        self.num_orders = 1

        self.load_data()
        log.debug(f'Initialized Grism with {len(self.books)} '
                  f'Books and {len(self.orders)} Orders.')

    def load_data(self) -> None:
        """
        Parse the contents of the HDUL into appropriate classes.

        A file is read in either on the basis of its name or the instrument.
        Some Grism files only have one extension, without associated
        image data. In this case, the `Book` object is not instantiated.
        """
        spectrum_present = self.spectral_test()
        if not spectrum_present:
            raise EyeError('No spectral data present')
        if self._spectra_only():
            self._load_order()
        elif self._image_only():
            self._load_book()
        elif self._general_only():
            self._load_order()
        else:
            self._load_order()
            self._load_book()

    def _image_only(self) -> bool:
        """Determine if there are only images in the file."""

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
        if self.filename == 'UNKNOWN':
            return True
        return False

    def _general_only(self) -> bool:
        """Identify data in non-traditional FITS or non-FITS files."""
        if 'fits' not in self.filename:
            return True

        if ('fits' in self.filename
                and self.hdul[0].header['instrume'] == 'General'):
            return True
        return False

    def _load_order(self) -> None:
        """
        Load all orders.

        A grism observation will only have one `Order`
        in it. `num_orders` should be one at the end.
        The order might be split across several
        extensions or combined in one. Both cases
        are handled by the `Order` class.

        """
        # technically, these are apertures, but we'll handle them as orders
        # for display purposes
        self.num_aperture = self._aperture_test()
        log.debug(f'Loading {self.num_aperture} apertures '
                  f'from {self.filename}')
        for i in range(self.num_aperture):
            self.orders.append(mid_model.Order(self.hdul, self.filename, 0, i))

    def _load_book(self) -> None:
        """
        Load all books.

        A grism observation will only have one `Book`
        in it. The book might be split across several
        extensions or combined in one. Both cases
        are handled by the `Book` class.
        """
        self.books = [mid_model.Book(self.hdul, self.filename, 0)]

    def _aperture_test(self) -> int:
        """
        Determine the number of apertures in the spectral data.

        Returns
        -------
        n_ap : int
            Number of apertures found
        """
        n_ap = 1
        if len(self.hdul) == 1:
            data_ = self.hdul[0].data
            if data_.ndim == 3:
                n_ap = data_.shape[0]
        elif self.default_field in self.hdul:
            data_ = self.hdul[self.default_field].data
            ndim = data_.ndim
            if ndim == 2:
                n_ap = data_.shape[0]
        return n_ap

    def retrieve(self, book: Optional[int] = None, field: str = '',
                 level: str = 'raw', order: Optional[int] = None,
                 aperture: Optional[int] = None) -> Optional[LowerModels]:
        """
        Access contents of lower level models.

        Parameters
        ----------
        book : int, optional
            Book number to be returned.
        field : str, optional
            The field to retrieve. Required for 'low' and 'raw' levels.
        level : {'high', 'low', 'raw'}, optional
            The level of the data to return.
        order : int, optional
            Order number to be returned.
        aperture : int, optional
            Aperture number to be returned.

        Returns
        -------
        data : low_model.LowModel or mid_model.MidModel or array-like
            The requested data set. Can be a `Book`, `Order`,
            `Image`, `Spectrum`, or numpy array. If the
            parameters are not valid (i.e. no loaded data matches
            them) then None is returned.
        """
        if book is not None and (order is None and aperture is None):
            identifier = 'book'
        elif book is None and (order is not None or aperture is not None):
            identifier = 'order'
        else:
            raise RuntimeError(f'Invalid identifier choices '
                               f'(book, order) = ({book}, {order}). '
                               f'Only one can be provided')
        if level not in ['high', 'low', 'raw']:
            raise RuntimeError(f'Invalid level choice {level}. Options are: '
                               f'"high", "low", "raw"')

        if identifier == 'book':
            if not self.books:
                return None
            elif level == 'high':
                return self.books[0]
            else:  # pragma: no cover
                return self.books[0].retrieve(field=field, level=level)
        else:
            if isinstance(order, str):
                try:
                    order, aperture = order.split('.')
                except ValueError:
                    order = int(order)
                    try:
                        aperture = int(aperture)
                    except (TypeError, ValueError):
                        aperture = 0
            if aperture is not None:
                index = int(order) * self.num_aperture + int(aperture)
            else:
                index = int(order)
            if not self.orders:
                return None
            elif level == 'high':
                return self.orders[index]
            else:
                if field:
                    try:
                        return self.orders[index].retrieve(field=field,
                                                           level=level)
                    except IndexError:
                        return None
                else:
                    log.debug('Need to provide field for low or raw '
                              'retrievals')
                    return None

    def valid_field(self, field: str) -> bool:
        """
        Determine if a field is valid.

        Parameters
        ----------
        field : str
            Name of field, e.g. 'wavepos' or 'flux'.

        Returns
        -------
        check : bool
            True if the field is valid, False otherwise.
        """
        book_check = [book.valid_field(field) for book in self.books]
        order_check = [order.valid_field(field) for order in self.orders]
        check = any(book_check + order_check)
        if check:
            log.debug(f'Field {field} is valid')
        else:
            log.debug(f'Field {field} is not valid')
        return check

    def list_enabled(self) -> Dict[str, List[int]]:
        """
        List the enabled Books and Orders.

        Returns
        -------
        full_enabled : dict
            Dictionary listing the enabled books and orders.
        """
        return super().list_enabled()


class MultiOrder(HighModel):
    """
    High-level model for FITS files with multiple independent spectra.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDU list from a spectral FITS file.

    Attributes
    ----------
    default_ndim : int
        Default number of dimensions.
    num_orders : int
        Number of orders in a dataset.
    books : list
        List of mid_model.Book objects.
    orders : list
        List of mid_model.Order objects.
    """

    def __init__(self, hdul: pf.HDUList, general: Optional[bool] = False
                 ) -> None:
        super().__init__(hdul)

        self.default_ndims = 1
        self.num_orders = 0
        self.books = list()
        self.orders = list()

        self.load_data(general)

    def load_data(self, general: Optional[bool] = False) -> None:
        """
        Parse the input Orders.

        Parameters
        ----------
        general : bool, optional
           If set, the input data is handled in a generic way,
           assuming all extra spectral dimensions are orders rather
           than apertures.
        """
        count = self._determine_aperture_count(general)
        if general:
            self.num_orders = count
            self.num_aperture = 1
        else:
            self.num_orders = int(self.hdul[0].header.get('NORDERS', 1))
            self.num_aperture = count

            spectrum_present = self.spectral_test()
            if not spectrum_present:
                raise EyeError('No spectral data present')

        log.debug(f'Loading {self.num_orders} orders and '
                  f'{self.num_aperture} apertures from {self.filename}')
        order_only_prodtypes = ['mrgordspec', 'orders_merged_1d',
                                'combined_spectrum_1d', 'spectra_1d',
                                'sky_orders_merged_1d', 'general',
                                'sky_combined_spectrum_1d', 'sky_spectra_1d']
        prodtype = str(self.hdul[0].header.get('PRODTYPE')).lower()
        log.debug(f'Len hdul: {len(self.hdul)}')
        if len(self.hdul) == 1:
            data = self.hdul[0].data
            header = self.hdul[0].header
            if self.num_orders == 1 and self.num_aperture == 1:
                # Case for EXES data where all orders have been
                # merged into a single array
                self.orders.append(
                    mid_model.Order(self.hdul, self.filename, 0))
                if prodtype not in order_only_prodtypes:
                    self.books.append(mid_model.Book(self.hdul,
                                                     self.filename, 0))
            elif data.shape[0] == self.num_orders:
                # Case for each order being a different slice of
                # the data array
                # if general:
                #     for i in range(self.num_aperture):
                #         self.orders.append(
                #             mid_model.Order(self.hdul, self.filename, 0, i))
                # else:
                for i in range(self.num_orders):
                    hdul = pf.HDUList(pf.ImageHDU(data[i], header))
                    self.orders.append(mid_model.Order(hdul, self.filename, i))

            elif self.num_orders * self.num_aperture == data.shape[0]:
                # Case for each order and each aperture being a different
                # slice of the data array
                for i in range(data.shape[0]):
                    order_number = i // self.num_aperture
                    ap_number = i % self.num_aperture
                    hdul = pf.HDUList(pf.ImageHDU(data[i], header))
                    self.orders.append(mid_model.Order(hdul, self.filename,
                                                       number=order_number,
                                                       aperture=ap_number))
        else:
            if self.num_orders == 1 and self.num_aperture == 1:
                # Case for EXES data where there's only one
                # order but it's been split into multiple extensions
                self.orders.append(mid_model.Order(self.hdul,
                                                   self.filename, 0))
                self.books.append(mid_model.Book(self.hdul, self.filename, 0))
            else:
                # Case for EXES data where each order gets its
                # own extension whose name ends with the order number
                for i in range(self.num_orders):
                    order_hdus = [hdu for hdu in self.hdul
                                  if hdu.name.endswith(f'_{i + 1:02d}')]
                    if len(order_hdus) == 0:
                        # Some files are split into multiple extensions
                        # but don't have the order number in their name
                        # As a last resort, try to pull extensions by name
                        # directly.
                        names = ['wavepos', 'spectral_flux', 'spectral_error',
                                 'transmission', 'response']
                        order_hdus = list()
                        for name in names:
                            try:
                                order_hdus.append(self.hdul[name])
                            except KeyError:
                                continue
                    if len(order_hdus) == 0:  # pragma: no cover
                        continue
                    args = {'hdul': order_hdus, 'filename': self.filename,
                            'number': i}
                    if self.num_aperture > 1:
                        for ap in range(self.num_aperture):
                            args['aperture'] = ap
                            args['num_apertures'] = self.num_aperture
                            self.orders.append(mid_model.Order(**args))
                            self.books.append(mid_model.Book(**args))
                    else:
                        self.orders.append(mid_model.Order(**args))
                        self.books.append(mid_model.Book(**args))

        if len(self.orders) == 0:  # pragma: no cover
            raise EyeError('No spectral data found in HDUL')

    def _determine_aperture_count(self, general: Optional[bool] = False
                                  ) -> int:
        log.debug('Determining number of apertures')
        try:
            count = self.hdul[0].header['NAPS']
            log.debug(f'Retrieved {self.num_aperture} from NAPS')
        except KeyError:
            # There is no way to reliably determine the number of apertures
            # from the header. It must be determined from the data
            # Try accessing 'spectral_flux_order_01' and getting the
            # aperture count from the number of columns.
            # No clue how to do this for FORCAST data which does not use
            # this naming structure.
            if general:
                extension = 'Primary'
            else:
                extension = 'SPECTRAL_FLUX_ORDER_01'
            try:
                data = self.hdul[extension].data
            except KeyError:
                try:
                    data = self.hdul['SPECTRAL_FLUX'].data
                except KeyError:
                    data = None
            if data is not None:
                if np.squeeze(data).ndim == 1:
                    count = 1
                else:
                    if general:
                        count = data.shape[0]
                    else:
                        count = min(data.shape)
                log.debug(f'Pulled {count} from data shape ({data.shape})')
            else:
                try:
                    positions = self.hdul[0].header['APPOSO01']
                except KeyError:
                    count = 1
                    log.debug('Defaulting to 1 aperture')
                else:
                    count = len(str(positions).strip().split(','))
                    log.debug(f'Pulled {count} from APPOSO01 ({positions})')
        return count

    def retrieve(self, order: int = 0, field: str = '',
                 level: str = 'raw', aperture: Optional[int] = None
                 ) -> Optional[LowerModels]:
        """
        Access contents of lower level models.

        Parameters
        ----------
        order : int
            Order number to be returned.
        field : str
            Name of the field to pull data for.
        level : {'high', 'low', 'raw'}
            Sets the level to return. 'High' will return
            the Order object, 'low', will return the
            Spectrum object, and 'raw' will return the
            raw numerical data.
        aperture: int, optional
            Aperture number to retrieve (zero-based indexing). If
            not included, assume the data only has a single aperture.

        Returns
        -------
        data : low_model.LowModel or mid_model.MidModel or array-like
            The requested data set. Can be a `Book`, `Order`,
            `Image`, `Spectrum`, or numpy array. If the
            parameters are not valid (i.e. no loaded data matches
            them) then None is returned.
        """
        try:
            float(order)
        except ValueError:
            raise TypeError('Identifier for MultiOrder must be '
                            'a number')
        if not self.orders:
            return None

        if isinstance(order, str):
            try:
                order, aperture = order.split('.')
            except ValueError:
                order = int(order)
                try:
                    aperture = int(aperture)
                except (TypeError, ValueError):
                    aperture = 0
        if aperture is not None:
            index = int(order) * self.num_aperture + int(aperture)
        else:
            index = int(order)
        if level == 'high':
            return self.orders[index]
        else:
            if field:
                return self.orders[index].retrieve(field, level=level)
            else:
                log.debug('Need to provide field for low or raw retrievals')
                return None

    def valid_field(self, field: str) -> bool:
        """
        Determine if a field is valid for the loaded data.

        Parameters
        ----------
        field : str
            Name of field.

        Returns
        -------
        check : bool
            True if the field is valid, False otherwise.
        """
        order_check = [order.valid_field(field)
                       for order in self.orders]
        log.debug(f'Valid field check for {field}: {any(order_check)}')
        return any(order_check)

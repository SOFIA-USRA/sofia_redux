.. currentmodule:: sofia_redux.toolkit.fits

The functions contained in :mod:`sofia_redux.toolkit.fits` can be categorized into those
that read and write FITS files, and those that operate on FITS headers.

FITS I/O
========
The FITS read and write operations written for general use and provide an
extra layer of robustness over the base :mod:`astropy.io.fits` read/write
functionality.  Files and directories are checked for accessibility and
attempts are made to continue processing in the case that corrupt data are
encountered.  Missing end cards are accounted for and any corrupt ASCII
characters will be removed from the header (on read operations).

:mod:`sofia_redux.toolkit.fits` should not raise any errors.  If an error does occur,
it was not anticipated and further more manual action should be taken.
However, expected errors will be reported through the :class:`astropy.log`
system.

Top Level I/O FITS Functions
----------------------------

- :func:`gethdul` returns the entire HDU list from a FITS file
- :func:`robust_read` returns a single data HDU and a single header HDU.  The
  default is the primary header/data array (zeroth extension).
- :func:`get_header` returns the FITS header of a single HDU
- :func:`get_data` returns the FITS data of a single HDU
- :func:`write_hdul` writes an full HDU list to file.

.. currentmodule:: sofia_redux.toolkit.fits

The majority of header functions were specifically developed for use with
SOFIA data pipeline products, performing tasks that are frequently required.

General Use Header Functions
============================
There are a few general use header functions:

:func:`get_key_value` is a very simple function that retrieves a keyword value
from a FITS header and if it is a string, removes leading/trailing whitespace
before converting it to uppercase.  If the value cannot be retrieved, a default
value is returned instead (default is the string, 'UNKNOWN'):

.. code-block:: python

    from astropy.io import fits
    from sofia_redux.toolkit.fits import get_key_value

    header = fits.Header()
    header['KEY1'] = '  this should be uppercase, no leading whitespace'
    print("Original header value: %s" % repr(header['KEY1']))
    print("Retrieved value: %s" % repr(get_key_value(header, 'KEY1')))
    print("Value for KEY2 (not in header): %s" % get_key_value(header, 'KEY2'))

Output::

    Original header value: '  this should be uppercase, no leading whitespace'
    Retrieved value: 'THIS SHOULD BE UPPERCASE, NO LEADING WHITESPACE'
    Value for KEY2 (not in header): UNKNOWN


:func:`hdinsert` acts as a more user friendly wrapper for
:func:`astropy.io.fits.Header.insert`, inserting new FITS header keywords at a
specified location.  If the key already exists in the header, a new value
and/or optional comment may be supplied.

.. code-block:: python

    from sofia_redux.toolkit.fits import hdinsert

    header = fits.Header()
    header['KEY1'] = 1, 'comment 1'
    header['KEY2'] = 2, 'comment 2'
    header['HISTORY'] = 'A history entry'
    hdinsert(header, 'KEY3', 3, comment='new comment')
    hdinsert(header, 'INSERTED', 2.5, refkey='KEY2', after=True)
    hdinsert(header, 'KEY2', header['KEY2'], comment='Replaced comment')
    header

Output::

    KEY1    =                    1 / comment 1
    KEY2    =                    2 / Replaced comment
    INSERTED=                  2.5
    KEY3    =                    3 / new comment
    HISTORY A history entry

SOFIA Specific Header Functions
===============================
The SOFIA DPS pipeline distinguishes keywords and history entries found in the
original raw data file from those added by the pipeline.  The key 'AAAAAAAA'
(eight A's) separates original and pipeline keywords while 'BBBBBBBB' (eight
B's) separates original and pipeline history entries.  These values are
hard coded into the `kref` and `href` variables in the
:mod:`sofia_redux.toolkit.fits` module where:

.. code-block:: python

    kref = 'AAAAAAAA'  # Keywords reference key marker
    href = 'BBBBBBBB'  # HISTORY reference key marker

:func:`hdinsert` is used to place pipeline specific keywords in the correct
partition by setting `refkey=kref` or `refkey=href`.  In addition,
:func:`add_history` and :func:`add_history_wrap` are wrappers for
:func:`hdinsert`, adding history entries from the pipeline at the correct
location.  For example, here is an excerpt of a SOFIA header::

    ...
    PIPEVERS= '2_0_0   '           / Pipeline version
    PRODTYPE= 'drooped '           / Product type
    AAAAAAAA= 'Keywords reference'
    RN_HIGH = '2400.   '           / Read noise for high capacitance mode
    RN_LOW  = '244.8   '           / Read noise for low capacitance mode
    BETA_G  = '1.0     '           / Excess noise
    JBCLEAN = 'MEDIAN  '           / Jail bar cleaning algorithm
    MINDROOP= '0.0     '           / minimum value for droop correction
    MAXDROOP= '65535.0 '           / maximum value for droop correction
    NRODROOP= '16      '           / number of rows for droop correction
    HISTORY keyword AOR_ID updated to NONE on Nov 27 01:23:02 2013 UTC
    BBBBBBBB= 'History reference'
    HISTORY keyword PLANID updated to NONE on Nov 27 01:23:02 2013 UTC
    HISTORY keyword OBJECT updated to Alpha_Ori on Nov 27 01:23:02 2013 UTC
    HISTORY keyword DITHER updated to F on Nov 27 01:23:02 2013 UTC
    ...
    HISTORY Droop: Channel suppression correction factor 0.003500

And here is the behavior of :func:`hdinsert` and :func:`add_history_wrap`:

.. code-block::

    from sofia_redux.toolkit.fits import kref, hdinsert, add_history_wrap

    history_function = add_history_wrap('My pipeline step')
    history_function(header, "Giving an example of add_history_wrap")
    hdinsert(header, 'NEW_KEY', 'hello')
    header

Output::

    ...
    PIPEVERS= '2_0_0   '           / Pipeline version
    PRODTYPE= 'drooped '           / Product type
    AAAAAAAA= 'Keywords reference'
    RN_HIGH = '2400.   '           / Read noise for high capacitance mode
    RN_LOW  = '244.8   '           / Read noise for low capacitance mode
    BETA_G  = '1.0     '           / Excess noise
    JBCLEAN = 'MEDIAN  '           / Jail bar cleaning algorithm
    MINDROOP= '0.0     '           / minimum value for droop correction
    MAXDROOP= '65535.0 '           / maximum value for droop correction
    NRODROOP= '16      '           / number of rows for droop correction
    NEW_KEY = 'hello   '
    HISTORY keyword AOR_ID updated to NONE on Nov 27 01:23:02 2013 UTC
    BBBBBBBB= 'History reference'
    HISTORY keyword PLANID updated to NONE on Nov 27 01:23:02 2013 UTC
    HISTORY keyword OBJECT updated to Alpha_Ori on Nov 27 01:23:02 2013 UTC
    HISTORY keyword DITHER updated to F on Nov 27 01:23:02 2013 UTC
    ...
    HISTORY Droop: Channel suppression correction factor 0.003500
    HISTORY My pipeline step: Giving an example of add_history_wrap


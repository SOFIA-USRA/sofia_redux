.. _scan_configuration:

Configuration
-------------

Default Configurations
^^^^^^^^^^^^^^^^^^^^^^

The main default configurations are located in the
`sofia_redux/scan/data/configurations` directory which contain the following
files:

  - default.cfg: The main configuration, loaded on initialization.
  - bright.cfg: Configuration updates applied to bright sources.
  - deep.cfg: Configuration updates applied for deep reductions.
  - faint.cfg: Configuration updates applied to faint sources.
  - flatfield.cfg: Configuration updates applied for flat-field reductions.

Configurations specific to instruments should be stored under
`sofia_redux/scan/data/configurations/<instrument>`, and will automatically be
applied after the main default configuration file has been read.

Format
""""""
Configuration files are parsed using
`configobj <https://configobj.readthedocs.io/en/latest/index.html>`__ and
should be formatted accordingly.  The basic structure is::

    key_with_bool_true = True
    key_with_bool_false = False
    key_with_list = 1, 2, 3, 4, 5
    key_with_range = 1:5
    key_with_list_of_ranges = 1:5, 11:14

    [options_a]
        value = foo
        [[sub_options_a]]
            bar = baz

Unlike standard configobj parsing, a key may refer to both a sub-branch of the
configuration, and also have a value assigned to it with the 'value' key.  For
example, `configuration['options_a'] == foo` and
`configuration['options_a.sub_options_a.bar'] == baz`.  Note that when
configurations are updated, the structure is updated accordingly so that any
prior values are safely stored.  For example, updating the above configuration
with `options_a.sub_options_a = 1` and `options_a.sub_options_a.bar.a = 2`
would result in::

    [options_a]
        value = foo
        [[sub_options_a]]
            value = 1
            [[[bar]]]
                value = baz
                a = 2


Order and the config command
""""""""""""""""""""""""""""
The configuration is read by generally reading the base default file followed
by an immediate update by the instrument configuration.  If the `config` key
is set to a file in the configuration directory, it will immediately be
parsed before proceeding with the rest of the current configuration.  For
example, reading the HAWC+ instrument proceeds as:

   #. default.cfg
   #. hawc_plus/default.cfg: encounters `config = sofia/default.cfg`
   #. sofia/default.cfg
   #. hawc_plus/default.cfg: continues parsing the rest of the file.

Note that if instrument specific configurations should be loaded, the
configuration file must exist at::

    sofia_redux/scan/data/configurations/<instrument>/default.cfg

and should only be referenced in other configurations using
``<instrument>/default.cfg`` without the configurations directory prefix.
Full file paths may also be supplied if necessary.

.. _configuration_conditionals:

Conditionals
""""""""""""
Whenever a configuration file has been read, a configuration validation is
performed to check the configuration conditionals.  Conditionals consist of
a requirement and set of configuration updates or instructions to perform if
that requirement is met.  All conditionals should appear in a configuration
file in the conditionals group::

    [conditionals]
        [[requirement]]
            key1 = value1
            key2 = value2

A requirement must be of the form

    - <key>: requirement is met if the configuration value for <key>
      evaluates as True
    - <key><comparison operator><value>: requirement is met if the
      configuration value for key evaluates to True using the
      <comparison operator> to check against <value>

For example::

    [conditionals]
        [[peakflux]]
            scale = 1.18

        [[fits.DIAG_HZ!=-9999]]
            notch.frequencies = {?fits.DIAG_HZ}

Would set `scale` to 1.18 if `peakflux` was True, and set
`notch.frequencies` to the value found in the FITS header, so long as it's not
equal to -9999 (see :ref:`aliases <configuration_aliases>`)


FITS configuration
""""""""""""""""""
Once a FITS scan file has been read by the reduction, all of the header
keyword values are accessed from the configuration via ``fits.<key>``.


Date
""""
Like conditionals, configuration updates may also occur based on whether the
observation occurs within certain date range::

    [date]
        [[2017-05-01--2017-05-18]]
            pointing = -0.1, -9.9

        [[*--2017-10-01]]
            jansky = 18.4

Would set the pointing correction if the observation date falls within the
given range.  The sub-heading must be of the form ``<start>--<end>`` where
<start> or <end> may be test to a YYYY-MM-DD date or ISOT datetime, or * to
indicate an open range.

Iteration
"""""""""
The configuration may be updated to apply changes based on the current
reduction iteration.  Sub-headings must be in one of the following formats:

    - Integer: Positive numbers check against a definite iteration value with
      1 indicating the first iteration.  Negative numbers a relative to the
      last iteration with -1 being the final iteration and -2 indicating the
      penultimate iteration.
    - Float: Applies configuration updates as a rounded decimal fraction of the
      total number of iterations.
    - Percentage: Applies configuration updates as a rounded percentage of the
      total number of iterations.

For example::

    [iteration]
        [[2]]
            despike.level = 30.0
            clip = 10.0

        [[-3]]
            blacklist = clip

        [[0.8]]
            add = filter.whiten

        [[90%]]
            forget = source.mem

.. _configuration_aliases:

Aliases
"""""""
Configuration keys and values may both be aliased in the configuration.
Aliased keys are generally short-hand ways of writing down a long configuration
path that may need to be frequently referenced and should appear under the
aliases branch::

    [aliases]
        sky = correlated.sky
        whiten = filter.whiten

The above example would allow the `correlated.sky` and `filter.whiten` keys
to be referenced with `sky` or `whiten` throughout the reduction and in all
other configuration files.

Values may also be aliased to point towards the value stored in a different
configuration key.  For example::

    mode = {?fits.INSTCFG}

would set the mode key value to whatever value is read from the FITS header
in a scan.  This is important when such values are added to the
configuration when a reduction has already started.


Commands
""""""""
There are several special keys that have special meaning in the
configuration.  These are:

    - blacklist :
        Never allow access to this key for any reason.  It may not
        be altered, retrieved, or made visible to the SOFSCAN reduction.  A
        blacklisted key should remain so for the entire reduction.
    - whitelist :
        Always allow access to this key and never allow any
        modification to its value for the entire reduction.
    - forget :
        Temporarily disable access to this keys value by the SOFSCAN
        reduction.  Access may be granted by using the "recall" command.
    - recall :
        Allow access to a previously forgotten key.
    - lock :
        Do not allow any further modifications to they key values or
        attributes.
    - unlock :
        Unlock a previously locked key.
    - add :
        Add a key to the configuration.  This will set the value of this
        key to "True" when retrieved by the reduction.
    - rounds :
        Set the maximum number of iterations for the SOFSCAN reduction.
        The value must be an integer (or reference an integer).
    - config :
        Read and merge the contents of another configuration whose
        file path is set as the value.



.. include:: include/glossary.rst

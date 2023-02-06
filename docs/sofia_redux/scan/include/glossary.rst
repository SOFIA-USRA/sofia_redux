.. _scan_glossary:

Configuration Glossary
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Configuration Keywords
   :class: longtable
   :widths: 20 20 40
   :header-rows: 1

   * - Keyword
     - Usage in configuration file
     - Description

   * - .. _1overf.freq:

       **1overf.freq**
     - | [1overf]
       | freq=<X>
     - The target frequency X (Hz) at which the 1/f is measured for logging
       global 1/f noise statistics.  The logged quantity is the ratio of the PSD
       measured at this frequency to that measured at the reference frequency
       `1overf.ref`_.  The actual measurement frequency will always be above the
       filter cutoff of drifts_ filtering.

   * - .. _1overf.ref:

       **1overf.ref**
     - | [1overf]
       | ref=<X>
     - The white noise reference frequency X (Hz) used when providing 1/f noise
       statistics.  If the value exceeds the Nyquist frequency of the
       timestream, then the Nyquist value will be used instead.  See
       `1overf.freq`_.

   * - .. _accel:

       | **accel**
       | Alias: correlated.accel-mag
     - | [accel]
       | <options...>
     - Can be used to enable acceleration response decorrelation, or to set
       options for it.  See `correlated.<modality>`_ for available options.

   * - .. _aclip:

       **aclip**
     - aclip=<X>
     - Clip data when the telescope acceleration is above X arcsec/s^2. Heavy
       accelerations can put mechanical energy into the detector system,
       changing the shape of the primary, thereby generating bright signals from
       the varying illumination of the bright atmosphere. Clipping data when
       there is danger of this happening is a good idea.  See accel_ for
       possible modelling of these signals.

   * - .. _add:

       **add**
     - add={<key>, <key>=<value>}, ...
     - Add a key to the configuration.  If only <key> is given, it's value will
       be set to 'True'.  Multiple keys and values may be added to the
       configuration by supplying a comma-separated list.

   * - .. _aliases:

       **aliases**
     - | [aliases]
       | <branch1>=<alias1>
       | <branch2>=<alias2>
       | ...
     - The [aliases] section specifies user defined convenient shorthand
       notations for configuration keywords.  For example, if one defines the
       alias sky=correlated.sky, then sky.gainrange will actually
       reference correlated.sky.gainrange for any configuration operation.
       Aliases may also reference other aliases, so sg=sky.gainrange would allow
       sg to reference correlated.sky.gainrange.

   * - .. _altaz:

       | **altaz**
       | Sets: system=horizontal
     - altaz={True, False}
     - A conditional switch to reduce in Alt/Az coordinates.  See system_.

   * - .. _array:

       | **array**
       | Alias for:  correlated.obs-channels
     - | [array]
       | <options...>
     - An alias for all radiation-sensitive channels of the instrument, or set
       options for it.  See `correlated.<modality>`_ for further details.

   * - .. _atran.altcoeffs:

       | **atran.altcoeffs**
       | Instrument: SOFIA
     - atran.altcoeffs=<c0>,<c1>,<c2>,...<cN>
     - The polynomial coefficients used to determine the altitude factor when
       determining the atmospheric transmission correction.  Used to fit for
       an altitude relative to 41 kft in units of kft.

   * - .. _atran.amcoeffs:

       | **atran.amcoeffs**
       | Instrument: SOFIA
     - atran.amcoeffs=<c0>,<c1>,<c2>,...<cN>
     - The polynomial coefficients used to determine the air mass factor when
       determining the atmospheric transmission correction.  Used to fit for
       the air mass relative to sqrt(2) (an elevation of 45 degrees).

   * - .. _atran.reference:

       | **atran.reference**
       | Instrument: SOFIA
     - atran.reference=<X>
     - The factor (f) used to provide the actual transmission value when
       multiplied by the transmission correction value (c).  The
       transmission (t) is given as t = f * c where c = am_factor * alt_factor
       (see `atran.altcoeffs`_ and `atran.amcoeffs`_).  The transmission is
       related to the opacity (tau) by t = exp(-tau * airmass).

   * - .. _beam:

       **beam**
     - beam=<X>
     - Set the instrument beam to X arcseconds.  Also see resolution_.

   * - .. _beammap:

       | **beammap**
       | Sets: pixelmap=True
     - beammap={True, False}
     - A conditional switch that sets pixelmap_ to True.

   * - .. _blacklist:

       **blacklist**
     - blacklist=<key1>,<key2>,...
     - Similar to forget_, except it will not set options even if they are
       specified at a later time.  This is useful for altogether removing
       settings from the configuration.

   * - .. _blank:

       **blank**
     - blank=<X>
     - Skip data from modelling over points that have a source flux exceeding the
       signal-to-noise level X.  This may be useful in reducing the filtering
       effect around bright  See clip_.

   * - .. _blind:

       **blind**
     - blind=<range list>
     - Specify a list of blind pixels.  Use data indices and ranges in a
       comma-separated form.  Blind channels may be used by some instruments to
       estimate instrumental signals, such as temperature fluctuations.
       Channels are numbered from 0 (C-style).  See flag_.  <range list> is
       a comma-separated list of individual channels or channel ranges.  For
       example:

         blind=10,15:18,33

       Would blind channels 10, 15, 16, 17, 18, and 33.

   * - .. _bright:

       | **bright**
       | Sets: config=bright.cfg
     - bright={True, False}
     - Use for bright sources (S/N > ~1000).  This setting entirely bypasses all
       filtering to produce a very faithful map.  The drawback is more noise.
       See config_, faint_, and deep_.

   * - .. _chopped:

       **chopped**
     - chopped={True, False}
     - Used for specifying a chopped data reduction.  Can be set manually or
       automatically based on the data itself.  The key
       may trigger conditional statements and extra decorrelation steps.
       See `correlated.<modality>.trigger`_.

   * - .. _chopper.invert:

       | **chopper.invert**
       | Instrument: HAWC+
     - chopper.invert={True, False}
     - An option to flip the direction associated with the analog chopper R/S
       signals.

   * - .. _chopper.shift:

       | **chopper.shift**
       | Instrument: HAWC+
     - chopper.shift=<N>
     - Shift the chopper R/S analog signals by N raw frames (sampled at
       203.25 Hz), relative to the detector readout to improve synchronization.
       See shift_.

   * - | **chopper.tolerance**
       | Instrument: HAWC+
     - chopper.tolerance=<X>
     - Allow setting a tolerance for the chopper position in arcseconds.  If the
       actual chopper distance is not within the tolerance from the nominal
       chopper amplitude, then the exposure will not be used to avoid smearing.

   * - .. _clip:

       **clip**
     - clip=<X>
     - In early generations of the source map, force map pixels with flux below
       signal-to-noise level X to zero.   This may help getting lesser
       baselines, and filtering artifacts around the brighter peaks.  Often used
       together with blank_ in the intermediate iterations.  See blank_ and
       iteration_.

   * - .. _cols:

       | **cols**
       | Alias: correlated.cols
     - | [cols]
       | <options...>
     - An alias for column based decorrelation of the detector array.  Used to
       perform decorrelation, or set decorrelation options.

   * - .. _commonwcs:

       **commonwcs**
     - commonwcs={True, False}
     - If the reduction consists of multiple sub-reductions (e.g. a sub
       reduction for each HAWC+ subarray), specify that the output map for all
       reductions should share a common WCS and equivalent dimensions.

   * - .. _conditionals:

       **conditionals**
     - | [conditionals]
       | [[<requirement>]]
       | <key1>=<value1>
       | ...
     - Used to set configuration values in specific circumstances.  Multiple
       key=value settings can be applied under each requirement once that
       requirement has been fulfilled.  Requirements should take the form
       [[<keyword>]] or [[<keyword><operator><value>]].  The first will apply
       settings should that keyword be set in the configuration.  The more
       complex alternative involves comparing one configuration keyword value
       with another in the requirement, and apply all settings if evaluated as
       true.  <operator> can be one of =, !=, <, <=, >, or >=.

   * - .. _config:

       **config**
     - config=<filename>
     - Load a configuration file filename.  Files are looked for in the
       following order from lowest to highest priority in the
       sofia_scan/scan/data/configurations folder (<c>) and a optional user
       configuration directory (~/.sofscan):

       1. <c>/<filename>
       2. ~/.sofscan/<filename>
       3. <c>/<instrument>/<filename>
       4. ~/.sofscan/<instrument>/<filename>

       Whenever a matching file is found, its contents are parsed.  Because of
       the ordering, it is convenient to create overriding configurations.  Each
       successively loaded file may override the options set before it.
       See bright_, faint_, and deep_.

   * - .. _correlated.<modality>:

       **correlated.<modality>**
     - | [correlated]
       | [[<modality>]]
       | <key>=<value>
       | ...
     - Remove the correlated noise term across the entire array where <modality>
       is the name of the modality on which decorrelation is performed.  E.g.
       'obs-channels' or 'gradients'.  This is an effective way of dealing with
       most atmospheric and instrumental signals, such as sky noise, ground
       pickup, temperature fluctuations, electromagnetic or microphonic pickups.
       The decorrelation of each modality can be further controlled by a number
       of <key>=<value> settings (see below).  The given decorrelation step must
       also appear in the pipeline ordering_ before it can be used.  See
       `division.<name>`_ and ordering_.

   * - .. _correlated.<modality>.gainrange:

       **correlated.<modality>. gainrange**
     - | [correlated]
       | [[<modality>]]
       | gainrange=<min>:<max>
     - Specify a range of acceptable gains to the given correlated signal
       <modality>, relative to the average gain response of the correlated mode.
       Channels that exhibit responses outside of this range will be
       appropriately flagged in the reduction, and ignored in the modelling
       steps until the flag is revised and cleared in another decorrelation
       step.  See `division.<name>.gainflag`_ and
       `correlated.<modality>.signed`_.

   * - .. _correlated.<modality>.nofield:

       **correlated.<modality>. nofield**
     - | [correlated]
       | [[<modality>]]
       | nofield={True, False}
     - Allow decoupling of the gains of the correlated mode from the gain fields
       stored under the channel (initialized from the file specified by
       pixeldata_).  See pixeldata_ and `source.fixedgains`_.

   * - .. _correlated.<modality>.nogains:

       **correlated.<modality>. nogains**
     - | [correlated]
       | [[<modality>]]
       | nogains={True, False}
     - Disable the solving of gains (i.e. channel responses) to the correlated
       signal <modality>.

   * - .. _correlated.<modality>.nosignals:

       **correlated.<modality>. nosignals**
     - | [correlated]
       | [[<modality>]]
       | nosignals={True, False}
     - Disable solving for the correlated signal <modality> whose value stays
       fixed.

   * - .. _correlated.<modality>.phases:

       **correlated.<modality>. phases**
     - | [correlated]
       | [[<modality>]]
       | phases={True, False}
     - Decorrelate the phase data (e.g. for chopped photometry scans) together
       with the fast samples.  The same gains are used as for the usual
       decorrelation on the fast samples.

   * - .. _correlated.<modality>.phasegains:

       **correlated.<modality>. phasegains**
     - | [correlated]
       | [[<modality>]]
       | phasegains={True, False}
     - Determine the gains from the phase data, rather than from the correlated
       fast samples.  You can also set this globally for all correlated
       modalities/modes using the phasegains_ keyword.  See phasegains_.

   * - .. _correlated.<modality>.resolution:

       **correlated.<modality>. resolution**
     - | [correlated]
       | [[<modality>]]
       | resolution=<X>
     - Set the time resolution (in seconds) for the decorrelation of <modality>.
       When dealing with 1/f-type signals, you probably want to set this to the
       1/f knee time-scale or below if you want optimal sensitivities.  Else,
       you may want to try larger values if you want to recover more large-scale
       emission and are not too worried about the loss of sensitivity.  See
       extended_.

   * - .. _correlated.<modality>.signed:

       **correlated.<modality>. signed**
     - | [correlated]
       | [[<modality>]]
       | signed={True, False}
     - by default, gain responses are allowed to be bidirectional, and flagging
       affects only those channels or pixels, where absolute gain values fall
       outside of the specified range.  When 'signed' is set, the gains are
       flagged with the signs also taken into account.  I.e., under 'signed',
       'gainrange' or '0.3:3.0' would flag pixels with a gain of -0.8, whereas
       the default behaviour is to tolerate them.  See
       `correlated.<modality>.gainrange`_ and `correlated.<modality>.nogains`_.

   * - .. _correlated.<modality>.span:

       **correlated.<modality>. span**
     - | [correlated]
       | [[<modality>]]
       | span={True, False}
     - Make the gains of the correlated modality span scans instead of
       integrations (subscans).  You can also set this option for all correlated
       modalities at once using the `gains.span`_ key.

   * - .. _correlated.<modality>.trigger:

       **correlated.<modality>. trigger**
     - | [correlated]
       | [[<modality>]]
       | trigger=<requirement>
     - You can specify a configuration key that is to serve as a trigger for
       activating the decorrelation of <modality>.  This is used, for example,
       to activate the decorrelation of chopper signals, if and when the
       chopped_ keyword is specified.  <requirement> may take the form <key>
       or <key><operator><value>.  If a single <key> is specified, the trigger
       will activate if the retrieved value from the configuration evaluates to
       True.  Otherwise <operator> (!=, =, <, <=, >, >=) may be used to check
       a value in the configuration against <value>.

   * - .. _correlated.<*>:

       **correlated.<*>**
     - correlated.*.gainrange=0.3:3.0
     - You can use wildcards '*' to set options for all decorrelation steps at
       once.  The above example sets the `correlated.<modality>.gainrange`_
       value for all currently defined branches (and modalities) to 0.3:3.

   * - .. _crushbugs:

       **crushbugs**
     - crushbugs={True, False}
     - Allow SOFSCAN to replicate some of the most prominent bugs found in the
       original CRUSH.  These bugs currently include:

       1. Double adding of frame (time) dependents for FFT fixed filters
          (see filter_).
       2. Adding frame (time) dependents N times rather than once during
          integration syncing with the source model, where N is the number of
          channels.

       The above issues become noticeable after many iterations (see rounds_)
       since the fraction by which dependents change are usually very small.
       However, after a while you may notice some data being flagged
       unnecessarily.  There is a significant bug that has not been covered by
       crushbugs_ in which the real and imaginary interleaved FFT spectrum
       (realf0, imagf0, realf1, imagf1, realf2...), as determined by the filter_
       step, is subtracted from the timestream in addition to it's inverse
       transform (correct method of removal).

   * - .. _darkcorrect:

       | **darkcorrect**
       | Instrument: HAWC+
     - darkcorrect={True,False}
     - Whether to perform the squid dark correction for blind channels.
       Otherwise, all blind channels will be flagged as dead.

   * - .. _datapath:

       **datapath**
     - datapath=<directory>
     - Look for raw data to reduce in the directory <directory>.

   * - .. _dataunit:

       **dataunit**
     - dataunit=<name>
     - Specify the units in which the data are stored.  Typically, 'counts' or
       'V', or any of their common multiples such as 'mV', 'uV' or astropy.units
       unit types are accepted.  The conversion from data units to Jansky-based
       units is set via the jansky_ option, while the choice of units in the
       data reduction is set be unit_.

   * - .. _date:

       **date**
     - | [date]
       | [[<start>--<end>]]
       | <key>=<value>
       | ...
     - A way to set date specific conditional statements.  <start> and <end>
       can be specified as ISOT strings or float MJD values, both in the UTC
       scale.  Wildcards ('*') may also be used to unbound the start or end
       time.  E.g.:

       | [date]
       | [[2021-12-14T10:00:00--`*`]]
       | instrument.gain=-1000
       | chopped=True

       would set the instrument gain to -1000, and indicate chopped observations
       for any time after 10:00 UTC on December 12, 2021.

   * - .. _deep:

       | **deep**
       | Sets: config=deep.cfg
     - deep={True, False}
     - Use for very faint sources which are not all detected in single scans, or
       if you think there is too much residual noise (baselines) in the map.
       This setting results in the most aggressive filtering and will load the
       configuration from 'deep.cfg'.  The output map is optimally filtered
       (smoothed) for point sources.  See config_, bright_, and faint_.

   * - .. _dejump:

       **dejump**
     - | [dejump]
       | <options...>
     - Used to specify options for the 'dejump' task which identifies places in
       the data stream where detectors jump together (especially SQUIDs under a
       transient B-field fluctuation) by the perceived increase in residual
       detector noise.  Sub-settings are `dejump.level`_ and
       `dejump.minlength`_.  This will only occur if 'dejump' appears in
       ordering_.

   * - .. _dejump.level:

       **dejump.level**
     - dejump.level=<X>
     - The relative noise level at which jumps are identified.  The value should
       be strictly greater than 1, with 2.0 being a safe starting point.  Change
       with extreme caution, if at all.  See dejump_.

   * - .. _dejump.minlength:

       **dejump.minlength**
     - dejump.minlength=<X>
     - The minimum length (in seconds) of a coincident detector jump that is
       kept alive in the data.  Jumps longer than this threshold will be
       re-levelled, wheras shorted jumps will be flagged out entirely.  See
       dejump_.

   * - .. _derive:

       | **derive**
       | Sets:
       | forget = pixeldata, vclip, aclip
       | blacklist = whiten
       | write.pixeldata = True
       | rounds = 30
     - derive={True, False}
     - A conditional switch which when activated will perform a reduction
       suitable for deriving pixel data.  See `write.pixeldata`_.

   * - .. _despike:

       **despike**
     - | [despike]
       | <options...>
     - Used to define despiking options.  SOFSCAN allows the use of up to three
       different spiking steps, each configurable on its own.  In order to be
       enabled, 'despike' must be specified in ordering_.  To specify a
       despiking method, S/N levels and flagging criteria, please see the
       various despiking options below.

   * - .. _despike.blocks:

       **despike.blocks**
     - despike.blocks={True, False}
     - Flag out an entire 'drifts' block of data around any spikes found.  This
       is probably an overkill in most cases, but may be useful if spikes are
       due to discontinuities (jumps) in individual detectors.  See drifts_.

   * - .. _despike.flagcount:

       **despike.flagcount**
     - despike.flagcount=<N>
     - Tolerate (without pixel flagging) up to N spikes in each pixel.

   *  - .. _despike.flagfraction:

        **despike.flagfraction**
      - dispike.flagfraction=<X>
      - Tolerate (without pixel flagging) spikes up to fraction X of the scan
        frames in each channel.

   * - .. _despike.framespikes:

       **despike.framespikes**
     - despike.framespikes=<N>
     - Tolerate up to N spikes per frame.

   * - .. _despike.level:

       **despike.level**
     - despike.level=<X>
     - Despike at an S/N level of X.

   * - .. _despike.method:

       **despike.method**
     - despike.method=<name>
     - SOFSCAN offsets a choice of despiking methods to choose from.  Each of
       these have their own pros and cons, and may produce different results and
       side effects in different environments.  The following methods are
       currently available:

       - *neighbours*: Despike by comparing neighbouring samples of data from
         the same channel.
       - *absolute*: Flag data that deviates by the specified S/N level
         (`despike.level`_).
       - *gradual*: Like *absolute* but proceeds more cautiously, removing only
         a fraction of the most offending spikes at each turn.
       - *multires*: Look for spikes wider than just a single sample.

       All methods will flag pixels and frames if these have too many spikes.
       The flagging of spiky channels and frames is controlled by the
       `despike.flagcount`_, `despike.flagfraction`_, and `despike.framespikes`_
       keys.

   * - .. _division.<name>:

       **division.<name>**
     - | [division]
       | [[<name>]]
       | value=<group1>,<group2>,...
     - An option to specify user-defined channel divisions containing specific
       channel groups.  This may be useful when creating a new modality.  All
       named groups must be available in the reduction in order to be included
       in the <name> division.  A channel division contains all channel groups
       relating to a modality of the same name.  See `correlated.<modality>`_,
       `division.<name>.gainfield`_, `division.<name>.gainflag`_,
       `division.<name>.id`_, and group_.

   * - .. _division.<name>.gainfield:

       **division.<name>.gainfield**
     - | [division]
       | [[<name>]]
       | gainfield=<attribute>
     - Specify which attribute of the channel data such as 'coupling' or
       'nonlinearity' should be used to provide gain values for the correlated
       modality <name>.  See `correlated.<modality>`_ and `division.<name>`_.

   * - .. _division.<name>.gainflag:

       **division.<name>.gainflag**
     - | [division]
       | [[<name>]]
       | gainflag={<N>, <flag>}
     - Set the gain flag used for flagging out-of-range gain values for the
       correlated modality <name>.  An integer (<N>) or flag name (<flag>) may
       be specified.  Take care if using an integer to ensure its value matches
       the desired flag.  If not specified, the default is 'GAIN'.

   * - .. _division.<name>.id:

       **division.<name>.id**
     - | [division]
       | [[<name>]]
       | id=<ID>
     - Specify a shorthand ID for the modality <name>.  This is usually a
       two-letter abbreviation of <name>.  If not supplied, defaults to <name>.

   * - .. _downsample:

       **downsample**
     - downsample={N, auto}
     - Downsample the data by a factor of N.  At times the raw data is sampled
       at unnecessarily high frequencies.  By downsampling, you can ease the
       memory requirement and speed up the reduction.  You can also set the
       value to 'auto' (default), in which case an optimal downsampling rate is
       determined based on the typical scanning speeds so that the loss of
       information will be insignificant due to unintended smearing of the data.

   * - .. _drifts:

       **drifts**
     - drifts={X, max, auto}
     - Filter low frequencies below the characteristic timescale of X seconds as
       an effective way of dealing with 1/f noise.  You can also use 'auto'
       to determine the filtering timescales automatically, based on
       sourcesize_, scanning speeds and instrument stability_ time-scales.  The
       'max' value is also accepted, producing results identical to that of
       offsets_.

   * - .. _ecliptic:

       | **ecliptic**
       | Sets: system=ecliptic
     - ecliptic={True, False}
     - Reduce using ecliptic coordinates (for mapping).

   * - .. _equatorial:

       | **equatorial**
       | Sets: system=equatorial
     - equatorial={True, False}
     - Reduce using equatorial coordinates (for mapping).

   * - .. _estimator:

       **estimator**
     - estimator={median, maximum-likelihood}
     - The estimator to use in deriving signal models.  'median' estimators are
       less sensitive to the presence of bright sources in the data, therefore
       it is the default for when bright_ is specified (see 'bright.cfg').
       When medians are used, the corresponding models are reported on the log
       output in square brackets ([]).  See `gains.estimator`_ and
       `weighting.method`_.

   * - .. _exposureclip:

       **exposureclip**
     - exposureclip=<X>
     - Flag (clip) map pixels whose relative time coverage is less than the
       specified value X.  This is helpful for discarding the underexposed noisy
       edges of the map.  See noiseclip_ and clip_.

   * - .. _extended:

       **extended**
     - extended={True, False}
     - Try to better preserve extended structures.  This setting can be used
       alone or in combination with brightness options.  For bright structures
       recovery up to FOV (or beyond) should be possible.  Faint structures
       ~1/4 FOV to ~FOV scales are maximally obtainable.  See sourcesize_,
       bright_, faint_, and deep_.

   * - .. _faint:

       | **faint**
       | Sets: config=faint.cfg
     - faint={True, False}
     - Use with faint sources (S/N < ~30) when the source is faint but still
       visible in a single scan.  This setting applies some more aggressive
       filtering of the timestreams, and extended structures.  It will result
       in applying the configuration settings found in 'faint.cfg'.  See bright_
       and deep_.

   * - .. _fifi_ls.insert_source:

       **fifi_ls.insert_source**
     - | [fifi_ls]
       | insert_source={True, False}
     - Used in conjunction with `fifi_ls.resample`_.  If True, the source
       model is injected back into the irregular frame data.  If False, the
       detected correlations and drifts are removed from the original frame
       data.  If using a filter, it is advisable to set this parameter to
       True, as the filtered signals cannot be removed from the original data.

   * - .. _fifi_ls.resample:

       **fifi_ls.resample**
     - | [fifi_ls]
       | resample={True, False}
     - If set to True, and reducing FIFI-LS data, instructs the reduction to
       perform a few additional steps post-reduction.  This is to set the
       irregular frame data to a state where it can then be manually passed
       into a more robust resampler to generate a final output map, rather
       than using the default nearest neighbor method.  Please see
       `fifi_ls.insert_source`_ for more details.

   * - .. _fifi_ls.uncorrected:

       **fifi_ls.uncorrected**
     - | [fifi_ls]
       | uncorrected={True, False}
     - If set to True, and reducing FIFI-LS data, instructs the reduction to
       use the uncorrected wavelength, data, and error values present in the
       UNCORRECTED_LAMBDA, UNCORRECTED_FLUX, and UNCORRECTED_STDDEV HDUs rather
       than the LAMBDA, FLUX, and STDDEV HDUs.

   * - .. _fillgaps:

       **fillgaps**
     - fillgaps={True, False}
     - Fill any gaps in the timestream data with empty frames so that time
       windows in the reduction work as expected and that no surprise
       discontinuities can cause real trouble.

   * - .. _filter:

       **filter**
     - | [filter]
       | value={True, False}
     - Activate spectral filtering of timestreams.  The filter components are
       set by `filter.ordering`_ and can be configured and activated separately.
       See `crushbugs`_, `filter.ordering`_, `filter.motion`_, `filter.kill`_,
       and `filter.whiten`_.

   * - .. _filter.kill:

       **filter.kill**
     - | [filter]
       | [[kill]]
       | value={True, False}
     - Allows completely quenching certain frequencies in the timestream data.
       To activate, both this option and the filter_ umbrella option must
       evaluate as True.  The bands of the kill-filter are set by
       `filter.kill.bands`_.

   * - .. _filter.kill.bands:

       **filter.kill.bands**
     - | [filter]
       | [[kill]]
       | bands=<f1>:<f2>, <f3>:<f4>, ...
     - Provide a comma-separated list of frequency ranges (Hz) that are to be
       quenched by the kill filter.  E.g.:

         filter.kill.bands=0.35:0.37,9.8:10.2.

       See filter_ and `filter.kill`_.

   * - .. _filter.motion:

       **filter.motion**
     - | [filter]
       | [[motion]]
       | value={True, False}
     - The (typically) periodic motion of the scanning can induce vibrations in
       the telescope and instrument.  Since these signals will be in sync with
       the scanning motion, they will produce definite mapping artifacts (e.g.
       broad pixels near the map edges).  The motion filter lets you perform
       spectral filtering on those frequencies where most of the scanning motion
       is concentrated.  To activate, bot this option and the filter_ umbrella
       options must be set.  The identification of rejected motion frequencies
       is controlled by the `filter.motion.s2n`_ `filter.motion.above`_, and
       `filter.motion.range`_ sub-keys.

   * - .. _filter.motion.above:

       **filter.motion.above**
     - | [filter]
       | [[motion]]
       | above=X
     - The fraction, relative to the peak spectral component of the scanning
       motion, above which to filter motion.  E.g.:

         filter.motion.above=0.1

       will identify components that are at least 10% of the main component
       amplitude.  See `filter.motion`_, `filter.motion.s2n`_, and
       `filter.motion.range`_.

   * - .. _filter.motion.harmonics:

       **filter.motion.harmonics**
     - | [filter]
       | [[motion]]
       | harmonics=<N>
     - Kill not just the dominant motion frequencies, but also up to N harmonics
       of these.  This may be useful when the motion response is non-linear.
       Otherwise, it's an overkill.  See `filter.motion.odd`_.

   * - .. _filter.motion.odd:

       **filter.motion.odd**
     - | [filter]
       | [[motion]]
       | odd={True, False}
     - When set, together with the `filter.motion.harmonics`_ setting, this
       option instructs SOFSCAN to restrict the motion filter to the odd
       harmonics only of the principle frequencies of the scanning motion.
       See `filter.motion.harmonics`_.

   * - .. _filter.motion.range:

       **filter.motion.range**
     - | [filter]
       | [[motion]]
       | range=<min>:<max>
     - Set the frequency range (Hz) in which the motion filter operates.  See
       `filter.motion`_, `filter.motion.above`_, and `filter.motion.s2n`_.

   * - .. _filter.motion.s2n:

       **filter.motion.s2n**
     - | [filter]
       | [[motion]]
       | s2n=<X>
     - The minimum significance of the motion spectral component to be
       considered for filtering.  See `filter.motion`_, `filter.motion.above`_,
       and `filter.motion.range`_.

   * - .. _filter.motion.stability:

       **filter.motion.stability**
     - | [filter]
       | [[motion]]
       | stability=<X>
     - Define a stability timescale (seconds) for the motion response.  When not
       set, it is assumed that the detectors respond to the same amount to the
       vibrations induced by the scanning motion during the entire duration of a
       scan.  If a timescale shorter than the scan length is set, then the
       filtering will become more aggressive to incorporate the AM modulation of
       detector signals on timescales shorter than this stability value.  See
       `filter.motion.range`_ and `filter.motion.stability`_.

   * - .. _filter.mrproper:

       **filter.mrproper**
     - | [filter]
       | mrproper={True, False}
     - Enables the explicit re-levelling of the filtered signal.  In practice,
       the re-levelling is unlikely to significantly improve the filter's
       effectiveness.  At the same time, it does slow it down somewhat, which is
       why it is off by default.

   * - .. _filter.ordering:

       **filter.ordering**
     - | [filter]
       | ordering=<filter1>,<filter2>,...
     - A comma-separated list of spectral filters, in the order they are to be
       applied.  The default is 'motion, kill, whiten' which firstly applies the
       motion filter, then kills specified spectral bands, and finally applies
       noise whitening on the remainder.  Each of the components can be
       controlled separately with the appropriate sub-keys of filter_ with the
       same names.  See `filter.motion`_, `filter.whiten`_, and `filter.kill`_.

   * - .. _filter.whiten:

       **filter.whiten**
     - | [filter]
       | [[whiten]]
       | value={True, False}
     - Use a noise whitening algorithm.  White noise assures that the noise in
       the map is independent pixel-to=pixel.  Otherwise noise may be correlated
       on specific scales.  Whitening is also useful to get rid of any signals
       (still) not modelled by other reduction steps.  It should always be a
       last resort only, as the modeling of signals is generally preferred.  To
       activate, both this option and the filter_ umbrella option must evaluate
       to True.  See filter_, whiten_, `filter.whiten.level`_,
       `filter.whiten.minchannels`_, and `filter.whiten.proberange`_.

   * - .. _filter.whiten.level:

       **filter.whiten.level**
     - | [filter]
       | [[whiten]]
       | level=<X>
     - Specify the noise whitening level at X times the average (median)
       spectral noise level.  Spectral channels that have noise in excess of the
       critical level will be appropriately filtered to bring them back in line.
       Value clearly above 1 are recommended, and typically values around
       1.5-2 are useful without over filtering.  See `filter.whiten`_.

   * - .. _filter.whiten.minchannels:

       **filter.whiten.minchannels**
     - | [filter]
       | [[whiten]]
       | minchannels=<N>
     - Make sure that at least N channels are used for estimating the white
       noise levels, even if the specified probe range is smaller of falls
       outside of the available spectrum.  In such cases, SOFSCAN will
       automatically expand the requested range to include at least N spectral
       channels, or as many as possible if the spectral range itself is too
       small.  See `filter.whiten`_ and `filter.whiten.proberange`_.

   * - .. _filter.whiten.proberange:

       **filter.whiten.proberange**
     - | [filter]
       | [[whiten]]
       | proberange={<from>:<to>, auto}
     - Specify the spectral range (Hz) in which to measure the white noise level
       before whitening.  It is best to use the truly flat part of the available
       spectral range where no 1/f, resonances, or lowpass roll-off are present.
       Wildcards ('*') can be used for specifying open ranges.  'auto` can be
       used to automatically adjust the probing range to the upper part of the
       spectrum occupied by point sources.  See `filter.whiten`_ and
       `filter.whiten.minchannels`_.

   * - .. _final:

       | **final**
       | Alias: iteration.-1
     - | [final]
       | <key>=<value>
       | ...
     - An alias for settings to be applied on the last iteration.  See last_.

   * - .. _fits.<key>:

       **fits.<key>**
     - <configuration_key>={?fits.<key>}
     - A way to reference FITS header keyword values from the configuration.
       For example:

         intcalfreq={?fits.DIAG_HZ}

       will always retrieve 'intcalfreq' in the configuration from the 'DIAG_HZ'
       key in the FITS header.

   * - .. _fits.addkeys:

       | **fits.addkeys**
       | Telescope: SOFIA
     - | [fits]
       | addkeys=<key1>,<key2>,...
     - Specify a comma-separated list of keys that should be migrated from the
       first scan to the image header, in addition to the list of required SOFIA
       header keys.

   * - .. _fixjumps:

       | **fixjumps**
       | Instrument: HAWC+
     - | [fixjumps]
       | value={True, False}
     - Attempt to 'fix' residual flux jumps that result from imprecise
       correction in the MCE.  Long jumps are re-levelled, while shorter ones
       are flagged out to minimize impact on source structure.  Alternatively,
       the same can be applied on a per-subarray basis as well as via the
       `fixjumps.<sub>`_ option.

   * - .. _fixjumps.detect:

       | **fixjumps.detect**
       | Instrument: HAWC+
     - | [fixjumps]
       | detect = <X>
     - If `fixjumps`_ is set to True, attempt to locate and correct any
       unreported jumps in the data.  <X> is a threshold value used to locate
       possible jumps such that diff = d - shift(d, 1), mad = medabsdev(diff),
       and possible jumps occur at abs(diff) >= <X> * mad.

   * - .. _fixjumps.<sub>:

       | **fixjumps.<sub>**
       | Instrument: HAWC+
     - | [fixjumps]
       | <sub> = {True, False}
     - The same as fixjumps_ but performed on a per-subarray basis.  <sub>
       may be currently one of {r0, r1, t0, t1}.

   * - .. _flag:

       **flag**
     - | [flag]
       | <field>=<list>
       | ...
     - Flag channels based on ranges of values or values within certain ranges.
       Here, <field> refers to a specific attribute of the channel data on which
       to base the flagging.  For example:

       | [flag]
       | col=10,20:22
       | pin_gain=-1:0

       Would flag channel columns 10, 20, 21, and 22 and any channels where
       pin gain is between -1 and 0.  All such channels will be flagged as
       'DEAD' and this process occurs only once following a scan read.  Note
       that <list> may contain range elements with `*` marking an open bound.
       the colon (:) is preferred over hyphen (-) to mark ranges in order to
       effectively distinguish negative numbers, although a hyphen will still
       work as expected for purely positive values.

   * - .. _flatweights:

       **flatweights**
     - flatweights={True, False}
     - Override the channel weights from pixeldata_ with their average value.
       This way all channels carry the same uniformed initial weight.  It can be
       useful when the pixeldata_ weights are suspect for some reason.

   * - .. _focalplane:

       | **focalplane**
       | Sets: system=focalplane
     - focalplane={True, False}
     - Produce maps in focal-plane coordinates.  This is practical only for
       beam-mapping.  Thus, focal-plane coordinates are default when
       `source.type`_ is set to 'pixelmap'.  See pixelmap_ and `source.type`_.

   * - .. _focus.<direction>coeff:

       **focus.<direction>coeff**
     - focus.<direction>coeff=<X>
     - Used to convert the asymmetry and elongation parameters of an elliptical
       model of the source to focus values (in mm) using focus=-1/coeff * param
       where coeff is the value supplied here, and param is the asymmetry x or
       y factor for directions x and y, and param is the elongation factor for
       the z direction.  <direction> may take values of x, y, or z.

   * - .. _focus.<direction>scatter:

       **focus.<direction>scatter**
     - focus.<direction>scatter=<X>
     - Adds extra noise to the reported focus measurements in the x, y, and/or
       z <direction>.  RMS values should be provided in units of mm.

   * - .. _focus.significance:

       **focus.significance**
     - focus.significance=<X>
     - Require focus calculation factors (asymmetry and elongation) to have a
       signal-to-noise ratio of greater than <X> in order for the focus results
       to be reported in the x, y, and z directions.

   * - .. _focus.elong0:

       **focus.elong0**
     - focus.elong0=<X>
     - Subtracts an offset correction from the elongation of an elliptical
       model of the source when and if focus calculations are performed.  <X>
       should be supplied as a percentage value.

   * - .. _forget:

       **forget**
     - forget=<key>, ...
     - Forget any prior values set for <key>, effectively removing it from the
       configuration.  New values may always be set, but you may also re-set
       a previously forgotten key using the recall_ command.  If <key> is set
       to 'conditionals' or 'blacklist', all currently stored conditionals or
       blacklisted keys will be removed.  See blacklist_ and conditionals_.

   * - .. _frames:

       **frames**
     - frames=<from>:<to>
     - Read only the specified frame ranges from the data.  Maybe useful for
       quick peeks at the data without processing the full scan, or when a part
       of the data is corrupted near the start or end of a scan.

   * - .. _gain:

       **gain**
     - gain=<X>
     - Specify an instrument gain of X from the detector stage (or fixed signal
       stage) to the readout.  Many instruments may automatically determine the
       relevant gain based on their data headers.  For others, the gains may
       have to be adjusted by hand, especially if they are changing.  Upon
       reading the scans, SOFSCAN will divide all data by the specified value,
       to bring all scans to a comparable signal level  Conversions to jansky_
       area referenced to such gain-scaled data.  See jansky_, dataunit_, and
       scale_.

   * - .. _gainnoise:

       **gainnoise**
     - gainnoise=<X>
     - Add noise to the initial gains.  There is not much use for this option,
       other than checking the robustness of the reduction on the initial gain
       assumption.  Since gains are usually measured in the reduction itself,
       typical reductions should not depend a lot on the initial gain values.
       See uniform_.

   * - .. _gains:

       **gains**
     - | [gains]
       | value={True, False}
     - Solve for pixel gains based on their response to the correlated noise
       (above).  If not specified, then all decorrelation steps will proceed
       without a gain solution.  A model-by-model control is offered by the
       `correlated.<modality>.nogains`_ option.  See `gains.estimator`_ and
       `correlated.<modality>.nogains`_.

   * - .. _gains.estimator:

       **gains.estimator**
     - | [gains]
       | estimator={median, maximum-likelihood}
     - Specify the type of estimator ('median' or 'maximum-likelihood') to be
       used for estimating pixel gains to correlated signals.  See estimator_
       and `correlated.<modality>`_.

   * - .. _gains.span:

       **gains.span**
     - | [gains]
       | span={True, False}
     - Make the gains of all correlated modalities span scans instead of
       integrations (subscans).  See `correlated.<modality>.span`_.

   * - .. _galactic:

       | **galactic**
       | Sets: system=galactic
     - galactic={True, False}
     - Reduce using new galactic coordinates (for mapping).  See system_,
       equatorial_, and altaz_.

   * - .. _gradients:

       | **gradients**
       | Alias: correlated.gradients
     - | [gradients]
       | value={True, False}
     - Shorthand for the decorrelation of gradients across the detector array.
       Such gradients can occur as a result of spatial sky-noise, or as
       temperature variation across the detectors.  See
       `correlated.<modality>`_.

   * - .. _grid:

       **grid**
     - grid={<X> or <dx>,<dy>}
     - Set the map pixelization to X arcseconds.  Pixelization smaller than 2/5
       of the beam is recommended.  The default is ~1/5 of the beam.  Non-square
       pixelization can be specified using <dx>,<dy> in arcseconds.

   * - .. _group:

       **group**
     - | [group]
       | <name>=10:20,45,50:60
       | ...
     - Specify a list of channels by IDs or fixed index (usually the same as
       storage index C-style 0-based), or ranges thereof that ought to belong
       to a group with name <name>.  See `division.<name>`_.

   * - .. _gyrocorrect:

       | **gyrocorrect**
       | Instrument: HAWC+
     - | [gyrocorrect]
       | <options...>
     - If present in the configuration, correct for gyrodrifts based on
       guide-star relock data stored in the scan headers.  This is not normally
       needed when the gyros function properly.  Occasionally however, they
       drift a fair bit, and this option can activate the correction scheme on
       demand.  See `gyrocorrect.max`_.

   * - .. _gyrocorrect.max:

       | **gyrocorrect.max**
       | Instrument: HAWC+
     - | [gyrocorrect]
       | max=<X>
     - Set a limit to how large of a gyro drift can be corrected for.  When
       drifts larger than X arcseconds are found in the scan, the correction is
       skipped for single scan reductions or dropped from the set in multi-scan
       reductions.

   * - .. _horizontal:

       | **horizontal**
       | Sets: system=horizontal
     - horizontal={True, False}
     - Reduce in horizontal coordinates (for mapping).  This is often useful for
       determining pointing offsets or for pixel location mapping.  See system_
       and pixelmap_.

   * - .. _indexing:

       **indexing**
     - | [indexing]
       | value={True, False}
     - Allow the use of data indexing to speed up coordinate calculations for
       mapping.  Without indexing the map coordinates are calculated at each
       mapping step.  This can be slow because of the complexity of the
       spherical projections, which often require several complex math
       evaluations.  With indexing enabled, the calculations are only performed
       once, and the relevant data is stored for future use.  However, this
       increases the memory requirement of SOFSCAN.  This, indexing may be
       disabled for very large reductions.  Alternatively, one may control the
       amount of memory such indexing may use via the `indexing.saturation`_
       option.  See grid_.

   * - .. _indexing.check_memory:

       **indexing.check_memory**
     - | [indexing]
       | check_memory=<True,False>
     - If True (default), performs a memory check to see if enough space
       exists in memory to index scans.  This should only really be turned
       off when running unit tests on a Windows virtual maching.  See
       indexing_.

   * - .. _indexing.saturation:

       **indexing.saturation**
     - | [indexing]
       | saturation=<X>
     - Specify the maximum fraction X of the total available memory that can be
       filled before indexing is automatically disabled.  Given a typical 20%
       overhead during reduction, values below 0.8 are recommended to avoid
       overflows.  See indexing_.

   * - .. _invert:

       **invert**
     - invert={True, False}
     - Invert signals.  This setting may be useful in creating custom
       jackknives, where the user wishes to retain control over which scans are
       inverted.  See gain_, scale_, and jackknife_.

   * - .. _iteration:

       **iteration**
     - | [iteration]
       | [[<N>, <X>, <x%>]]
       | <key>=<value>
       | ...
     - Use as a condition to delay settings until the Nth iteration.  E.g:

       | [iteration]
       | [[3]]
       | smooth=halfbeam

       will specify half-beam smoothing beginning on the 3rd iteration.  Note
       that the first iteration is numbered as 1.  Negative values for N are
       relative to the last iteration at -1.  For example, -2 references the
       penultimate iteration.  A fraction X or percentage x may also be supplied
       relative to the maximum number of rounds_.  For example, for a reduction
       with 10 rounds, the following settings will all be triggered on the 5th
       iteration:

       | [iteration]
       | [[5]]
       | smooth=5.0
       | [[0.5]]
       | smooth=6.0
       | [[-6]]
       | smooth=7.0
       | [[50%]]
       | smooth=8.0

       SOFSCAN will parse options as they are encountered in the configuration,
       so the resultant smooth setting on the 5th round will by 8.0.

   * - .. _jackknife:

       **jackknife**
     - | [jackknife]
       | value={True, False}
     - Jackkniving is a useful technique to produce accurate noise maps from
       large datasets.  When the option is used, the scan signals are randomly
       inverted so that the source signals in the large datasets will tend to
       cancel out, leaving noise maps.  The sign inversion is truly random in
       which repeated runs with the 'jackknife' flag will produce differenct
       jackknives every time.  If you want more control over which scans are
       inverted, consider using the invert_ flag instead.  See invert_,
       scramble_, `jackknife.frames`_, `jackknife.channels`_, and
       `jackknife.alternate`_.

   * - .. _jackknife.alternate:

       **jackknife.alternate**
     - | [jackknife]
       | alternate={True, False}
     - Rather than randomly inverting scans for a jackknife, this option will
       invert every other scan.  This may be preferred for small datasets,
       because it leads to better cancellation of source signals, especially
       with an even number of scans, chronologically listed.  To have the
       desired effect, use instead of jackknife_, rather than together with it
       (otherwise, the ordered inversion will simply compound the random method
       of the standard jackknife_.

   * - .. _jackknife.channels:

       **jackknife.channels**
     - | [jackknife]
       | channels={True, False}
     - Jackknife channels, such that they are randomly inverted for the source
       model.  Beware however, that channel-wise jackknives are not as
       representative of the true noise as the regular scan-wise jackknife_,
       because they will reject spatial correlations and instrumental
       channel-to-channel correlations.  See jackknife_, `jackknife.frames`_,
       and scramble_.

   * - .. _jackknife.frames:

       **jackknife.frames**
     - | [jackknife]
       | frames={True, False}
     - Jackknife frames, such that they are randomly inverted for the source
       model.  Beware however, that frame jackknives are not as representative
       if the true noise as the regular scan-wise jackknife_, because they will
       reject temporal correlations.

   * - .. _jansky:

       **jansky**
     - | [jansky]
       | value=<X>
     - Specify the calibration factor from dataunit_ to Jy such that
       Jansky's = dataunit * X.  See dataunit_, gain_, and `jansky.inverse`_.

   * - .. _jansky.inverse:

       **jansky.inverse**
     - | [jansky]
       | inverse={True, False}
     - When used, the jansky_ definition is inverted to mean Jy to dataunit_
       such that dataunit = X * Jansky's.

   * - .. _k2jy:

       **k2jy**
     - k2jy=<X>
     - The Jy/K conversion factor to X.  This allows SOFSCAN to calculate a data
       conversion to units of Kelvin if jansky_ is also defined.  Alternatively,
       the conversion to Kelvins can be specified directly via the kelvin_ key.

   * - .. _kelvin:

       **kelvin**
     - kelvin=<X>
     - Set the conversion to units of Kelvin (or more precisely, to K/beam
       units).  X defines the equivalent value of 1 K/beam expressed in the
       native dataunit_.  See dataunit_, jansky_, and k2jy_.

   * - .. _lab:

       | **lab**
       | Sets:
       | blacklist=source, filter.motion, tau, filter, whiten, shift, point
       | forget=downsample
       | write.spectrum=True
     - lab={True, False}
     - A conditional switch that indicates no astronomical observation was made.
       Effectively disables most tasks related to telescope motion or source
       derivation, and instead writes channel spectra to file.  See
       `write.spectrum`_.

   * - .. _last:

       | **last**
       | Alias: iteration.-1
     - | [last]
       | <key>=<value>
       | ...
     - An alias for settings to be applied on the last iteration.  See final_.

   * - .. _lock:

       **lock**
     - lock=<key1>,<key2>,...
     - Set a persistent option value that cannot be changed, cleared, or
       blacklisted later (e.g. by conditionally activated settings).  Users may
       use locks to ensure that their manually set reduction options are
       applied and never overridden.  For the lock to take effect, the option
       must not be blacklisted or locked to a different value before.  The
       value of a key will be set to its current value.  To release a lock,
       the unlock_ command may be issued.  See unlock_ and blacklist_.

   * - .. _los:

       | **los**
       | Instrument: HAWC+
       | Alias: correlated.los
     - | [los]
       | value={True, False}
     - Remove correlations with the second-derivative to the telescope
       line-of-sight (LOS) angle.  It is a good proxy for removing pitch-type
       acceleration response from the detector timestream.  See
       `correlated.<modality>`_.

   * - .. _map:

       | **map**
       | Sets: source.type=map
     - map={True, False}
     - A switch to produce a source map on output.

   * - .. _mappingfraction:

       **mappingfraction**
     - mappingfraction=<X>
     - Specify a minimum fraction of pixels (X) in the array that have to remain
       unflagged for creating a map from the scan.  If too many pixels are
       flagged in the reduction, it may be a sign of bigger problems,
       questioning the reliability of the scan data.  It is best to skip over
       problematic scans in order to minimize their impact on the mapping.  See
       mappingpixels_.

   * - .. _mappingpixels:

       **mappingpixels**
     - mappingpixels=<N>
     - Specify a minimum number of pixels (N) which have to be unflagged by the
       reduction in order for the scan to contribute to the mapping step.  See
       mappingfraction_.

   * - .. _map.size:

       **map.size**
     - | [map]
       | size=<dx>{x or X or , or tab or :}<dy>
     - Explicitly set the size of the mapped area centered on the source to a dx
       by dy arcseconds rectangle.  Normally, the map size is automatically
       calculated to contain all of the data.  One may want to restrict mapping
       to smaller regions (outside of which there should be no bright signals).
       See system_.

   * - .. _moving:

       **moving**
     - moving={True, False}
     - Explicitly specify that the object is moving in the celestial frame (such
       as solar system objects like plants, asteroids, comets, and moons).  This
       way, data will be properly aligned on the coordinates of the first scan.
       If the data headers are correctly set up (and interpreted by SOFSCAN),
       moving objects can be automatically detected.  This option is there in
       case things do not work as expected (e.g., if you notice that your solar
       system object smears or moves across the image with the default
       reduction.  Currently, this option forces equatorial coordinates.  This
       option is also aliased as planetary_.  See system_.

   * - .. _multibeam:

       | **multibeam**
       | Sets: source.type=multibeam
     - multibeam={True, False}
     - An alias for setting the source type to multibeam.

   * - .. _name:

       **name**
     - name=<filename>
     - Specify the output image filename, relative to the directory specified
       by outpath_.  When not given, SOFSCAN will choose a file name based on
       the source name and scan number(s), which is either:

         <sourcename>.<scanno>.fits

       or:

         <sourcename>.<firstscan>-<lastscan>.fits

       For mapping, other source model types (e.g. skydips or pixel maps) may
       have different default naming conventions.

   * - .. _nefd.map:

       **nefd.map**
     - | [nefd]
       | map={True, False}
     - True to use apparent map noise (if available, e.g. via
       `weighting.scans`_) to refine the reported NEFD estimate.  Else, the NEFD
       estimate will be based on the timestream noise alone.

   * - .. _noiseclip:

       **noiseclip**
     - noiseclip=<X>
     - Flag (clip) map pixels with a noise level that is more than X times
       higher than the deepest covered parts of the map.  See exposureclip_ and
       clip_.

   * - .. _noslim:

       **noslim**
     - noslim={True, False}
     - After reading the scans, SOFSCAN will discard data from channels flagged
       with a hardware problem to free up memory, and to speed up the reduction.
       This option overrides this behaviour, and retains all channels for the
       reduction whether used or not.

   * - .. _notch:

       **notch**
     - | [notch]
       | value={True, False}
     - Enable notch filtering the raw detector timestreams before further
       initial processing (e.g. downsampling).  The sub-options
       `notch.frequencies`_, `notch.harmonics`_. and `notch.width`_ are used to
       customize the notch filter response.

   * - .. _notch.frequencies:

       **notch.frequencies**
     - | [notch]
       | frequencies=<freq1>, <freq2>,...
     - A comma-separated list of frequencies (Hz) to notch out from the raw
       detector timestreams.  See `notch.harmonics`_. and `notch.width`_.

   * - .. _notch.harmonics:

       **notch.harmonics**
     - | [notch]
       | harmonics=<N>
     - Specify that the notch filter should also notch out N harmonics of the
       specified `notch.frequencies`_.  If not set, only the list of frequencies
       are notched, i.e. the same as 'harmonics=1'.  For example:

         notch.harmonics=2

       will notch out the list of frequencies set by `notch.frequencies`_ as
       well as their second harmonics.  See `notch.frequencies`_ and
       `notch.width`_.


   * - .. _notch.width:

       **notch.width**
     - | [notch]
       | width=<X>
     - Set the frequency width (Hz) of the notch filter response.  See
       `notch.frequencies`_.

   * - .. _obstime:

       **obstime**
     - | [conditionals]
       | [[obstime<operator><T>]]
       | <key>=<value>
       | ...
     - Configure settings based on the total observing time of all input scans.
       The total obstime is compared agains T (seconds) using <operator>, and
       all settings are applied if the requirement is met.  For example:

       | [conditionals]
       | [[obstime>60]]
       | stability=10

       will set the stability value to 10 if the total observation time is
       longer than one minute.  Nesting obstime conditions is possible with
       some limitations.  It is evaluated only once, after all scans have been
       read.  Thus, the condition will have no effect if activated later (e.g.
       if nested inside an iteration condition).

   * - .. _offset:

       | **offset**
       | Instrument: HAWC+
     - | [offset]
       | <sub>=<dx>,<dy>
       | ...
     - Specify subarray offsets.  For HAWC+ <sub> may take values of 'R0', 'R1',
       'T0', and/or 'T1'.  dx and dy are in units of pixels.  See rotation_.

   * - .. _offsets:

       | **offsets**
       | Sets: forget=drifts
     - offsets={True, False}
     - Remove the residual DC offsets from the bolometer signals using the
       'offsets' task in ordering_ rather than drifts_.

   * - .. _ordering:

       **ordering**
     - ordering=<task1>,<task2>,...
     - Specify the order of pipeline elements as a comma-separated list of keys.
       See offsets_, `correlated.<modality>`_, whiten_, and `weighting.frames`_.

   * - .. _organization:

       | **organization**
       | Telescope: SOFIA
     - organization=<text>
     - Specify the organization at which SOFSCAN is being used for reducing
       data.  The value of this option is stored directly in the FITS ORIGIN
       header key as required by the DCS.  If you want the ORIGIN key to be set
       properly, you might consider adding the organization option to
       '~/.sofscan/sofia/default.cfg' as 'SOFIA Science and Mission Ops'.

   * - .. _outpath:

       **outpath**
     - | [outpath]
       | value=<directory>
     - Specify the output path where all SOFSCAN output will be written
       (including maps etc.).  If not specified, will default to the current
       working directory.

   * - .. _outpath.create:

       **outpath.create**
     - | [outpath]
       | create={True, False}
     - When set, the output path will be automatically created as necessary.  If
       not, SOFSCAN will exit with an error if the output path does not exist.
       See outpath_.

   * - .. _parallel.cores:

       **parallel.idle**
     - | [parallel]
       | cores={N, x, X%}
     - Instruct SOFSCAN to use N number of CPU cores, fraction x of
       available processors, or X percent of available processors.  By default
       SOFSCAN will try to use 50% of the processing cores in your machine for
       decent performance without taking up too many resources.  This option
       allow modification of this behaviour according to need.

   * - .. _parallel.idle:

       **parallel.idle**
     - | [parallel]
       | idle={N, x, X%}
     - Instruct SOFSCAN to avoid using N number of CPU cores, fraction x of
       available processors, or X percent of available processors.

   * - .. _parallel.jobs:

       **parallel.jobs**
     - | [parallel]
       | jobs={N, x, X%}
     - Instruct SOFSCAN to allow a maximum of N jobs, fraction x of
       available cores, or X percent of available cores.  The maximum
       number of cores is set by `parallel.idle`_ or `parallel.cores`_.  This
       relates not only to the number of cores, but the number of threads inside
       each core, so that:

         cores * threads <= parallel.jobs

       The default is -1, indicating that the number of jobs is capped by the
       number of cores.

   * - .. _parallel.mode:

       **parallel.mode**
     - | [parallel]
       | mode=<mode>
     - Set the parallel processing mode.  <mode> may be one of:

         - *scans*: process scans in parallel.
         - *ops*: process each scan with parallel threads where possible.
         - *hybrid*: process as many scans in parallel as possible, each with
           an optimal number of threads.

       The default mode is 'hybrid'.

   * - .. _parallel.scans:

       **parallel.scans**
     - | [parallel]
       | scans=<True,False>
     - Perform the reduction tasks for all scans in parallel.  This is not
       recommended when dealing with large data sets due to memory pressure.

   * - .. _parallel.source:

       **parallel.source**
     - | [parallel]
       | source=<True,False>
     - Update the scan source models in parallel if True.  This is recommended
       when dealing with large sets of data due to better memory management
       procedures.

   * - .. _pcenter:

       | **pcenter**
       | Instrument: HAWC+
     - pcenter={<X> or <x>,<y>}
     - Specify the boresight position (pixels) on the detector array.  If a
       single value <X> is given, it will be applied to both the <x> and <y>
       directions (columns and rows).

   * - .. _peakflux:

       | **peakflux**
       | Instrument: HAWC+
     - peakflux={True, False}
     - Switch to peak-flux calibration instead of the default aperture flux
       calibration.  Recommended for point sources only.

   * - .. _perimeter:

       **perimeter**
     - perimeter={<N>, auto}
     - To speed up the sizing of the output image for large arrays (e.g. HAWC+)
       do not use the positions of each and every pixel.  Instead, identify a
       set of pixels that define an array perimeter from N sections around the
       centroid of the array.  N values up to a few hundred should be fail-safe
       for most typical array layouts, even when these have lots of pixels.

   * - .. _phases:

       **phases**
     - | [phases]
       | value={True, False}
     - Decorrelate the phase data (e.g. for chopped observations) for all
       correlated modes.  Alternatively, phase decorrelation can be turned on
       individually using the `correlated.<modality>.phases`_ options.

   * - .. _phases.estimator:

       **phases.estimator**
     - | [phases]
       | estimator={median, maximum-likelihood}
     - Overrides the global estimator setting for the phases (e.g. chopper
       phases).  The estimator may be either 'median' or 'maximum-likelihood'.
       If neither of these, it will default to 'maximum-likelihood'.  If not
       set, the global estimator_ will be used.


   * - .. _phasegains:

       **phasegains**
     - phasegains={True, False}
     - Use the information in the phases to calculate gains for all correlated
       modes.  The default is to use the fast samples for calculating gains.
       Alternatively, you can set this property separately for each correlated
       modality using `correlated.<modality>.phasegains`_.

   * - .. _pixeldata:

       **pixeldata**
     - pixeldata=<filename>
     - Specifies a pixel data file, providing initial gains, weights, and flags
       for detectors, and possibly other information as well depending on the
       specific instrument.  Such files can be produced via the
       `write.pixeldata`_ options (in addition to which you may want to specify
       'forget=pixeldata' so that flags are determined without prior bias).  See
       gainnoise_, uniform_, flag_, and blind_.

   * - .. _pixelmap:

       | **pixelmap**
       | Sets: source.type=pixelmap
     - | [pixelmap]
       | value={True, False}
     - Effectively the same as 'source.type=pixelmap' which is invoked by a
       condition.  Used for reducing pixel map data.  Instead of making a single
       map from all pixels, separate maps are create for each pixel.  (Note,
       this can chew up some memory if you have a lot of pixels).  At the end of
       the reduction, SOFSCAN determines the actual pixel offsets in the focal
       plane.  See `source.type`_, skydip_, and grid_.

   * - .. _pixelmap.process:

       **pixelmap.process**
     - | [pixelmap]
       | process={True, False}
     - Specify that pixel maps should undergo the same post-processing steps
       (e.g. smoothing, clipping, filtering, etc.) that are used for regular
       map-making.  When the option is not set, pixel maps are used in their
       raw maximum-likelihood forms.  See pixelmap_ and `pixelmap.writemaps`_.

   * - .. _pixelmap.writemaps:

       **pixelmap.writemaps**
     - | [pixelmap]
       | writemaps={True, False, <list>}
     - Pixel maps normally only produce the pixel position information as
       output.  Use this option if you want SOFSCAN to write individual pixel
       maps as well.  See pixelmap_ and `pixelmap.process`_.  You can specify
       which pixels to write by setting <list> which may contain comma-separated
       values or ranges referring to the integer fixed channel indices.  For
       example:

         pixelmap.writemaps=10,15:17

       would write pixel maps for channels 10, 15, 16, and 17.

   * - .. _pixels:

       **pixels**
     - | [pixels]
       | <options...>
     - Set user defined options relating to how the initial channel data is
       read and validated.  See pixeldata_ and rcp_.

   * - .. _pixel.criticalflags:

       **pixel.criticalflags**
     - | [pixel]
       | criticalflags=<flag1>, <flag2>,...
     - Determines which flags present in the initial channel data should
       continue to mark a channel as being flagged for the remainder of the
       reduction (unless removed by another reduction step).  The <flag>
       arguments may take the form of an integer, letter, or string (e.g. 'G',
       'GAIN', or 4).  Note that channel flags are usually specific to different
       instruments, so please ensure such flags are defined correctly.  For
       example, a pixeldata_ file may define one channel as spiky ('s') but
       if 'SPIKY' is not included in the critical flags, that channel will not
       flagged as such at the start of the reduction.  The default critical
       flags are 'GAIN', 'DEAD', and 'DISCARD'.

   * - .. _pixels.coupling.range:

       | **pixels.coupling.range**
       | Instrument: HAWC+
     - | [pixels]
       | [[coupling]]
       | range=<min>:<max>
     - Specify a valid range of coupling values for the initial channel data.
       Standard range syntax is observed such that `*` may indicated an
       unbounded limit. Any channel that has a coupling value outside of this in
       the initial channel data will be flagged as 'DEAD'.

   * - .. _pixels.coupling.exclude:

       | **pixels.coupling.exclude**
       | Instrument: HAWC+
     - | [pixels]
       | [[coupling]]
       | exclude=<x1>,<x2>,...
     - Flag channels with a coupling equal to certain values as 'DEAD' in the
       initial channel data.  For example:

         pixels.coupling.exclude=0,1

       would flag channels with initial coupling values exactly equal to 0 or 1
       as 'DEAD'.

   * - .. _pixels.gain.range:

       | **pixels.gain.range**
       | Instrument: HAWC+
     - | [pixels]
       | [[gain]]
       | range=<min>:<max>
     - Specify a valid range of gains for the initial channel data.  Standard
       range syntax is observed such that `*` may indicated an unbounded limit.
       Any channel that has a gain value outside of this in the initial channel
       data will be flagged as 'DEAD'.

   * - .. _pixels.gain.exclude:

       | **pixels.gain.exclude**
       | Instrument: HAWC+
     - | [pixels]
       | [[gain]]
       | exclude=<x1>,<x2>,...
     - Flag channels with gain equal to certain values as 'DEAD' in the initial
       channel data.  For example:

         pixels.gain.exclude=0,1

       would flag channels with initial gain values exactly equal to 0 or 1 as
       'DEAD'.

   * - .. _pixelsize:

       | **pixelsize**
       | Instrument: HAWC+
     - pixelsize={<X> or <x>,<y>}
     - Specify the pixel sizes (arcseconds) for the detector array.

   * - .. _planetary:

       | **planetary**
       | Alias: moving
     - planetary={True, False}
     - An alias for moving_.

   * - .. _point:

       **point**
     - point={True, False}
     - This is a convenience key for triggering settings for reducing pointing
       scans.  By default, it invokes:

       | [iteration]
       | [[last]]
       | pointing.suggest=True

       i.e. suggesting the pointing corrections in the last iteration.  See
       pointing_, `pointing.suggest`_ and `pointing.method`_.

   * - .. _pointing:

       **pointing**
     - | [pointing]
       | value={<x>,<y> or suggest}
     - Specify pointing corrections, or the way these should be derived.  The
       following values are accepted:

       - *<x>,<y>*: Specify relative pointing offsets as comma-separated values
         (arcseconds) in the system of the telescope mount.  I.e., these should
         be horizontal offsets for ground-based telescopes with an Alt/Az mount.
         Some instruments may allow more ways to specify pointing corrections.
       - *suggest*: Suggest pointing offsets (at the end of the reduction) from
         the scan itself.  This is only suitable when reducing compact pointing
         sources with sufficient S/N to be clearly visible in single scans.

       See point_.

   * - .. _pointing.degree:

       **pointing.degree**
     - | [pointing]
       | degree=<X>
     - Sets the degree (integer <X>) of spline used to fit the peak source
       amplitude value. This may be important for pixel maps where the map
       coverage is not sufficient to provide the required number of points
       for a third degree spline fit (default).

   * - .. _pointing.exposureclip:

       **pointing.exposureclip**
     - | [pointing]
       | exposureclip=<X>
     - Clip away the underexposed part of the map, below a relative exposure
       X times the most exposed part of the map.  This option works similarly to
       the exposureclip_ option, but applies only to the map used for deriving
       the pointing internally.

   * - .. _pointing.lsq:

       **pointing.lsq**
     - | [pointing]
       | lsq={True, False}
     - Attempt to fit the pointing using Least-Squares method rather than the
       chosen `pointing.method`_.  This will usually result in a better fit,
       but does not always successfully converge when the source is not easily
       modelled by a Gaussian.  In case the LSQ method fails, a secondary
       attempt will be made using `pointing.method`_.

   * - .. _pointing.method:

       **pointing.method**
     - | [pointing]
       | method={centroid, position, peak}
     - Specify the method used for obtaining positions of pointing sources.
       The available methods are:

       - *peak*: Take the maximum value as the peak location.
       - *centroid*: Take the centroid as the peak location.
       - *position*: The same as 'peak'.

       See `pointing.suggest`_.

   * - .. _pointing.radius:

       **pointing.radius**
     - | [pointing]
       | radius=<X>
     - Restrict the pointing fit to a circular area, with radius X (arcseconds),
       around the nominal map center.  it may be useful for deriving pointing in
       a crowded field.  See `pointing.suggest`_.

   * - .. _pointing.reduce_degrees:

       **pointing.reduce_degrees**
     - | [pointing]
       | reduce_degrees={True, False}
     - Allows the degree of spline fit to be lowered if there are insufficient
       points to allow for the requested fit (see `pointing.degree`_).

   * - .. _pointing.significance:

       **pointing.significance**
     - | [pointing]
       | significance=<X>
     - Set the significance (S/N) level required for pointing sources to provide
       a valid pointing result.  If the option is not set, a value of 5.0 is
       assumed.

   * - .. _pointing.suggest:

       **pointing.suggest**
     - | [pointing]
       | suggest={True, False}
     - Fit pointing for each input scan at the end of the reduction.  It can
       also be triggered by the point_ shorthand (alias), and may be enabled by
       default for certain types of scans, depending on the instrument.  E.g.,
       for HAWC+, pointing fits are automatically enabled for short single-scan
       reductions.  See `pointing.significance`_, `pointing.radius`_,
       `pointing.exposureclip`_, and `pointing.method`_.

   * - .. _pointing.tolerance:

       **pointing.tolerance**
     - | [pointing]
       | tolerance=<X>
     - Control how close (relative to the beam FWHM) the telescope pointing must
       be to its target position for determining photometry.  A distance of 1/5
       beams can result in a 10% degradation on the boundaries, while the signal
       would degrade by 25% at 1/3 beams distance.  This setting has no effect
       outside of photometry reductions.  See phases_ and chopped_.

   * - .. _positions.smooth:

       **positions.smooth**
     - | [positions]
       | smooth=<X>
     - Specify that the telescope encoder data should be smoothed with a time
       window X seconds wide in order to minimize the effects on encoder noise
       on the calculation of scanning speeds and accelerations.  These
       calculations may result in data being discarded, and are used in
       determining the optimal downsampling rates.  See aclip_, vclip_ and
       downsample_.

   * - .. _projection:

       **projection**
     - projection=<name>
     - Choose a map projection to use.  The following projections are available:

       - *SFL*: Sanson-Flamsteed
       - *SIN*: Slant Orthographic
       - *TAN*: Gnomonic
       - *ZEA*: Zenithal Equal Area
       - *MER*: Mercator
       - *CAR*: Plate-Carree
       - *AIT*: Hammer-Aitoff
       - *GLS*: Global Sinusoidal
       - *STG*: Stereographic
       - *ARC*: Zenithal Equidistant

       See system_, grid_ and `map.size`_.

   * - .. _pwv41k:

       | **pwv41k**
       | Telescope: SOFIA
     - pwv41k=<X>
     - Set a typical PWV value to X microns at 41k feet altitude.  See
       `tau.pwvmodel`_ and pwvscale_.

   * - .. _pwvscale:

       | **pwvscale**
       | Telescope: SOFIA
     - pwvscale=<X>
     - The typical water vapor scale height (kft) around 41 kilofeet altitude.
       See `tau.pwvmodel`_ and pwv41k_.

   * - .. _radec:

       | **radec**
       | Sets: system=equatorial
     - radec={True, False}
     - Reduce using equatorial coordinates for mapping (default).  See altaz_
       and system_.

   * - .. _range:

       **range**
     - | [range]
       | value=<min>:<max>
     - Set the acceptable range of data (in units it is stored).  Values outside
       of this range will be flagged, and pixels that are consistent offenders
       will be removed from the reduction (as set by `range.flagfraction`_.  See
       dataunit_, and `range.flagfraction`_.

   * - .. _range.flagfraction:

       **range.flagfraction**
     - | [range]
       | flagfraction=<X>
     - Specify the maximum fraction of samples for which a channel can be out of
       range (as set by range_) before that channel is flagged and removed from
       the reduction.  See range_.

   * - .. _rcp:

       **rcp**
     - | [rcp]
       | value=<filename>
     - Use the RCP file from <filename>.  RCP files can be produces by the
       pixelmap_ option from scans and for certain instruments, when the
       observation moves a bright source over all pixels.  For rectangular
       arrays, pixel positions can also be calculated on a regular grid using
       pixelsize_ and pcenter_.  See pixelmap_, pixelsize_, and pcenter_

   * - .. _rcp.center:

       **rcp.center**
     - | [rcp]
       | center=<x>,<y>
     - Define the center RCP position at x, y in arcseconds.  Centering takes
       place immediately after the parsing of RCP data.  See rcp_.

   * - .. _rcp.gains:

       **rcp.gains**
     - | [rcp]
       | gains={True, False}
     - Calculate coupling efficiencies using gains from the RCP files.
       Otherwise, uniform coupling is assumed with sky noise gains from the
       pixeldata_ file.  See rcp_.

   * - .. _rcp.rotate:

       **rcp.rotate**
     - | [rcp]
       | rotate=<X>
     - Rotate the RCP positions by X degrees (anti-clockwise).  Rotations take
       place after centering (if specified).  See rcp_.

   * - .. _rcp.zoom:

       **rcp.zoom**
     - | [rcp]
       | zoom=<X>
     - Zoom (rescale) the RCP position data by the scaling factor X.  Rescaling
       takes place after the centering (if defined).  See rcp_.

   * - .. _recall:

       **recall**
     - recall=<key1>,<key2>,...
     - Undo forget_, and reinstates <key> to its old value.  See forget_.

   * - .. _regrid:

       **regrid**
     - regrid=<X>
     - Re-grid the final map to a different grid than that used during the
       reduction where X is the final image pixel size in arcseconds.  See
       grid_.

   * - .. _resolution:

       **resolution**
     - resolution=<X>
     - Define the resolution of the instrument.  For single color imaging
       arrays, this is equivalent to beam_ with X specifying the instrument's
       main beam FWHM in arcseconds.  Other instruments (e.g. heterodyne
       receivers) may interpret 'resolution' differently.  See beam_.

   * - .. _roll:

       | **roll**
       | Instrument: HAWC+
       | Alias: correlated.roll
     - | [roll]
       | value={True, False}
     - Remove correlations with the second-derivative of the aircraft roll angle
       (roll-type accelerations).  See `correlated.<modality>`_.

   * - .. _rotation:

       **rotation**
     - | [rotation]
       | value=<X>
     - Define the instrument rotation X in degrees if applicable.

   * - .. _rotation.<sub>:

       | **rotation.<sub>**
       | Instrument: HAWC+
     - | [rotation]
       | <sub>=<X>
     - Specify subarray rotations X (degrees) where <sub can be R0, R1, T0,
       and/or T1.

   * - .. _rounds:

       **rounds**
     - rounds=<N>
     - Iterate N times.  You may want to increase the number of default
       iterations either to recover more extended emission (e.g. when extended_
       is set), or to go deeper (especially when the faint_ or deep_ options are
       used).  See iteration_, extended_, faint_, and deep_.

   * - .. _rows:

       | **rows**
       | Instrument: HAWC+
       | Alias:  correlated.rows
     - | [rows]
       | value={True, False}
     - Decorrelate on detector rows, or set options for it.  See
       `correlated.<modality>`_.

   * - .. _rtoc:

       | **rtoc**
       | Instrument: HAWC+
     - rtoc={True, False}
     - Instruct SOFSCAN to reference maps to Real-Time Object Coordinates (RTOC)
       for sidereal and non-sideral sources alike.  Normally, sidereal object
       coordinates are determined via the header keywords OBSRA/OBDEC or
       OBJRA/OBJDEC.  However, these were not always filled correctly during the
       2016 October flights, so this option provides a workaround in those
       scans.

   * - .. _scale:

       **scale**
     - | [scale]
       | value={<X>, <filename>}
     - Set the calibration scaling of the data.  The following values are
       available:

       - *X*: An explicit scaling value X, by which the entire scan data is
         scaled.
       - *filename*: The name of a calibration file which among other things,
         contains the ISO timestamp and the corresponding calibration values for

       Note: not all instruments support the <filename> value.  See tau_, gain_,
       invert_, and jackknife_.

   * - .. _scale.grid:

       **scale.grid**
     - | [scale]
       | grid=<X>
     - The grid resolution in arcseconds for which the scale_ value was
       derived.  If set, this correctly conserves flux values if grid_ is
       set to a different value.

   * - .. _scanmaps:

       **scanmaps**
     - scanmaps={True, False}
     - When specified, a map will be written for each scan (every time it is
       solved), under the name 'scan-<scanno>.fits' in the usual output path.
       Best to use as:

       | [iteration]
       | [[final]]
       | scanmaps=True

       To avoid unnecessary writing of scan maps for every iteration.  See
       final_ and source_.

   * - .. _scanpol:

       | **scanpol**
       | Instrument: HAWC+
       | Sets: config=scanpol.cfg
     - scanpol={True, False}
     - Use for scanning polarimetry scans with HAWC+.  Reads and applies the
       'scanpol.cfg' configuration file.

   * - .. _scramble:

       **scramble**
     - scramble={True, False}
     - Make a map with inverted scanning offsets.  Under the typical scanning
       patterns, this will not produce a coherent source.  Therefore, it is a
       good method for checking on the noise properties of deep maps.  The
       method essentially smears the source flux all over the map.  While not as
       good as jackknife_ for producing pure noise maps, jackknife_ requires a
       large number of scans for robust results (because of the random
       inversion), whereas 'scramble' can be used also for few, or even single
       scans to nearly the same effect.

   * - .. _segment:

       **segment**
     - segment=<X>
     - Break long integrations into shorter ones, with a maximum duration of X
       seconds.  It is the complement option to `subscan.merge`_, which does the
       opposite.  'segment' can also be used together with `subscan.split`_ to
       break the shorter segments into separate scans altogether.

   * - .. _serial:

       **serial**
     - | [serial]
       | [[<scan_range>]]
       | <key>=<value>
       | ...
     - Specify settings to apply when the scan's serial number falls within a
       specified range.  <scan_range> may be specified as:

       - `*`: always
       - a:b: Falls between the range (a, b)
       - >X: After serial number X
       - >=X: From serial number X
       - <X: Before serial number X
       - <=X: Before and up to serial number X

   * - .. _shift:

       **shift**
     - shift=<X>
     - Shift the data by X seconds to the frame headers.  It can be used to
       diagnose or correct for timing problems.

   * - .. _sigmaclip:

       **sigmaclip**
     - sigmaclip={n, True,False}
     - Removes frames that are outside of the permissible scanning speed range
       by iteratively remove speeds that are `n` times the
       standard deviation away from the median speed value. If `simgaclip` is 
       set to True `n` defaults to 5. 

   * - .. _signal-response:

       **signal-response**
     - signal-response={True, False}
     - This is a diagnostic option and affects the log output of decorrelation
       steps.  When set, each decorrelation step will produce a sequence of
       numbers, corresponding to the normalized covariances of the detector
       signals in each correlated mode in the modality.  The user may take this
       number as an indication of the importance of each type of correlated
       signal, and make decisions as to whether a decorrelation step is truly
       necessary.  Values close to 1.0 indicate signals that are (almost)
       perfectly correlated, whereas values near zero are indicative of
       negligible corrections.  See `correlated.<modality>`_ and ordering_.

   * - .. _skydip:

       | **skydip**
       | Sets: source.type=skydip
     - | [skydip]
       | value={True, False}
     - Reduce skydip data instead of trying to make in impossibly large map out
       of it.  This option is equivalent to specifying 'source.type=skydip'
       which is activated conditionally instead of an alias.

   * - .. _skydip.elrange:

       **skydip.elrange**
     - | [skydip]
       | elrange=<min>:<max>
     - Set the elevation range (degrees) to use for fitting the skydip model.
       In some cases, either the data may be corrupted at low or high
       elevations, or both.  This is a useful option to restrict the skydip data
       to the desired elevation range.  Use with caution to keep the skydip
       results robust.  See skydip_.

   * - .. _skydip.fit:

       **skydip.fit**
     - | [skydip]
       | fit=<p1>,<p2>,...
     - Specify the list of parameters to fit for the skydip model.  The standard
       model is:

         y(EL) = kelvin * tsky * (1-exp(-tau/sin(EL))) + offset

       where parameters (<pN>) may be:

       - *kelvin*: conversion from Kelvin to dataunits.  See kelvin_, dataunit_,
         and k2jy_.
       - *tsky*: sky temperature (in Kelvins).  See `skydip.tsky`_.
       - *tau*: the in band zenith opacity.  See `skydip.tau`_.
       - *offset*: an offset in dataunits.  See `skydip.offset`_.

       The default is to fit 'kelvin', 'tau', and 'offset', and assume that the
       sky temperature is close to ambient.  The assumption os the sky
       temperature is not critical so long as the conversion factor 'kelvin' is
       fitted to absorb an overall scaling.

   * - .. _skydip.grid:

       **skydip.grid**
     - | [skydip]
       | grid=<X>
     - Set the elevation binning (arcseconds) of the skydip data.  See grid_.

   * - .. _skydip.offset:

       **skydip.offset**
     - | [skydip]
       | offset=<X>
     - Specify the initial offset value in dataunit_.  See `skydip.fit`_.

   * - .. _skydip.tau:

       **skydip.tau**
     - | [skydip]
       | tau=<X>
     - Specify the initial in-band zenith opacity.  See `skydip.fit`_.

   * - .. _skydip.tsky:

       **skydip.tsky**
     - | [skydip]
       | tsky=<X>
     - Specify the initial sky temperature in Kelvins.  By default, the ambient
       temperature (if available) will be used.  See `skydip.fit`_.

   * - .. _smooth:

       **smooth**
     - | [smooth]
       | value={<X>, minimal, halfbeam, 2/3beam, beam, optimal}
     - Smooth the map by X arcsec FWHM beam.  Smoothing helps improve visual
       appearance, but is also useful during reduction to create more
       redundancy in the data in the intermediate reduction steps.  Also,
       smoothing by the beam is optimal for point source extraction from deep
       fields.  Therefore, beam smoothing is default with the deep_ option
       (see 'deep.cfg').  Typically you want to use some smoothing during
       reduction, and you may want to turn it off in the final map.  Such a
       typical configuration may look like:

       | smooth=9.0  # 9" smoothing at first
       | [iteration]
       | [[2]]
       | smooth=12.0  # smooth more later
       | [[last]]
       | forget=smooth  # no smoothing at end

       Other than specifying explicit values, you can use the predefined
       values: 'minimal', 'halfbeam', '2/3beam', 'beam', or 'optimal'.  See
       `smooth.optimal`_, final_, `source.filter`_, and grid_.

   * - .. _smooth.external:

       | **smooth.external**
       | *(Not implemented yet)*
     - | [smooth]
       | external={True, False}
     - Do not actually perform the smoothing set by the smooth_ option.
       Instead, use the smooth_ value as an assumption in calculating
       smoothing-related corrections.  The option is designed for the reduction
       of very large datasets, which have to be "split" into smaller,
       manageable sized chunks.  The unsmoothed outputs can be coadded and then
       smoothed to the desired amount before feeding the result back for further
       rounds of reduction via `source.model`_.  See smooth_, `subscan.split`_,
       and `source.model`_.

   * - .. _smooth.optimal:

       **smooth.optimal**
     - | [smooth]
       | optimal=<X>
     - Define the optimal smoothing for point-source extraction if it is
       different from beam-smoothing.  For arrays whose detectors are completely
       independent, beam-smoothing produces the optimal signal-to-noise for
       point sources.  However, if the detectors are not independent, the
       optimal smoothing may vary.  This is expected to be the case for some
       filled arrays, where one expects a certain level of beam-sized photon
       correlations.  See smooth_.

   * - .. _source:

       **source**
     - | [source]
       | value={True, False}
     - Solve for the source model, or set options for it.

   * - .. _source.correct:

       **source.correct**
     - | [source]
       | value={True, False}
     - Correct peak fluxes for the point source filtering effect of the various
       reduction steps (default).  The filtering of point sources is carefully
       calculated through the reduction steps, this with the correction scheme,
       point source fluxes ought to stay constant (within a few percent)
       independent of the pipeline configuration.  See faint_, deep_, bright_,
       ordering_, and whiten_.

   * - .. _source.coupling:

       **source.coupling**
     - | [source]
       | [[coupling]]
       | <options...>
     - If present in the configuration, (re-)calculate point source coupling
       efficiencies (the ratio of point-source and sky-noise response) as part
       of the source modeling step.  This is only really useful for bright
       sources.  See `source.coupling.range`_.

   * - .. _source.coupling.range:

       **source.coupling.range**
     - | [source]
       | [[coupling]]
       | range=<min>:<max>
     - Specify the range of acceptable coupling efficiencies relative to the
       "average" of all pixels when `source.coupling`_ is used to calculate
       these based on bright source responses.  Pixels with efficiencies outside
       of the specified range will be flagged and ignored from further source
       modeling steps until these flags are cleared again in the reduction.
       See `correlated.<modality>.gainrange`_.

   * - .. _source.coupling.s2n:

       **source.coupling.s2n**
     - | [source]
       | [[coupling]]
       | s2n=<min>:<max>
     - Set the acceptable range of S/N required in the map for using the
       position for estimating detector coupling gains when the
       `source.coupling`_ option is enabled.

   * - .. _source.delete_scan:

       **source.delete_scan**
     - | [source]
       | delete_scan=<True,False>
     - If True, and updating the source in parallel is also True (see
       `parallel.source`_, delete the individual scan source model once all
       required processing has been performed.  This is recommended when
       dealing with large sets of data to reduce memory pressure.

   * - .. _source.despike:

       **source.despike**
     - | [source]
       | [[despike]]
       | <options...>
     - If present in the configuration, despike the scan maps using an S/N
       threshold of `source.despike.level`_.  Clearly, this should be higher
       than the most significant source in your map.  Therefore, it is only
       really useful in deep_ model, where 5-signa despiking is default.
       See 'deep.cfg'.

   * - .. _source.despike.level:

       **source.despike.level**
     - | [source]
       | [[despike]]
       | level=<X>
     - Set the source despiking level to an S/N of X.  You probably want to set
       X to be no more than about 10 times the most significant source in your
       map.  See `source.despike`_.

   * - .. _source.filter:

       **source.filter**
     - | [source]
       | [[filter]]
       | <options...>
     - Filter extended structures.  By default, the filter will skip over map
       pixels that are above the `source.filter.blank`_ S/N level (>6 by
       default).  Thus, any structure above this significance level will remain
       unfiltered.  Filtering is useful to get deeper in the map when retaining
       the very faint extended structures is not an issue.  Filtering above 5
       times the source size (see sourcesize_`) is default when the filter is
       used.

   * - .. _source.filter.blank:

       **source.filter.blank**
     - | [source]
       | [[filter]]
       | blank=<X>
     - Set the blanking level of the large-scale structure (LSS) filter.  Any
       map pixels with an S/N above the specified level will be skipped over,
       and therefore remain unaffected by the filter.  See
       `source.filter.fwhm`_.

   * - .. _source.filter.fwhm:

       **source.filter.fwhm**
     - | [source]
       | [[filter]]
       | fwhm=<X>
     - Specify the Gaussian FWHM of the large-scale structure (LSS) filter.
       Values greater than about 5-times the beam size are recommended in order
       to avoid the unnecessary filtering of compact or point sources.  See
       `source.filter.blank`_.

   * - .. _source.filter.type:

       **source.filter.type**
     - | [source]
       | [[filter]]
       | type={convolution, fft}
     - Specify the type of the large-scale structure filter.  Convolution is
       more accurate but may be slower than FFT, especially for very large maps.

   * - .. _source.fixedgains:

       **source.fixedgains**
     - | [source]
       | fixedgains={True, False}
     - Specify the use of fixed source gains (e.g. from an RCP file).  Normally,
       SOFSCAN calculates source gains based on the correlated noise response
       and the specified point source couplings ( e.g. as derived from the two
       gain columns of RCP files).  This option can be used to treat the
       supplied source gains as static (i.e. decoupled from the sky-noise
       gains).  See `source.coupling`_ and pixelmap_.

   * - .. _source.flatfield:

       | **source.flatfield**
       | Sets: config=flatfield.cfg
     - | [source]
       | flatfield={True, False}
     - Use for deriving flatfields based on response to a source.  For it to
       work effectively, you need a scan that moves bright source emission over
       all fields.  It is a soft option, defined in 'default.cfg', and it
       results in loading 'flatfield.cfg' for configuring optimal settings for
       source gain derivation.

   * - .. _source.intermediates:

       **source.intermediates**
     - | [source]
       | intermediates={True, False}
     - Write the maps made during the reduction into 'intermediate.fits' (inside
       the SOFSCAN output directory).  This allows the user to keep an eye on
       the evolution of maps iteration-by-iteration.  Each iteration will
       overwrite this temporary file, and it will be erased at the end of the
       reduction.

   * - .. _source.mem:

       **source.mem**
     - | [source]
       | mem={True, False}
     - Use the maximum-entropy method (MEM) correction to the source map.  The
       MEM requirement suppresses some of the noise on the small spatial scales,
       and pushes solutions closer to the zero level for low S/N structures.
       This increases contrast between significant source structures and
       background.  It is similar to the MEM used in radio interferometry,
       although there are key differences.  For one, interferometry measures
       components in the uv-plane, and MEM corrections are applied in xy
       coordinate space.  For SOFSCAN, both the solutions and corrections are
       applied in the same configuration space.  See `source.mem.lambda`_.

   * - .. _source.mem.lambda:

       **source.mem.lambda**
     - | [source]
       | [[mem]]
       | lambda=<X>
     - Specify the desirability of MEM solutions relative to the
       maximum-likelihood solution.  Typical values of lambda are in the range
       0.1 to 1, but higher or lower values may be set to give extra weight
       towards one type of solution.

   * - .. _source.model:

       | **source.model**
       | *(Not implemented yet)*
     - | [source]
       | model=<filename>
     - Specify an initial source model to use in the reduction.  This may be
       useful when reducing large datasets where all data cannot be reduced
       together.  Instead, the data can be split into manageable sized chunks
       which are reduced separately.  The results can be coadded to create a
       composite map.  This may be further manipulated (e.g. S/N clipping,
       smoothing, filtering, etc.) before feeding back into another round of
       reduction.  Clipping and blanking settings are usually altered when an
       a-priori source-model is thus defined.  See blank_, clip_, and
       `smooth.external`_.

   * - .. _source.nosync:

       **source.nosync**
     - | [source]
       | nosync={True, False}
     - Do not bother syncing the source solution back into the raw timestream.
       This saves a bit of time in the last round of most reductions when the
       source_ is the last step in the pipeline, and the residuals are not used
       otherwise (e.g. by `write.covar`_, `write.ascii`_ or `write.spectrum`_).

   * - .. _source.redundancy:

       **source.redundancy**
     - | [source]
       | redundancy=<N>
     - Specify the minimum redundancy (N samples) that each scan-map pixel
       output ought to have in order to be considered valid.  Pixels with
       redundancies smaller than this critical value will be flagged and not
       used in the composite source making.

   * - .. _source.sign:

       **source.sign**
     - | [source]
       | sign=<spec>
     - Most astronomical source have a definite signedness.  For continuum, we
       expect to see emission, except when looking at SZ clusters at 2-mm, which
       have a unique negative signature.  SOFSCAN can do a better job if the
       signature of the source is predetermined.  The sign specification <spec>
       can be:

       - *positive*: +, positive, plus, pos, >0
       - *negative*: -, negative, minus, neg, <0
       - *any*: `*`, any, 0

       When not set, the default is to assume that sources be may of either sign
       (same as `*`, any, or 0).  The signature determines how source clipping
       and blanking are implemented.  See clip_ and blank_.

   * - .. _source.type:

       **source.type**
     - | [source]
       | type=<type>
     - By default, SOFSCAN will try to make a map from the data.  However, some
       instruments may take data that is analyzed differently.  For example, you
       may want to use SOFSCAN to reduce pixels maps (to determine the position
       of pixels on the sky), or skydips (to derive appropriate opacities), or
       do point source photometry.  Presently, the following source types
       (<type>) are supported for all instruments:

       - *map*: Make a map of the source (default)
       - *cube*: Make a spectral cube
       - *skydip*: Reduce skydips and determine opacities by fitting a model.
       - *pixelmap*: Create individual maps for every pixel, and use it to
         determine their location in the field of view.
       - *None*: Do not generate a source model.  Useful for lab/diagnostic
         reductions.

       Note: you may also just use skydip_ and pixelmap_ shorthands to the same
       effect.

   * - .. _sourcesize:

       **sourcesize**
     - sourcesize=<X>
     - This option can be used instead of extended_ in conjunction with faint_
       or deep_ to specify the typical size of sources (FWHM in arcseconds) that
       are expected.  The reduction then allows filtering structures that are
       much larger than the specified source size.  If sourcesize_ or extended_
       are not specified, then point-like compact sources are assumed.  The
       source size helps tune the 1/f filter (see drifts_) optimally.  The 1/f
       timescale is set to be the larger of the stability_ or 5 times the
       typical source crossing time (calculated via sourcesize_).  Note that
       noise whitening will mute the effect of this settings almost completely.
       See faint_, extended_, and whiten_.

   * - .. _split:

       | **split**
       | Sets:
       | smooth.external=True
       | [last]
       | forget=exposureclip
     - split={True, False}
     - A convenience key for adjusting options for very large data sets which
       have to be split into manageable sized chunks in the reduction.  See
       `smooth.external`_ and `source.model`_.

   * - .. _stability:

       **stability**
     - stability=<X>
     - Specify the instrument's 1/f stability time scale in seconds.  This value
       is used for optimmizing reduction parameters when these options are not
       explicitly specified (e.g. the filtering timescale for the drifts_
       option).  See drifts_ and sourcesize_.

   * - .. _subarray:

       | **subarray**
       | Instrument: HAWC+
     - subarray=<sub1>,<sub2>,...
     - Restrict the analysis to just the selected subarrays.  For HAWC+, the
       <sub?> may contain the subarray IDs: R0, R1, T0, and T1, or R to specify
       R0 and R1, or T to specify T0 and T1.

   * - .. _subscan.merge:

       **subscan.merge**
     - | [subscan]
       | [[merge]]
       | value={True, False}
     - Specifies that the integrations (subscans) in a scan should be merged
       into a single timestream, will invalid frames filling potential gaps at
       the boundaries to ensure proper time-spacing of all data (for time window
       processing of FFTs).  See `subscan.split`_.

   * - .. _subscan.merge.maxgap:

       **subscan.merge.maxgap**
     - | [subscan]
       | [merge]
       | maxgap=<X>
     - Merging integrations (subscans) will pad gaps between them with invalid
       frames as needed.  Use this option to limit how much padding X (seconds)
       is allowed.  If the gap between two consecutive subscans is larger than
       the maximum gap specified by this option, then the merge will continue in
       a separate scan.

   * - .. _subscan.minlength:

       **subscan.minlength**
     - | [subscan]
       | minlength=<X>
     - Set the minimum length of integrations (subscans) to X seconds.
       Integrations shorter than the specified value will be skipped during the
       scan reading phase.  Most reductions rely on the background variations to
       create signals from which detector gains can be estimated with the
       required accuracy.  Very short integrations may not have sufficient
       background signals for the robust estimation of gains, and it is thus
       best to simply ignore such data.

   * - .. _subscan.split:

       **subscan.split**
     - | [subscan]
       | split={True, False}
     - Allow subscans (integrations) to be split into separate scans.  This is
       practical to speed up the reduction of single scans with may subscans on
       machines with multi-core CPUs, since the reduction does not generally
       process integrations in parallel, but nearly always does for scans.  See
       `subscan.merge`_.

   * - .. _supergalactic:

       | **supergalactic**
       | Sets: system=supergalactic
     - supergalactic={True, False}
     - Make maps in supergalactic coordinates.  See system_.

   * - .. _system:

       **system**
     - system=<type>
     - Select the coordinate system for mapping.  Available <type> values are:

       - *equatorial* (default)
       - *horizontal*
       - *ecliptic*
       - *galactic*
       - *supergalactic*
       - *focalplane*
       - *native*

       Most of these values are aliased to simply keys.  See altaz_,
       equatorial_, ecliptic_, galactic_, supergalactic_, radec_, horizontal_,
       and focalplane_.

   * - .. _tau:

       **tau**
     - | [tau]
       | value={<X>, <spec>}
     - Specify an in-band zenith opacity value to use (<X>).  For some
       instruments, the <spec> may be used to specify a filename with lookup
       information, or tau in another band (see `tau.<?>`_) with an appropriate
       scaling relation to in-band values (see `tau.<?>.a`_ and `tau.<?>.b`_).

       When lookup tables are used, the tau values will be interpolated for each
       scan, so long as the scan falls inside the interpolator's range.
       Otherwise, a tau of 0.0 will be used.  For SOFIA instruments, <spec>
       may also take the values {atran, pwvmodel}.  Please see
       `atran.reference`_, `tau.pwvmodel`_, and `tau.<?>`_ for further details.

   * - .. _tau.pwvmodel:

       | **tau.pwvmodel**
       | Telescope: SOFIA
     - | [tau]
       | pwvmodel={True, False}
     - Estimate a typical PWV value (for opacity correction) based on altitude
       alone.  See pwv41k_ and pwvscale_.

   * - .. _tau.<?>:

       **tau.<?>**
     - | [tau]
       | [[<?>]]
       | value=<X>
     - Specify the tau value for X for <?> where <?> can stand for any
       user-specified relation.  Some useful conversion relations are predefined
       for certain instruments.  E.g. some typical values may be 'pwv'
       (millimeters of precipitable water vapor).  The values will be scaled to
       in-band zenith opacities using the linear scaling relations defined via
       the `tau.<?>.a`_ and `tau.<?>.b`_ constants.

   * - .. _tau.<?>.a:

       **tau.<?>.a**
     - | [tau]
       | [[<?>]]
       | a=<X>
     - Define the scaling term for the opacity measure <?>.  Zenith opacities
       are expressed in a linear relationship to some user-defined tau parameter
       t as:

         tau(<?>) = (a * t) + b

       This key sets the linear scaling constant 'a' in the above equation,
       while `tau.<?>.b`_ specifies the offset value.

   * - .. _tau.<?>.b:

       **tau.<?>.b**
     - | [tau]
       | [[<?>]]
       | b=<X>
     - Set the offset value in a linear tau scaling relationship.  See
       `tau.<?>.a`_ for details.

   * - .. _uniform:

       **uniform**
     - uniform={True, False}
     - Instruct the use of uniform pixel gains initially instead of the values
       read from the appropriate pixel data file.  See pixeldata_.

   * - .. _unit:

       **unit**
     - unit=<name>
     - Set the output units to <name>.  You can use either the instrumental
       units (e.g. 'V/beam' or 'count/beam') or the more typical 'Jy/beam'
       (default).  All names must be parseable by the astropy.units.Unit Python
       class.  See dataunit_ and jansky_.

   * - .. _unlock:

       **unlock**
     - unlock=<key1>,<key2>,...
     - Release the lock on a configuration option, allowing it to be changed.
       See lock_ for further details.

   * - .. _vclip:

       **vclip**
     - | [vclip]
       | value={auto, <min>:<max>}
     - Clip data where the field scan velocity is outside the specified range
       (<min>:<max> in arcseconds/second).  The successful disentangling of the
       source structures from the various noise terms release on these being
       separated in the frequency space.  With typical 1/f type limiting noise,
       this is harder when the scan speed is low such that the source signals
       occupy the low frequencies.  Therefore, requiring a minimum scanning
       speed is a good idea.  Likewise, too high scanning speeds will smear out
       sources if the movement between samples is larger than ~1/3 beam.  A
       value of 'auto' can be specified to set the velocity clipping range
       optimally based on the typical scanning speeds.  See `vclip.strict`_,
       aclip_, and resolution_.

   * - .. _vclip.strict:

       **vclip.strict**
     - | [vclip]
       | strict={True, False}
     - When set, discard any frames outside of the acceptable range of mapping
       speeds (as defined by the vclip_ option), rather than the default
       approach of simply flagging slow motion for source modelling only.

   * - .. _weighting:

       **weighting**
     - | [weighting]
       | value={True, False}
     - Derive pixel weights based on the RMS of the unmodelled timestream
       signals.

   * - .. _weighting.frames:

       **weighting.frames**
     - | [weighting]
       | [[frames]]
       | <options...>
     - If configured, calculate time weights in addition to pixel weighting to
       allow for non-stationary noise.  See `weighting.frames.resolution`_.

   * - .. _weighting.frames.noiserange:

       **weighting.frames.noiserange**
     - | [weighting]
       | [[frames]]
       | noiserange=<min>:<max>
     - Set the acceptable range of temporal noise variation.  Standard range
       syntax may be used such as wildcards (`*`) to indicate an open range or
       a hyphen (-) instead of a colon (:).  See `weighting.noiserange`_.

   * - .. _weighting.frames.resolution:

       **weighting.frames.resolution**
     - | [weighting]
       | [[frames]]
       | resolution={<X>, auto}
     - By default, all exposures are weighted independently.  With this option
       set, weights are derived for blocks of exposures spanning X seconds.  The
       value 'auto' can also be used to match the time-constant to that of
       drifts_.  Time weighting is often desired but can cause instabilities
       during the reduction, especially if the time-scale is mismatched to other
       reduction steps.  Adjust the time scale only if you really understand
       what you are doing.

   * - .. _weighting.method:

       **weighting.method**
     - | [weighting]
       | method=<name>
     - Set the method used for deriving pixel weights from the residuals.  The
       following methods (<name>) are available:

       - *rms*: Standard RMS calculation.
       - *robust*: Use robust estimates for the standard deviation.
       - *differential*: Estimate noise based on pairs of data separated by some
         interval.

   * - .. _weighting.noiserange:

       **weighting.noiserange**
     - | [weighting]
       | noiserange=<min>:<max>
     - Specify what range of pixel noises are admissible relative to the median
       pixel noise.  Pixels that fall outside of the <min> or <max> will be
       flagged.  Standard range syntax may be used such as wildcards (`*`) to
       indicate an open range or a hyphen (-) instead of a colon (:).  See
       `weighting.frames.noiserange`_.

   * - .. _weighting.scans:

       **weighting.scans**
     - | [weighting]
       | [[scans]]
       | value={True, False}
     - If set, each scan gets an assigned weight with which it contributes to
       the composite map.  This weight is measured directly from the noise
       properties of the produced map.

   * - .. _weighting.scans.method:

       **weighting.scans.method**
     - | [weighting]
       | [[scans]]
       | method={robust, maximum-likelihood}
     - The method by which to calculate the scan weighting.  'robust' method
       weights by median(V) / 0.454937, whereas any other method weights by
       mean(V) where V is the significance map variance.

   * - .. _whitelist:

       **whitelist**
     - whitelist=<key1>, <key2>,...
     - Remove any key from the blacklist, allowing it to be set again if
       desired.  Whitelisting an option may not set it to its prior value, so
       you should explicitly set it again or recall_ it to it's prior state.

   * - .. _whiten:

       | **whiten**
       | Alias: `filter.whiten`_
     - | [whiten]
       | <options...>
     - An alias for `filter.whiten`_.

   * - .. _whiten.level:

       | **whiten.level**
       | Alias: `filter.whiten.level`_
     - whiten.level=<X>
     - An alias for `filter.whiten.level`_.

   * - .. _whiten.minchannels:

       | **whiten.minchannels**
       | Alias: `filter.whiten.minchannels`_
     - whiten.minchannels=<N>
     - An alias for `filter.whiten.minchannels`_.

   * - .. _whiten.proberange:

       | **whiten.proberange**
       | Alias: `filter.whiten.proberange`_
     - whiten.proberange=<spec>
     - An alias for `filter.whiten.proberange`_.

   * - .. _wiring:

       **wiring**
     - wiring=<filename>
     - This option is commonly used to specify a file containing the wiring
       information of the detectors, which can be used to establish the typical
       groupings of the instruments.  There is no standard format for the wiring
       file (if may vary by instrument), and not all instruments may use such
       information.  See pixeldata_ and rcp_.

   * - .. _write.ascii:

       **write.ascii**
     - | [write]
       | ascii={True, False}
     - Write the residual timestreams to an ASCII table.  The file will contain
       as many columns as there are pixels in the reduction (see noslim_), each
       corresponding to a pixel timestream.  The first row contains the sampling
       rate (Hz).  Flagged data is indicated with a NaN character.  See noslim_
       and `write.spectrum`_.

   * - .. _write.coupling:

       **write.coupling**
     - | [write]
       | [[coupling]]
       | value=<sig1>, <sig2>,...
     - Measure and write coupling gains to the given signals (<sig>).  Coupling
       gains are similar to correlation coefficients but normalized differently
       so that they can be used directly to remove the correlated signal from
       the timestream.  For example:

         write.coupling=telescope-x,accel-mag

       will write out the coupling gains of each detector to the telescope
       azimuth motion ('telescope-x') and scalar acceleration ('accel-mag').
       See `correlated.<modality>`_.

   * - .. _write.covar:

       **write.covar**
     - | [write]
       | [[covar]]
       | <value>=<spec1>, <spec2>,...
     - Write covariance data.  If no value is specified, the full pixel-to-pixel
       covariance data will be writen to a FITS image.  The optional <value> can
       specify the ordering of the covariance matrix according to pixel
       divisions.  Each group in the pixel division will be blocked together for
       easy identification of block-diagonal covariance structures.  Other than
       the division names, the list can contain 'full' and 'reduced' to indicate
       the full covariance matrix of all instrument pixels, or only those that
       were used in the reduction.  See `division.<name>`_ and noslim_.

   * - .. _write.covar.condensed:

       **write.covar.condensed**
     - | [write]
       | [[covar]]
       | condensed={True, False}
     - When writing covariance matrices, write only 'live' channels.  I.e. those
       that are unflagged by the reduction.  This results in a covariance matrix
       without gaps.  The downside is that identifying particular
       pixels/channels may be difficult in that form.  See `write.covar`_.

   * - .. _write.flatfield:

       **write.flatfield**
     - | [write]
       | [[flatfield]]
       | value={True, False}
     - Write a DRP flatfield FITS file to be used by the chop-nod pipeline.  The
       file format is specified by Marc Berthoud.

   * - .. _write.flatfield.name:

       **write.flatfield.name**
     - | [write]
       | [[flatfield]]
       | name=<filename>
     - An optional setting to specify the FITS file name for `write.flatfield`_.
       If not present, a default name containing the scan ID is written to
       outpath_.

   * - .. _write.pixeldata:

       **write.pixeldata**
     - | [write]
       | pixeldata={True, False}
     - Write the pixel data file (gains, weights, flags).  The output will be
       pixel-<scanno>.dat' in outpath_.  You can use these files to update
       instrumental defaults in the instrument subdirectory.  E.g., to replace
       'pixel-A.170mK.F445.dat' in data/configurations/hawc_plus/.  See rcp_ and
       wiring_.

   * - .. _write.png:

       **write.png**
     - | [write]
       | [[png]]
       | value={True, False}
     - Write a PNG thumbnail with the final result.  The PNG image has the same
       name as the output file with a '.png' appended.  See `write.png.color`_,
       `write.png.crop`_, `write.png.plane`_, `write.png.size`_, and
       `write.png.smooth`_.

   * - .. _write.png.color:

       **write.png.color**
     - | [write]
       | [[png]]
       | color=<name>
     - Set the color scheme for rendering the PNG image.  The available color
       scheme names are any that may be passed into the 'cmap' parameter of the
       Python function matplotlib.pyplot.imshow.  If not supplied, the default
       will be 'viridis'.

   * - .. _write.png.crop:

       **write.png.crop**
     - | [write]
       | [[png]]
       | crop={auto or <xmin>,<ymin>, <xmax>,<ymax>}
     - Set rectangular bounds to the PNG output image in the instrument's native
       size unit (usually arcseconds).  The argument is usually a list of
       comma-separated corners relative the the source position.  If a single
       value is given then the PNG output will be a square area with +/- that
       size in X and Y.  If 2 or 3 values are supplied, the missing offsets will
       be assumed to be the negative equivalent to the coordinates given.  Thus:

       - 90 = -90, -90, 90, 90
       - 60, 90 = -60, -90, 60, 90
       - -45, -50, 60 = -45, -50, 60, 50

       If 'auto' is used, the map will automatically be cropped to the best
       dimensions for all valid pixels in the map.  See `write.png`_.

   * - .. _write.png.plane:

       **write.png.plane**
     - | [write]
       | [[png]]
       | plane={flux, noise, weight, time, s2n}
     - Selects the FITS image plane to write into the PNG.  Unrecognized planes
       will be interpreted as 'flux' (default).  See `write.png`_.

   * - .. _write.png.size:

       **write.png.size**
     - | [write]
       | [[png]]
       | size={<x>X<y> or <x>,<y> or <X>}
     - Set the size of the PNG thumbnails.  You can specify both a single
       integer for square images or two integers separated by 'x', ',' or 'X'.
       E.g., 640x480.  The default size is 300x300.  See `write.png`_.

   * - .. _write.png.smooth:

       **write.png.smooth**
     - | [write]
       | [[png]]
       | smooth=<spec>
     - Specify how much to smooth the PNG output.  The options works in the same
       manner to the regular smooth_ option for FITS images, but is not
       completely independent from it.  PNG images are always smoothed as much
       as required by smooth_, and this option is only effective if the PNG
       smoothing is larger.

   * - .. _write.signals:

       **write.signals**
     - | [write]
       | signals={True, False}
     - Write out all the correlated signals that were calculated in the
       reduction as ASCII timestreams.  Each signal mode is written in its own
       file, named after the mode's name and carrying a '.tms' extension.  The
       files are simple ASCII timestreams with the sampling frequency appearing
       in the first row.

   * - .. _write.scandata:

       **write.scandata**
     - | [write]
       | [[scandata]]
       | value={True, False}
     -  Whether or not to add HDUs at the end of the output FITS image
        describing each scan (default).  Each scan will contribute an extra HDU
        at the end of the image.  Disabling this option (e.g. via forget_) can
        decrease the size of the output images, especially for large data sets
        containing many scans.

   * - .. _write.scandata.details:

       **write.scandata.details**
     - | [write]
       | [[scandata]]
       | details={True, False}
     - when enabled, `write.scandata`_ will add extra detail into the FITS
       outputs such as channel gains, weights, flags, spectral filtering,
       profiles, and residual noise power spectra.  See `write.scandata`_.

   * - .. _write.spectrum:

       **write.spectrum**
     - | [write]
       | [[spectrum]]
       | value=<window>
     - Writes channel spectra (of residuals) into an ASCII table.  The optional
       argument <window> can specify a window function to use.  This is passed
       into the Python function scipy.signal.welch in the 'window' parameter.
       Please see scipy.signal.get_window for a list of available window types.
       The default is 'hamming'.

       The first column in the output file indicated the frequency, after which
       come the power-spectral-densities (PSF) of each channel used in the
       reduction.  See noslim_ and `write.ascii`_.

   * - .. _write.spectrum.size:

       **write.spectrum.size**
     - | [write]
       | [[spectrum]]
       | size=<N>
     - Specify the window size (in powers of 2) to use for measuring spectra.
       By default, the spectral range is set by the 1/f filtering timescale
       (drifts_).

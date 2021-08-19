Models should be in Spextool format, in directories named for the object modeled.
If more than one model is present, it should contain an date/hour of
applicability, in UT, as YYYYMMDDHH. For example, if a model of Alpha Boo should
be used for all times near 20150129, hour 8, it should be named 
alphaboo_2015012908_*.fits, and placed in a directory named alphaboo.

Note that a full set of reference models is included in the source
distribution of this package, but not in the pip or conda versions.
They may be downloaded separately, if desired, from the
`GitHub repository <https://github.com/SOFIA-USRA/sofia_redux>`__.

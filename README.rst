SOFIA Data Reduction Pipelines
==============================

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://zenodo.org/badge/311773000.svg
    :target: https://zenodo.org/badge/latestdoi/311773000
    :alt: DOI Badge


SOFIA Redux (`sofia_redux`) contains data processing pipelines and algorithms
for instruments on the Stratospheric Observatory for Infrared Astronomy
(SOFIA).

Currently, the SOFIA instruments supported by this package are the FORCAST
and FLITECAM imaging and spectroscopic instruments, the FIFI-LS integral field
spectrometer, and the HAWC+ imaging and polarimetric
instrument.

SOFIA raw and processed data can be accessed from the
`SOFIA archive <https://irsa.ipac.caltech.edu/applications/sofia/>`__.
Archived data may not match the results of data processed
with this pipeline software.  Questions specific to particular data sets
should be directed to the `SOFIA helpdesk <sofia_help@sofia.usra.edu>`__.

SOFIA pipelines are developed internally by the USRA/SOFIA data processing
software team, then are published publicly at the
`SOFIA Redux GitHub project
<https://github.com/SOFIA-USRA/sofia_redux>`__.
Contributions and feedback are welcome via the GitHub project, but
merge requests cannot be directly accepted.  They will be internally reviewed,
and pushed to the public site as needed.

For more information about installing and using this package, see
the `online documentation <https://SOFIA-USRA.github.io/sofia_redux/>`__,
or `docs/install.rst <docs/install.rst>`__ in the source distribution.

License
-------

This project is Copyright (c) Universities Space Research Association
under Government Prime Contract Number NNA17BF53C and licensed under
the terms of the BSD 3-Clause license.

This package is released under the BSD 3-clause license by written permission
from Dr. Attila Kovacs, copyright holder of the CRUSH software, which is the
original work of authorship for the sofia_redux.scan module.

This package is also based upon the
`Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.
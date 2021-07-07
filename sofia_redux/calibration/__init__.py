# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = []

if not _ASTROPY_SETUP_:   # noqa
    import sofia_redux.calibration.pipecal_applyphot
    import sofia_redux.calibration.pipecal_calfac
    import sofia_redux.calibration.pipecal_config
    import sofia_redux.calibration.pipecal_error
    import sofia_redux.calibration.pipecal_fitpeak
    import sofia_redux.calibration.pipecal_photometry
    import sofia_redux.calibration.pipecal_rratio
    import sofia_redux.calibration.pipecal_util

    import sofia_redux.calibration.standard_model
    import sofia_redux.calibration.standard_model.hawc_calibration
    import sofia_redux.calibration.standard_model.hawc_calib
    import sofia_redux.calibration.standard_model.genastmodel2
    import sofia_redux.calibration.standard_model.horizons
    import sofia_redux.calibration.standard_model.modconvert
    import sofia_redux.calibration.standard_model.thermast
    import sofia_redux.calibration.standard_model.isophotal_wavelength
    import sofia_redux.calibration.standard_model.background
    import sofia_redux.calibration.standard_model.calibration_io
    import sofia_redux.calibration.standard_model.derived_optics

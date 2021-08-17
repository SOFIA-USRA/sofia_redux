# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reduction objects for SOFIA instruments."""

if not _ASTROPY_SETUP_:
    from sofia_redux.pipeline.sofia.sofia_chooser import SOFIAChooser
    from sofia_redux.pipeline.sofia.sofia_configuration \
        import SOFIAConfiguration
    from sofia_redux.pipeline.sofia.sofia_pipe import main as redux_pipe
    from sofia_redux.pipeline.sofia.sofia_app import main as redux_app

    try:
        from sofia_redux.pipeline.sofia.fifils_reduction import FIFILSReduction
        from sofia_redux.pipeline.sofia.parameters.fifils_parameters \
            import FIFILSParameters
    except ImportError:
        pass

    try:
        from sofia_redux.pipeline.sofia.forcast_reduction \
            import FORCASTReduction
        from sofia_redux.pipeline.sofia.forcast_imaging_reduction \
            import FORCASTImagingReduction
        from sofia_redux.pipeline.sofia.parameters.forcast_parameters \
            import FORCASTParameters
        from sofia_redux.pipeline.sofia.parameters.forcast_imaging_parameters \
            import FORCASTImagingParameters
    except ImportError:
        pass

    try:
        from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
            import FORCASTSpectroscopyReduction
        from sofia_redux.pipeline.sofia.forcast_wavecal_reduction \
            import FORCASTWavecalReduction
        from sofia_redux.pipeline.sofia.forcast_spatcal_reduction \
            import FORCASTSpatcalReduction
        from sofia_redux.pipeline.sofia.forcast_slitcorr_reduction \
            import FORCASTSlitcorrReduction
        from sofia_redux.pipeline.sofia.parameters\
            .forcast_spectroscopy_parameters \
            import FORCASTSpectroscopyParameters
        from sofia_redux.pipeline.sofia.parameters\
            .forcast_wavecal_parameters \
            import FORCASTWavecalParameters
        from sofia_redux.pipeline.sofia.parameters \
            .forcast_spatcal_parameters \
            import FORCASTSpatcalParameters
        from sofia_redux.pipeline.sofia.parameters \
            .forcast_slitcorr_parameters \
            import FORCASTSlitcorrParameters
    except ImportError:
        pass

    try:
        from sofia_redux.pipeline.sofia.hawc_reduction import HAWCReduction
        from sofia_redux.pipeline.sofia.parameters.hawc_parameters \
            import HAWCParameters
    except ImportError:
        pass

    try:
        from sofia_redux.pipeline.sofia.flitecam_reduction \
            import FLITECAMReduction
        from sofia_redux.pipeline.sofia.flitecam_imaging_reduction \
            import FLITECAMImagingReduction
        from sofia_redux.pipeline.sofia.parameters.flitecam_parameters \
            import FLITECAMParameters
        from sofia_redux.pipeline.sofia.parameters.flitecam_imaging_parameters \
            import FLITECAMImagingParameters
    except ImportError:
        pass

    try:
        from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction \
            import FLITECAMSpectroscopyReduction
        from sofia_redux.pipeline.sofia.flitecam_wavecal_reduction \
            import FLITECAMWavecalReduction
        from sofia_redux.pipeline.sofia.flitecam_spatcal_reduction \
            import FLITECAMSpatcalReduction
        from sofia_redux.pipeline.sofia.flitecam_slitcorr_reduction \
            import FLITECAMSlitcorrReduction
        from sofia_redux.pipeline.sofia.parameters\
            .flitecam_spectroscopy_parameters \
            import FLITECAMSpectroscopyParameters
        from sofia_redux.pipeline.sofia.parameters\
            .flitecam_wavecal_parameters \
            import FLITECAMWavecalParameters
        from sofia_redux.pipeline.sofia.parameters \
            .flitecam_spatcal_parameters \
            import FLITECAMSpatcalParameters
        from sofia_redux.pipeline.sofia.parameters \
            .flitecam_slitcorr_parameters \
            import FLITECAMSlitcorrParameters
    except ImportError:
        pass
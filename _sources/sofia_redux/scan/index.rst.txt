**********************************************************
sofia_redux.scan: Scan Mode Algorithms for SOFIA Pipelines
**********************************************************

The :mod:`sofia_redux.scan` package contains supporting algorithms
and helper functions for reconstructing image maps from scanned
observations.

Description
===========

.. toctree::
   :maxdepth: 2

   scan_description


Architecture
============

.. toctree::
   :maxdepth: 2

   scan_architecture


Usage
=====

.. toctree::
   :maxdepth: 2

   scan_usage
   scan_configuration


Reference/API
=============

sofia_redux.scan.channels
-------------------------

.. automodapi:: sofia_redux.scan.channels.channels
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.channel_numba_functions
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.camera.camera
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.camera.color_arrangement
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.camera.single_color_arrangement
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.channel_data.channel_data
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.channel_data.color_arrangement_data
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.channel_data.single_color_channel_data
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.channel_group.channel_group
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.division.division
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.gain_provider.gain_provider
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.gain_provider.field_gain_provider
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.gain_provider.sky_gradient
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.gain_provider.zero_mean_gains
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.modality.modality
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.modality.correlated_modality
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.modality.coupled_modality
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.modality.non_linear_modality
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.mode.mode
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.response
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.acceleration_response
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.chopper_response
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.correlated_mode
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.field_response
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.motion_response
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.non_linear_response
   :headings: ~^
.. automodapi:: sofia_redux.scan.channels.mode.pointing_response
   :headings: ~^

.. automodapi:: sofia_redux.scan.channels.rotating.rotating
   :headings: ~^

sofia_redux.scan.chopper
------------------------

.. automodapi:: sofia_redux.scan.chopper.chopper
   :headings: ~^
.. automodapi:: sofia_redux.scan.chopper.chopper_numba_functions
   :headings: ~^

sofia_redux.scan.configuration
------------------------------
.. automodapi:: sofia_redux.scan.configuration.configuration
   :headings: ~^
.. automodapi:: sofia_redux.scan.configuration.aliases
   :headings: ~^
.. automodapi:: sofia_redux.scan.configuration.conditions
   :headings: ~^
.. automodapi:: sofia_redux.scan.configuration.dates
   :headings: ~^
.. automodapi:: sofia_redux.scan.configuration.fits
   :headings: ~^
.. automodapi:: sofia_redux.scan.configuration.iterations
   :headings: ~^
.. automodapi:: sofia_redux.scan.configuration.objects
   :headings: ~^
.. automodapi:: sofia_redux.scan.configuration.options
   :headings: ~^

sofia_redux.scan.coordinate_systems
-----------------------------------
.. automodapi:: sofia_redux.scan.coordinate_systems.coordinate_system
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.coordinate_systems_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.cartesian_system
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.coordinate
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.coordinate_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.coordinate_axis
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.index_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.offset_2d
   :headings: ~^

.. automodapi:: sofia_redux.scan.coordinate_systems.celestial_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.ecliptic_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.equatorial_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.focal_plane_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.galactic_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.geocentric_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.geodetic_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.horizontal_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.precessing_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.spherical_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.super_galactic_coordinates
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.telescope_coordinates
   :headings: ~^

.. automodapi:: sofia_redux.scan.coordinate_systems.epoch.epoch
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.epoch.precession
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.epoch.precession_numba_functions
   :headings: ~^

.. automodapi:: sofia_redux.scan.coordinate_systems.grid.grid
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.grid.grid_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.grid.flat_grid_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.grid.spherical_grid
   :headings: ~^

.. automodapi:: sofia_redux.scan.coordinate_systems.projection.projection_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.projection_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.default_projection_2d
   :headings: ~^

.. automodapi:: sofia_redux.scan.coordinate_systems.projection.bonnes_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.cylindrical_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.cylindrical_equal_area_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.cylindrical_perspective_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.global_sinusoidal_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.gnomonic_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.hammer_aitoff_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.mercator_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.parabolic_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.plate_carree_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.polyconic_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.sanson_flamsteed_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.slant_orthographic_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.spherical_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.stereographic_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.zenithal_equal_area_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.zenithal_equidistant_projection
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projection.zenithal_projection
   :headings: ~^

.. automodapi:: sofia_redux.scan.coordinate_systems.projector.projector_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.coordinate_systems.projector.astro_projector
   :headings: ~^


sofia_redux.scan.custom
-----------------------
.. automodapi:: sofia_redux.scan.custom.example.channels.channels
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.channels.channel_data.channel_data
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.channels.channel_group.channel_group
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.example.flags.channel_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.flags.frame_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.frames.frames
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.example.info.info
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.info.astrometry
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.info.detector_array
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.info.instrument
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.info.observation
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.info.telescope
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.example.integration.integration
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.example.scan.scan
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.hawc_plus.channels.channels
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.channels.channel_data.channel_data
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.channels.channel_group.channel_group
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.channels.gain_provider.pol_imbalance
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.channels.mode.los_response
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.channels.mode.roll_response
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.hawc_plus.flags.channel_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.flags.frame_flags
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.hawc_plus.frames.frames
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.frames.hawc_plus_frame_numba_functions
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.hawc_plus.info.info
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.info.astrometry
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.info.chopping
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.info.detector_array
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.info.instrument
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.info.observation
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.info.telescope
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.hawc_plus.integration.integration
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.integration.hawc_integration_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.hawc_plus.scan.scan
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.hawc_plus.simulation.simulation
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.sofia.channels.camera
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.channels.channel_data.channel_data
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.sofia.flags.quality_flags
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.sofia.frames.frames
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.sofia.info.info
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.sofia_info_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.aircraft
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.astrometry
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.chopping
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.detector_array
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.dithering
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.environment
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.extended_scanning
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.gyro_drifts
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.instrument
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.mapping
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.mission
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.mode
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.nodding
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.observation
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.origination
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.processing
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.scanning
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.spectroscopy
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.info.telescope
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.sofia.integration.integration
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.integration.models.atran
   :headings: ~^
.. automodapi:: sofia_redux.scan.custom.sofia.scan.scan
   :headings: ~^

.. automodapi:: sofia_redux.scan.custom.sofia.simulation.aircraft
   :headings: ~^

sofia_redux.scan.filters
------------------------
.. automodapi:: sofia_redux.scan.filters.filter
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.filters_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.adaptive_filter
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.fixed_filter
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.kill_filter
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.motion_filter
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.multi_filter
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.varied_filter
   :headings: ~^
.. automodapi:: sofia_redux.scan.filters.whitening_filter
   :headings: ~^

sofia_redux.scan.flags
----------------------
.. automodapi:: sofia_redux.scan.flags.flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.flag_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.array_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.channel_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.flagged_array
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.flagged_data
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.flagged_data_group
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.frame_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.instrument_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.map_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.motion_flags
   :headings: ~^
.. automodapi:: sofia_redux.scan.flags.mounts
   :headings: ~^

sofia_redux.scan.frames
-----------------------
.. automodapi:: sofia_redux.scan.frames.frames
   :headings: ~^
.. automodapi:: sofia_redux.scan.frames.frames_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.frames.horizontal_frames
   :headings: ~^

sofia_redux.scan.info
---------------------
.. automodapi:: sofia_redux.scan.info.info
   :headings: ~^
.. automodapi:: sofia_redux.scan.info.astrometry
   :headings: ~^
.. automodapi:: sofia_redux.scan.info.base
   :headings: ~^
.. automodapi:: sofia_redux.scan.info.instrument
   :headings: ~^
.. automodapi:: sofia_redux.scan.info.observation
   :headings: ~^
.. automodapi:: sofia_redux.scan.info.origination
   :headings: ~^
.. automodapi:: sofia_redux.scan.info.telescope
   :headings: ~^
.. automodapi:: sofia_redux.scan.info.weather_info
   :headings: ~^

sofia_redux.scan.integration
----------------------------
.. automodapi:: sofia_redux.scan.integration.integration
   :headings: ~^
.. automodapi:: sofia_redux.scan.integration.integration_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.integration.dependents.dependents
   :headings: ~^

sofia_redux.scan.pipeline
-------------------------
.. automodapi:: sofia_redux.scan.pipeline.pipeline
   :headings: ~^

sofia_redux.scan.reduction
--------------------------
.. automodapi:: sofia_redux.scan.reduction.reduction
   :headings: ~^
.. automodapi:: sofia_redux.scan.reduction.version
   :headings: ~^

sofia_redux.scan.scan
---------------------
.. automodapi:: sofia_redux.scan.scan.scan
   :headings: ~^

sofia_redux.scan.signal
-----------------------
.. automodapi:: sofia_redux.scan.signal.signal
   :headings: ~^
.. automodapi:: sofia_redux.scan.signal.signal_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.signal.correlated_signal
   :headings: ~^

sofia_redux.scan.simulation
---------------------------
.. automodapi:: sofia_redux.scan.simulation.info
   :headings: ~^

.. automodapi:: sofia_redux.scan.simulation.scan_patterns.constant_speed
   :headings: ~^
.. automodapi:: sofia_redux.scan.simulation.scan_patterns.daisy
   :headings: ~^
.. automodapi:: sofia_redux.scan.simulation.scan_patterns.lissajous
   :headings: ~^
.. automodapi:: sofia_redux.scan.simulation.scan_patterns.skydip
   :headings: ~^

.. automodapi:: sofia_redux.scan.simulation.source_models.simulated_source
   :headings: ~^
.. automodapi:: sofia_redux.scan.simulation.source_models.single_gaussian
   :headings: ~^
.. automodapi:: sofia_redux.scan.simulation.source_models.sky
   :headings: ~^

sofia_redux.scan.source_models
------------------------------
.. automodapi:: sofia_redux.scan.source_models.source_model
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.source_numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.astro_data_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.astro_intensity_map
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.astro_model_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.pixel_map
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.sky_dip
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.sky_dip_model
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.beams.asymmetry_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.beams.elliptical_source
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.beams.gaussian_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.beams.gaussian_source
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.beams.instant_focus
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.fits_properties.fits_properties
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.exposure_map
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.fits_data
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.image
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.image_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.map_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.noise_map
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.observation_2d
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.overlay
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.significance_map
   :headings: ~^
.. automodapi:: sofia_redux.scan.source_models.maps.weight_map
   :headings: ~^

sofia_redux.scan.utilities
--------------------------
.. automodapi:: sofia_redux.scan.utilities.utils
   :headings: ~^
.. automodapi:: sofia_redux.scan.utilities.numba_functions
   :headings: ~^
.. automodapi:: sofia_redux.scan.utilities.bracketed_values
   :headings: ~^
.. automodapi:: sofia_redux.scan.utilities.class_provider
   :headings: ~^
.. automodapi:: sofia_redux.scan.utilities.range
   :headings: ~^

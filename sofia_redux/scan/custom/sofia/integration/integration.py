# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
import numpy as np

from sofia_redux.scan.custom.sofia.integration.models.atran import AtranModel
from sofia_redux.scan.integration.integration import Integration

__all__ = ['SofiaIntegration']


class SofiaIntegration(Integration):

    def __init__(self, scan=None):
        super().__init__(scan=scan)

    def set_tau(self, spec=None, value=None):
        """
        Set the tau values for the integration.

        If a value is explicitly provided without a specification, will be used
        to set the zenith tau if ground based, or transmission.  If a
        specification and value is provided, will set the zenith tau as:

        ((band_a / t_a) * (value - t_b)) + band_b

        where band_a/b are retrieved from the configuration as tau.<spec>.a/b,
        and t_a/b are retrieved from the configuration as tau.<instrument>.a/b.

        Parameters
        ----------
        spec : str, optional
            The tau specification to read from the configuration.  If not
            supplied, will be read from the configuration 'tau' setting.
        value : float, optional
            The tau value to set.  If not supplied, will be retrieved from the
            configuration as tau.<spec>.

        Returns
        -------
        None
        """
        if spec is None:
            if not self.configuration.is_configured('tau'):
                return

        if spec is None and value is None:
            spec = self.configuration.get_string('tau').lower()

        if spec == 'atran':
            self.set_atran_tau()
        elif spec == 'pwvmodel':
            self.set_pwv_model_tau()
        else:
            super().set_tau(spec=spec, value=value)

    def set_atran_tau(self):
        """
        Set tau based on the ATRAN model.

        Returns
        -------
        None
        """
        model = AtranModel(self.configuration.get_options('atran'))
        altitude = self.info.aircraft.altitude.midpoint
        elevation = self.get_mid_elevation()
        c = model.get_relative_transmission(altitude, elevation)
        log.debug(f"Applying ATRAN model atmospheric correction: {c:.3f}")

        tau = -np.log(model.reference_transmission * c)
        tau *= np.sin(elevation).value
        self.set_tau_value(tau)

    def set_pwv_model_tau(self):
        """
        Set tau based on the PWV (precipitable water vapor) model.

        Returns
        -------
        None
        """
        pwv = self.get_model_pwv().to('micrometer').value
        log.debug(f"Using PWV model to correct fluxes: PWV = {pwv:.1f} um")
        self.configuration.parse_key_value('tau.pwv', str(pwv.value))
        self.set_tau(spec='pwv', value=pwv.to('micrometer').value)

    def get_modulation_frequency(self, signal_flag):
        """
        Return the modulation frequency.

        The modulation frequency is taken from the chopper frequency if
        available, or set to 0 Hz otherwise.

        Parameters
        ----------
        signal_flag : FrameFlagTypes or str or int
            The signal flag (not relevant for this implementation).

        Returns
        -------
        frequency : astropy.units.Quantity.
            The modulation frequency in Hz.
        """
        if self.info.chopping.chopping:
            return self.info.chopping.frequency
        return super().get_modulation_frequency(signal_flag)

    def get_mean_pwv(self):
        """
        Get the mean PWV of the integration.

        Returns
        -------
        pwv : astropy.units.Quantity
            The mean PWV.
        """
        pwv = self.frames.pwv[self.frames.valid]
        if pwv.size == 0 or np.isnan(pwv).all():
            return np.nan * units.Unit('micrometer')
        return np.nanmean(self.frames.pwv[self.frames.valid])

    def get_mid_elevation(self):
        """
        Return the mid-point elevation of the integration.

        The mid-point is defined as the average of the elevations of the
        first and last valid integration frame.

        Returns
        -------
        elevation : astropy.units.Quantity
        """
        el0 = self.frames.horizontal.el[self.get_first_frame_index()]
        el1 = self.frames.horizontal.el[self.get_last_frame_index()]
        return 0.5 * (el0 + el1)

    def get_mean_chopper_position(self):
        """
        Return the mean chopper position for the integration.

        Returns
        -------
        position : Coordinate2D
            The mean (x, y) chopper position.
        """
        return self.frames.chopper_position[self.frames.valid].mean()

    def get_model_pwv(self):
        """
        Estimate the PWV from the altitude of the observation.

        Returns
        -------
        PWV : astropy.units.Quantity
            The derived PWV.
        """
        log.debug("Estimating PWV based on altitude...")
        pwv41k = self.configuration.get_float(
            'pwv41k', default=29.0) * units.Unit('micrometer')
        b = 1.0 / self.configuration.get_float('pwvscale', default=5.0)
        alt_kf = (self.info.aircraft.altitude.midpoint / 1000).value
        return pwv41k * np.exp(-b * (alt_kf - 41.0))

    def validate(self):
        """
        Validate the integration after a read.

        Returns
        -------
        None
        """
        self.validate_pwv()
        super().validate()

    def validate_pwv(self):
        """
        Validate the frame PWV set a new value in the configuration if
        necessary.

        Returns
        -------
        None
        """
        pwv = self.get_mean_pwv()
        if pwv == 0 or np.isnan(pwv):
            log.debug("--> FIX: Using default PWV model...")
            pwv = self.get_model_pwv()

        log.debug(f"PWV: {pwv:.3f}")
        self.configuration.parse_key_value('tau.pwv', str(pwv.value))

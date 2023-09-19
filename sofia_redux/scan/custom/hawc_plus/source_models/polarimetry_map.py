# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy import wcs as astwcs
from astropy.io import fits
import numpy as np
import os

from sofia_redux.scan.custom.hawc_plus.flags.frame_flags import \
    HawcPlusFrameFlags
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.custom.hawc_plus.flags.polarimetry_flags import \
    HawcPlusPolarModulation
from sofia_redux.scan.source_models.polarimetry_map import PolarimetryMap
from sofia_redux.scan.source_models.source_numba_functions import (
    calculate_polarized_power)

__all__ = ['HawcPlusPolarimetryMap']


class HawcPlusPolarimetryMap(PolarimetryMap):

    polarimetry_flags = HawcPlusPolarModulation

    def __init__(self, info, reduction=None):
        """
        Initialize a polarization map for HAWC+.

        This is the parent class for `HawcPlusPolarimetryMapSeparateRt` and
        `HawcPlusPolarimetryMapDirect`, and should not be used to represent
        a polarimetry source model for HAWC+.

        Both HAWC+ polarization maps assume HWP observations consisting of
        sets of 4 scans where the HWP angle separation between each is ~22.5
        degrees and contain data from both the R0 and T0 subarray channels.
        No checking is done to see whether this is the case, so it is
        incumbent upon the user to supply scans accordingly.  The order of the
        scans doesn't matter, as long as it contains sets of scans following
        the above rules on the same area of the sky.

        Parameters
        ----------
        info : sofia_redux.scan.info.info.Info
            The Info object which should belong to this source model.
        reduction : sofia_redux.scan.reduction.reduction.Reduction, optional
            The reduction for which this source model should be applied.
        """
        self.i = AstroIntensityMap(info, reduction=reduction)
        super().__init__(info, reduction=reduction)
        self.hwp_scaled = False
        self.efficiency_applied = False
        self.id = 'PMP'
        self.total_scans = 0
        self.hwp_sets = 0

    def copy(self, with_contents=True):
        """
        Return a copy of the polarimetry model.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, return a true copy of the map.  Otherwise, just return
            a map with basic metadata.

        Returns
        -------
        PolarimetryMap
        """
        new = super().copy(with_contents=with_contents)
        new.i = self.i.copy(with_contents=with_contents)
        return new

    def set_info(self, info):
        """
        Set the Info object for the source model.

        This sets the provided `info` as the primary Info object containing
        the configuration and reduction information for the source model.
        The source model will also take ownership of the `info` and set
        various parameters from the contents.

        Parameters
        ----------
        info : sofia_redux.info.info.Info

        Returns
        -------
        None
        """
        self.i.set_info(info)
        super().set_info(info)

    def set_scans(self, scans):
        """
        Set the scans for this model by reference.

        Nothing fancy, but important for child classes.

        Parameters
        ----------
        scans : list (Scan)

        Returns
        -------
        None
        """
        self.i.set_scans(scans)
        super().set_scans(scans)

    def purge_artifacts(self):
        """
        Generally used to remove all data but that relevant to the model.

        Returns
        -------
        None
        """
        self.i.purge_artifacts()
        super().purge_artifacts()

    @property
    def referenced_attributes(self):
        """
        Return attributes that should be referenced during a copy.

        Bypasses the standard copy algorithm for speed by referencing I,Q,U
        maps which are individually copied by the above copy method.

        Returns
        -------
        set (str)
        """
        referenced = super().referenced_attributes
        referenced.add('i')
        return referenced

    @property
    def use_polarization(self):
        """
        Return whether to use polarization from the configuration.

        Returns
        -------
        bool
        """
        if self.configuration.has_option('polarization.enable'):
            enable_polarization = self.configuration.get_bool(
                'polarization.enable')
            if not enable_polarization:
                return False
        return super().use_polarization

    def create_from(self, scans, assign_scans=True):
        """
        Initialize model from scans.

        Sets the model scans to those provided, and the source model for each
        scan as this.  All integration gains are normalized to the first scan.
        If the first scan is non-sidereal, the system will be forced to an
        equatorial frame.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans from which to create the model.
        assign_scans : bool, optional
            If `True`, assign the scans to this source model.  Otherwise,
            there will be no hard link between the scans and source model.

        Returns
        -------
        None
        """
        super().create_from(scans, assign_scans=assign_scans)
        self.total_scans = len(scans)
        if self.total_scans % 4 != 0:
            log.warning("Number of scans is not a multiple of 4")
        self.hwp_sets = self.total_scans // 4
        self.n.enable_weighting = True
        self.q.enable_weighting = True
        self.u.enable_weighting = True
        self.i = self.n.copy()
        self.i.signal_mode = HawcPlusFrameFlags.flags.TOTAL_POWER
        self.i.stand_alone()
        self.i.id = 'I'

    def clear_process_brief(self):
        """
        Remove all process brief information.

        Returns
        -------
        None
        """
        super().clear_process_brief()
        self.i.clear_process_brief()

    def set_base(self):
        """
        Set the base to the map (copy of).

        Returns
        -------
        None
        """
        self.n.set_base()
        if self.use_polarization:
            self.q.set_base()
            self.u.set_base()

    def post_process_scan(self, scan):
        """
        Perform post-processing steps on a scan.

        Performs the post processing steps on the unpolarized N Stokes map.
        This typically involves determining the pointing correction for the
        source. Processing steps may affect properties of *this* map, but
        should not impact the primary reduction source polarization model.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        self.n.post_process_scan(scan)

    def sync_integration(self, integration, signal_mode=None):
        """
        Remove the source model from integration frame data.

        Parameters
        ----------
        integration : Integration
        signal_mode : FrameFlagTypes, optional
            The signal mode flag, indicating which signal should be used to
            extract the frame source gains.  Typically, TOTAL_POWER.

        Returns
        -------
        None
        """
        self.n.sync_integration(integration)
        if self.use_polarization:
            self.q.sync_integration(integration)
            self.u.sync_integration(integration)

    def get_polarized_fraction(self, polarized_power, total_power,
                               accuracy=None):
        """
        Return the polarized fraction.

        Returns a polarized fraction map in percentage units.

        Parameters
        ----------
        polarized_power : AstroIntensityMap
            The polarized power map.
        total_power : AstroIntensityMap
            The total power map.
        accuracy : float, optional
            The fractional accuracy.  Anything less than this will be flagged
            in the output product.  The default is 3 percent.

        Returns
        -------
        polarized_fraction : AstroIntensityMap
        """
        fraction = super().get_polarized_fraction(
            polarized_power, total_power, accuracy=accuracy)
        fraction.map.scale(100)
        fraction.map.data = abs(fraction.map.data)
        fraction.map.set_unit(units.Unit('percent'))
        return fraction

    def get_angles(self, polarized_power, polarized_fraction):
        """
        Return the polarization angle map.

        The angles are measured East of North.

        Parameters
        ----------
        polarized_power : AstroIntensityMap
            The polarized power map.
        polarized_fraction : AstroIntensityMap
            The polarized fraction map.

        Returns
        -------
        angles : AstroIntensityMap
        """
        angles = super().get_angles(polarized_power, polarized_fraction)
        angles.enable_level = False
        angles.enable_weighting = False
        angles.enable_bias = False
        return angles

    @staticmethod
    def get_rotated_angles(angles):
        """
        Apply rotation to the polarization angles.

        Parameters
        ----------
        angles : AstroIntensityMap

        Returns
        -------
        rotated_angles : AstroIntensityMap
        """
        rot = angles.copy()
        a = rot.map.data.copy()
        a += 90
        a[a > 90] -= 180
        rot.map.data = a
        rot.map.id = 'A_ROT'
        return rot

    def write(self, path):
        """
        Write the maps.

        If polarization.write=redux in the configuration, then the generated
        HDU list will be suitable for subsequent redux processing.  Otherwise,
        a standard SOFSCAN polarization output map will be generated.  If this
        is the case, it is HIGHLY recommended that a final smooth
        (configuration keyword = smooth.final) is performed in order to
        acquire more polarization vectors.

        Parameters
        ----------
        path : str
            The directory to write to.

        Returns
        -------
        None
        """
        write_method = self.configuration.get_string(
            'polarization.write', default='')
        if write_method.strip().lower() == 'redux':
            self.write_redux_rotate(path)
        else:
            self.write_full(path)

    def write_redux_rotate(self, path):
        """
        Write the source to file.

        This is designed to pass the ROT file to redux for subsequent
        redux processing.  The final HDU lst will contain the extensions::

          - STOKES I
          - ERROR I
          - STOKES Q
          - ERROR Q
          - STOKES U
          - ERROR U
          - COVAR Q I (will be blank)
          - COVAR U I (will be blank)
          - COVAR Q U (will be blank, and zero due to rotation here)
          - BAD PIXEL MASK (0 for good, 3 for bad)

        The generated HDU list will set at the "rotate" step for subsequent
        redux processing.

        Parameters
        ----------
        path : str
            The directory to write to.

        Returns
        -------
        None
        """
        self.process_final()
        covar_maps = self.get_covariance_maps()
        self.apply_efficiency(forward=False)  # Remove efficiency if applied

        valid = self.n.map.is_valid()
        valid &= self.q.map.is_valid()
        valid &= self.u.is_valid()

        # uncorrected_polarized_flux = self.get_p(debias=False)
        # total_power = self.get_i(p=uncorrected_polarized_flux,
        #                          add_np_weights=True)

        stokes_i = self.get_main_map_component()
        stokes_q = self.q.copy()
        stokes_u = self.u.copy()

        stokes_maps = [stokes_i, stokes_q, stokes_u]
        if covar_maps['valid']:
            stokes_maps += [
                covar_maps['QI'], covar_maps['UI'], covar_maps['QU']]

        for stokes_map in stokes_maps:
            stokes_map.map.data[~valid] = np.nan
            stokes_map.map.weight.data[~valid] = np.nan
            stokes_map.map.exposure.data[~valid] = np.nan
            stokes_map.write(path)

        if covar_maps['valid']:
            qi_hdu = covar_maps['QI'].hdul[0]
            ui_hdu = covar_maps['UI'].hdul[0]
            qu_hdu = covar_maps['QU'].hdul[0]
        else:
            qi_hdu = fits.ImageHDU(np.zeros(self.shape, dtype=float))
            ui_hdu = fits.ImageHDU(np.zeros(self.shape, dtype=float))
            qu_hdu = fits.ImageHDU(np.zeros(self.shape, dtype=float))

        bad_pixel_mask = np.zeros(self.shape, dtype=int)
        bad_pixel_mask[~valid] = 3
        bad_pixel_hdu = fits.ImageHDU(data=bad_pixel_mask)

        self.hdul = fits.HDUList()
        for extname, hdu in [
                ('STOKES I', stokes_i.hdul[0]),
                ('ERROR I', stokes_i.hdul[2]),
                ('STOKES Q', stokes_q.hdul[0]),
                ('ERROR Q', stokes_q.hdul[2]),
                ('STOKES U', stokes_u.hdul[0]),
                ('ERROR U', stokes_u.hdul[2]),
                ('COVAR Q I', qi_hdu),
                ('COVAR U I', ui_hdu),
                ('COVAR Q U', qu_hdu),
                ('BAD PIXEL MASK', bad_pixel_hdu)]:
            hdu.header['EXTNAME'] = extname
            self.hdul.append(hdu)

        self.hdul[0].header['PRODTYPE'] = 'rotate'
        self.hdul[0].header['NHWP'] = 4
        self.hdul[0].header['PIXSCAL'] = self.configuration.get_float(
            'grid')

        file_name = os.path.join(
            path, f'{self.get_core_name()}.ROT.fits')

        if self.is_empty():
            source_name = ('ROT ' if self.id not in [None, ''] else '')
            log.warning(f"Source {source_name}is empty. Skipping")
            if os.path.isfile(file_name):
                os.remove(file_name)
            return

        self.write_fits(file_name)
        if self.configuration.get_bool('write.png'):
            self.write_png(self, file_name)

    def get_covariance_maps(self):
        """
        Return the covariance HDUs for the final write.

        Returns
        -------
        covariance_hdus : dict
            The {name: HDU} dictionary.  The name key should be one of
            {'QI', 'UI', 'QU'}.
        """
        return {
            'QI': None,
            'UI': None,
            'QU': None,
            'valid': False
        }

    def write_full(self, path):
        """
        Write the source to file.

        Performing a write operation will write various products to the
        `path` directory.  If any intermediate.<id>.fits file is found
        it will be deleted.  Unlike `write_rot`, this function will create an
        HDU list processed using internal SOFSCAN polarimetry calculations.

        Parameters
        ----------
        path : str
            The directory to write to.

        Returns
        -------
        None
        """
        self.process_final()
        self.apply_efficiency(forward=False)  # Remove efficiency if applied

        uncorrected_polarized_flux = self.get_p(debias=False)
        total_power = self.get_i(p=uncorrected_polarized_flux)

        uncorrected_fraction = self.get_polarized_fraction(
            uncorrected_polarized_flux, total_power, accuracy=np.inf)

        angles = self.get_angles(uncorrected_polarized_flux,
                                 uncorrected_fraction)
        rotated_angles = self.get_rotated_angles(angles)

        self.apply_efficiency(forward=True)

        polarized_flux = self.get_p(debias=False)
        debiased_polarized_flux = self.get_p(debias=True)
        percent = self.get_polarized_fraction(
            polarized_flux, total_power, accuracy=np.inf)
        debiased_percent = self.get_polarized_fraction(
            debiased_polarized_flux, total_power, accuracy=np.inf)

        debiased_percent_data = debiased_percent.map.data.copy()
        debiased_percent = percent.copy()
        debiased_percent.map.data = debiased_percent_data
        debiased_percent.id = 'DebiasedPercent'

        self.n.write(path)
        total_power.write(path)
        self.q.write(path)
        self.u.write(path)
        percent.write(path)
        debiased_percent.write(path)
        angles.write(path)
        rotated_angles.write(path)
        polarized_flux.write(path)
        debiased_polarized_flux.write(path)

        self.hdul = fits.HDUList()
        for extname, hdu in [
                ('STOKES I', total_power.hdul[0]),
                ('ERROR I', total_power.hdul[2]),
                ('STOKES N', self.n.hdul[0]),
                ('ERROR N', self.n.hdul[2]),
                ('STOKES Q', self.q.hdul[0]),
                ('ERROR Q', self.q.hdul[2]),
                ('STOKES U', self.u.hdul[0]),
                ('ERROR U', self.u.hdul[2]),
                ('PERCENT POL', percent.hdul[0]),
                ('DEBIASED PERCENT POL', debiased_percent.hdul[0]),
                ('ERROR PERCENT POL', percent.hdul[2]),
                ('POL ANGLE', angles.hdul[0]),
                ('ROTATED POL ANGLE', rotated_angles.hdul[0]),
                ('ERROR POL ANGLE', angles.hdul[2]),
                ('POL FLUX', polarized_flux.hdul[0]),
                ('ERROR POL FLUX', polarized_flux.hdul[2]),
                ('DEBIASED POL FLUX', debiased_polarized_flux.hdul[0])]:

            hdu.header['EXTNAME'] = extname
            self.hdul.append(hdu)

        ny, nx = self.shape
        y, x = np.mgrid[0:ny, 0:nx] + 1
        x = x.flatten()
        y = y.flatten()
        wcs = astwcs.WCS(self.n.hdul[0].header)
        ra, dec = wcs.wcs_pix2world(x, y, 1)  # zero-based input pixels

        # create table columns
        cols = [fits.Column(name="Pixel X", format='J', array=x),
                fits.Column(name="Pixel Y", format='J', array=y),
                fits.Column(name="Right Ascension", format='D',
                            array=ra, unit='deg'),
                fits.Column(name="Declination", format='D',
                            array=dec, unit='deg'),
                fits.Column(name="Percent Pol", format='D',
                            array=percent.hdul[0].data.flatten()),
                fits.Column(name="Debiased Percent Pol", format='D',
                            array=debiased_percent.hdul[0].data.flatten()),
                fits.Column(name="Err. Percent Pol", format='D',
                            array=percent.hdul[2].data.flatten()),
                fits.Column(name="Theta", format='D', unit='deg',
                            array=angles.hdul[0].data.flatten()),
                fits.Column(name="Rotated Theta", format='D', unit='deg',
                            array=rotated_angles.hdul[0].data.flatten()),
                fits.Column(name="Err. Theta", format='D', unit='deg',
                            array=angles.hdul[2].data.flatten())]

        c = fits.ColDefs(cols)
        tb_hdu = fits.BinTableHDU.from_columns(c)
        tb_hdu.header['EXTNAME'] = 'POL DATA'
        self.hdul.append(tb_hdu)

        # Remove the intermediate image file
        intermediate = os.path.join(path, f'intermediate.{self.id}.fits')
        if os.path.isfile(intermediate):
            os.remove(intermediate)

        # Remove the intermediate image file
        intermediate = os.path.join(path, f'intermediate.{self.id}.fits')
        if os.path.isfile(intermediate):
            os.remove(intermediate)

        if self.id not in [None, '']:
            file_name = os.path.join(
                path, f'{self.get_core_name()}.{self.id}.fits')
        else:
            file_name = os.path.join(path, f'{self.get_core_name()}.fits')

        if self.is_empty():
            source_name = ((self.id + ' ')
                           if self.id not in [None, ''] else '')
            log.warning(f"Source {source_name}is empty. Skipping")
            if os.path.isfile(file_name):
                os.remove(file_name)
            return

        self.write_fits(file_name)
        if self.configuration.get_bool('write.png'):
            self.write_png(self, file_name)

    def process(self):
        """
        Process the source model.

        The processing step will normalize source map data from `weight * data`
        values to `data` values.  This is usually followed by smoothing and
        filtering steps if required, and also masking certain areas of the
        source due to things like exposure time and S2N ranges.

        Returns
        -------
        None
        """
        self.apply_stokes_scaling(normalized=False)
        self.add_process_brief('[N]')
        self.n.process()
        if self.use_polarization:
            sign = self.configuration.get_string('source.sign')
            self.configuration.parse_key_value('source.sign', '0')
            self.add_process_brief('[Q]')
            self.q.process()
            self.add_process_brief('[U]')
            self.u.process()
            self.configuration.parse_key_value('source.sign', sign)
            # self.create_total_power_map(allow_invalid=True)
            # self.add_process_brief('[I]')
            # self.process_total_power_map()
            self.merge_all_flags()

    def apply_stokes_scaling(self, normalized=False):
        """
        Apply scaling to the Stokes maps to account for HWP observations.

        Parameters
        ----------
        normalized : bool, optional
            If `True`, indicates that the maps have already been normalized.
            Otherwise, the map data is assumed to be of the form data * weight.

        Returns
        -------
        None
        """
        self.apply_stokes_gain_scaling_to(self.n, normalized=normalized)
        if self.use_polarization:
            self.apply_stokes_gain_scaling_to(self.q, normalized=normalized)
            self.apply_stokes_gain_scaling_to(self.u, normalized=normalized)

    def process_final_main_map(self):
        """
        Perform the process final step on the main map

        Returns
        -------
        None
        """
        super().process_final()

    def process_final(self):
        """
        Perform the final processing steps.

        The additional steps performed for the AstroIntensityMap are
        map leveling (if not extended or deep) and map re-weighting.
        The map may also be resampled if re-griding is enabled.

        Returns
        -------
        None
        """
        self.n.process_final()
        if self.use_polarization:
            self.q.process_final()
            self.u.process_final()

        # self.reweight_products()

    def reset_processing(self):
        """
        Reset the source processing.

        Returns
        -------
        None
        """
        super().reset_processing()
        if self.use_polarization:
            self.i.reset_processing()

    def clear_content(self):
        """
        Clear the data.

        Returns
        -------
        None
        """
        super().clear_content()
        if self.use_polarization:
            self.i.clear_content()

    def merge_accumulate(self, other):
        """
        Merge another source with this one.

        Parameters
        ----------
        other : HawcPlusPolarimetryMap

        Returns
        -------
        None
        """
        super().merge_accumulate(other)
        self.i.map.merge_accumulate(other.i.map)

    # def process_final(self):
    #     """
    #     Perform the final processing steps.
    #
    #     The final processing steps are performed individually for the Stokes
    #     Q, U, and N maps.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     for stokes in [self.n, self.q, self.u]:
    #         stokes.enable_weighting = False
    #     self.n.process_final()
    #     self.q.process_final()
    #     self.u.process_final()
    #     self.create_total_power_map()

    def write_fits(self, filename):
        """
        Write the results to a FITS file.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        if self.hdul is None:
            return
        for hdu in self.hdul:
            hdu.header['FILENAME'] = filename, 'Name at creation'

        self.add_scan_hdus_to(self.hdul)
        if not self.configuration.get_bool('write.source'):
            return

        log.info(f'Writing to {filename}')
        self.hdul.writeto(filename, overwrite=True)
        self.hdul.close()

    def write_png(self, map_2d, file_name):
        """
        Write a PNG of the map.

        Parameters
        ----------
        map_2d : HawcPlusPolarimetryMap
        file_name : str
            The file path to write the PNG to.

        Returns
        -------
        None
        """
        if not self.configuration.get_bool('write.png'):
            return
        raise NotImplementedError("Build this if you want it, or process via"
                                  "redux (set polarization.write=redux in "
                                  "the configuration).")
        # Do fancy stuff here...

    def get_i(self, p=None, allow_invalid=False, add_np_weights=True,
              copy_self=False):
        """
        Return the total power map.

        Parameters
        ----------
        p : AstroIntensityMap, optional
            The polarized power map.  If not supplied, defaults to the
            polarized power map derived from this data.
        allow_invalid : bool, optional
            If `True`, does not pay attention to invalid Q/U map points.  This
            is important if they have been marked as invalid due to previous
            clipping operations that are irrelevant when creating the
            total intensity map.
        add_np_weights : bool, optional
           If `True`, calculate I weights by aggregating N and P weights.
           Otherwise, just use N weights (copy).
       copy_self: bool, optional
           If `True`, copy the I from the main map here.

        Returns
        -------
        AstroIntensityMap
        """
        if not copy_self:
            i = super().get_i(p=p, allow_invalid=allow_invalid,
                              add_np_weights=add_np_weights)
        else:
            i = self.get_main_map_component()

        i.enable_bias = False
        i.enable_level = False
        i.enable_weighting = False
        i.signal_mode = HawcPlusFrameFlags.flags.TOTAL_POWER
        return i

    def get_main_map_component(self):
        """
        Return a sub map of the main map containing stokes I data.

        Returns
        -------
        AstroIntensityMap
        """
        i = self.n.copy()
        i.clear_content()
        i.map.data = self.map.data.copy()
        i.map.weight.data = self.map.weight.data.copy()
        i.map.exposure.data = self.map.exposure.data.copy()
        i.id = 'I'
        i.map.validate()
        return i

    def get_p(self, debias=True, allow_invalid=False):
        """
        Return the polarized power map.

        Parameters
        ----------
        debias : bool, optional
            If `True`, apply Ricean debiasing.
        allow_invalid : bool, optional
            If `True`, does not pay attention to invalid Q/U map points.  This
            is important if they have been marked as invalid due to previous
            clipping operations that are irrelevant when creating the
            total intensity map (a dependent of this polarization map).

        Returns
        -------
        AstroIntensityMap
        """
        p_map = self.n.copy()
        p_map.clear_content()
        p = p_map.map
        discard_flag = p.flagspace.convert_flag('DISCARD').value
        p_map.enable_level = False
        p_map.enable_weighting = False
        p_map.enable_bias = False

        calculate_polarized_power(p=p.data,
                                  p_weight=p.weight.data,
                                  p_flag=p.flag,
                                  q=self.q.map.data,
                                  q_weight=self.q.map.weight.data,
                                  q_valid=self.q.map.is_valid(),
                                  u=self.u.map.data,
                                  u_weight=self.u.map.weight.data,
                                  u_valid=self.u.map.is_valid(),
                                  bad_flag=discard_flag,
                                  debias=debias,
                                  efficiency=1.0,
                                  allow_invalid=allow_invalid)
        if debias:
            p_map.id = 'DbP'
        else:
            p_map.id = 'P'

        p.validate()
        return p_map

    def apply_efficiency(self, forward=True):
        """
        Apply the polarization efficiency to the Q/U parameters

        Parameters
        ----------
        forward : bool, optional
            If `False`, unapply the efficiency correction

        Returns
        -------

        """
        if forward and self.efficiency_applied:
            return
        if not forward and not self.efficiency_applied:
            return
        efficiency = self.configuration.get_float('polarization.efficiency',
                                                  default=1.0)
        scale = 1.0 / efficiency if forward else efficiency
        self.q.map.scale(scale)
        self.u.map.scale(scale)
        self.efficiency_applied = forward

    def is_empty(self):
        """
        Return whether source map is empty.

        Returns
        -------
        bool
        """
        return self.count_points() == 0

    def create_total_power_map(self, allow_invalid=True):
        """
        Return a total power map generated using non-debiased polarization.

        Parameters
        ----------
        allow_invalid : bool, optional
            If `True`, does not pay attention to invalid Q/U map points.  This
            is important if they have been marked as invalid due to previous
            clipping operations that are irrelevant when creating the
            total intensity map (a dependent of this polarization map).

        Returns
        -------
        AstroIntensityMap
        """
        p = self.get_p(debias=False, allow_invalid=allow_invalid)
        total_power = self.get_i(p=p, allow_invalid=allow_invalid)
        self.i.map = total_power.map
        self.i.map.copy_processing_from(self.n.map)

    def process_total_power_map(self):
        """
        Perform the process step on the total power map.

        Returns
        -------
        None
        """
        self.i.map.data *= self.i.map.weight.data
        self.i.map.flag.fill(0)
        self.i.process()

    def merge_all_flags(self):
        """
        Merge masking flags for all maps

        Sets the map flags for all Stokes (N, Q, U) maps equal to the union
        of the flags of all Stokes maps.

        Returns
        -------
        None
        """
        flags = (self.n.map.flag | self.q.map.flag |
                 self.u.map.flag | self.i.map.flag)
        self.n.map.flag = flags.copy()
        self.q.map.flag = flags.copy()
        self.u.map.flag = flags.copy()
        self.i.map.flag = flags.copy()

    def apply_stokes_gain_scaling_to(self, stokes_map, normalized=False):
        """
        Apply Stokes gain scaling to a given map.

        For polarization maps, the rotation of the Stokes parameters occurs
        at the timestream level before being accumulated into map space.
        This is done at the gain calculation phase, and are applied via::

            gain_N = unpolarized_gain * frame_gain * N_scaling
            gain_Q = Q_rotation_gain * frame_gain * Q_scaling
            gain_U = U_rotation_gain * frame_gain * U_scaling

        Typically, the N, Q, and U scaling factors are 0.5, but for HAWC+
        HWP observations, may be different.

        For the maps where Stokes maps are updated directly, scaling are
        applied to weight (w) and accumulated data (d = weight * data) values
        as::

            {x}_weight_scaling = (4 * {x}_scaling)^(-2)
            dN_scaling = weight_scaling * N_scaling
            d{Q or U}_scaling = dN_scaling / sqrt(8)

        Notes
        -----
        Currently the actual value of {N,Q,U} scaling is irrelevant to the
        process and will not change the output values.  This is because
        *this* correction factor is applied post processing.  However, there
        is more than one way to accumulate/sync/process the Stokes parameters
        from the HAWC+ R and T data.  One such method is to apply *this*
        correction factor during accumulation, in which case the definition of
        these scaling factors becomes more important.  The gain scaling
        factors are defined in
        `sofia_redux.scan.custom.flags.polarimetry_flags`.

        Parameters
        ----------
        stokes_map : AstroIntensityMap
            The Stokes map for which to rescale according to the polarization
            gain scaling factor.
        normalized : bool, optional
            If `True`, indicates that the `stokes_map` data values have already
            been normalized from `weight * data` to `weight`.  If so, they
            must first be converted back to the accumulated state before
            scaling, and reconverted to normalized data once complete.

        Returns
        -------
        None
        """
        gain_scale = self.polarimetry_flags.get_gain_factor(
            stokes_map.signal_mode)

        # weight scaling notes
        # increase factor -> decrease noise
        # increase rounds -> decrease noise

        # 4 sets (RQ,RU,TQ,TU) x 4 (HWP angles) x 0.5 (polarization scaling)
        # For 4 scans (1 hwp):
        #    w = 0.5, s = 0.125
        # For 8 scans (2 hwp):
        #    w = ?, s = ?
        # For 12 scans (3 hwp):
        #    w = 0.125, s = 1

        hwp_factor = 2 ** (-self.hwp_sets)
        gain_factor = gain_scale ** -2

        weight_scale = gain_factor * hwp_factor
        # gain factor only affects weight, not value (divides out later)
        value_scale = gain_factor
        if stokes_map.signal_mode != self.polarimetry_flags.N:
            value_scale /= 16  # 8 * np.sqrt(2)  # Q/U
        else:
            value_scale *= hwp_factor  # N

        if normalized:
            stokes_map.map.data *= stokes_map.map.weight.data

        stokes_map.map.data *= value_scale
        stokes_map.map.weight.data *= weight_scale

        if normalized:
            stokes_map.end_accumulation()

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.source_models.maps.map_2d import Map2D
from sofia_redux.scan.source_models.maps.image_2d import Image2D
from sofia_redux.scan.flags.flagged_array import FlaggedArray
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.source_models.maps.weight_map import WeightMap
from sofia_redux.scan.source_models.maps.exposure_map import ExposureMap
from sofia_redux.scan.source_models.maps.noise_map import NoiseMap
from sofia_redux.scan.source_models.maps.significance_map \
    import SignificanceMap

__all__ = ['Observation2D']


class Observation2D(Map2D):

    def __init__(self, data=None, blanking_value=np.nan, dtype=float,
                 shape=None, unit=None, weight_dtype=float):
        self.weight = Image2D(dtype=weight_dtype)
        self.exposure = Image2D(dtype=weight_dtype)
        self.noise_rescale = 1.0
        self.is_zero_weight_valid = False
        self.weight_dtype = weight_dtype
        super().__init__(data=data, blanking_value=blanking_value, dtype=dtype,
                         shape=shape, unit=unit)
        shape = self.shape
        if shape != ():
            self.weight.shape = shape
            self.exposure.shape = shape

    def copy_processing_from(self, other):
        """
        Copy the processing from another map.

        Parameters
        ----------
        other : Observation2D

        Returns
        -------
        None
        """
        super().copy_processing_from(other)
        self.noise_rescale = other.noise_rescale

    def reset_processing(self):
        """
        Reset the processing status.

        Returns
        -------
        None
        """
        super().reset_processing()
        self.noise_rescale = 1.0

    @property
    def valid(self):
        """
        Return a boolean mask array of valid data elements.

        Valid elements are neither NaN, set to the blanking value, or
        flagged as the validating_flags.

        Returns
        -------
        numpy.ndarray (bool)
           A boolean mask where `True` indicates a valid element.
        """
        valid = self.is_valid()
        if self.size == 0:
            return valid
        if self.weight is None or self.weight.data is None:
            return valid
        valid &= np.isfinite(self.weight.data)
        if self.is_zero_weight_valid:
            valid &= self.weight.data >= 0
        else:
            valid &= self.weight.data > 0
        return valid

    def clear(self, indices=None):
        """
        Clear flags and set data to zero.  Clear history.

        Parameters
        ----------
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).

        Returns
        -------
        None
        """
        super().clear(indices=indices)
        if self.exposure is not None:
            self.exposure.clear(indices=indices)
        if self.weight is not None:
            self.weight.clear(indices=indices)

    def discard(self, indices=None):
        """
        Set the flags for discarded indices to DISCARD and data to zero.

        Parameters
        ----------
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).

        Returns
        -------
        None
        """
        super().discard(indices=indices)
        if self.exposure is not None:
            self.exposure.clear(indices)
        if self.weight is not None:
            self.weight.clear(indices)

    def destroy(self):
        """
        Destroy the image data.

        Returns
        -------
        None
        """
        super().destroy()
        if self.weight is not None:
            self.weight.destroy()
        if self.exposure is not None:
            self.exposure.destroy()

    def set_data_shape(self, shape):
        """
        Set the shape of the data, weight, and exposure images.

        Parameters
        ----------
        shape : tuple (int)

        Returns
        -------
        None
        """
        super().set_data_shape(shape)
        self.weight.set_data_shape(shape)
        self.claim_image(self.weight)
        self.exposure.set_data_shape(shape)
        self.claim_image(self.exposure)

    def to_weight_image(self, data):
        """
        Convert data to a weight image.

        Parameters
        ----------
        data : FlaggedArray or FitsData or numpy.ndarray or None

        Returns
        -------
        Image2D
        """
        if data is None:
            data = Image2D(x_size=self.shape[1],
                           y_size=self.shape[0],
                           blanking_value=self.blanking_value,
                           dtype=self.weight_dtype)
        elif isinstance(data, np.ndarray):
            data = Image2D(data=data,
                           blanking_value=self.blanking_value,
                           dtype=self.weight_dtype)
        return data

    def get_weights(self):
        """
        Return the weights overlay.

        Returns
        -------
        WeightMap
        """
        return WeightMap(self)

    def get_weight_image(self):
        """
        Return the weights image.

        Returns
        -------
        Image2D
        """
        return self.weight

    def weight_values(self):
        """
        Return the array of weights.

        Returns
        -------
        numpy.ndarray (float)
        """
        return self.get_weights().data

    def set_weight_image(self, weight_image):
        """
        Set the weight image.

        Parameters
        ----------
        weight_image : Image2D or numpy.ndarray or None

        Returns
        -------
        None
        """
        weight_image = self.to_weight_image(weight_image)
        self.weight = weight_image
        self.claim_image(weight_image)

    def get_exposures(self):
        """
        Return an exposure overlay.

        Returns
        -------
        ExposureMap
        """
        return ExposureMap(self)

    def exposure_values(self):
        """
        Return the array of weights.

        Returns
        -------
        numpy.ndarray (float)
        """
        return self.get_exposures().data

    def get_exposure_image(self):
        """
        Return the exposure image.

        Returns
        -------
        Image2D
        """
        return self.exposure

    def set_exposure_image(self, exposure_image):
        """
        Set the weight image.

        Parameters
        ----------
        exposure_image : Image2D or numpy.ndarray or None

        Returns
        -------
        None
        """
        exposure_image = self.to_weight_image(exposure_image)
        self.exposure = exposure_image
        self.claim_image(exposure_image)

    def get_noise(self):
        """
        Return a noise overlay.

        Returns
        -------
        NoiseMap
        """
        return NoiseMap(self)

    def noise_values(self):
        """
        Return the array of weights.

        Returns
        -------
        numpy.ndarray (float)
        """
        return self.get_noise().data

    def set_noise(self, noise_image):
        """
        Set the noise image.

        Parameters
        ----------
        noise_image : Image2D or Overlay or numpy.ndarray or None

        Returns
        -------
        None
        """
        self.get_noise().data = noise_image

    def get_significance(self):
        """
        Return a significance overlay.

        Returns
        -------
        SignificanceMap
        """
        return SignificanceMap(self)

    def significance_values(self):
        """
        Return the array of significance (S2N).

        Returns
        -------
        numpy.ndarray (float)
        """
        return self.get_significance().data

    def set_significance(self, significance_image):
        """
        Set the significance image.

        Parameters
        ----------
        significance_image : Image2D or Overlay or numpy.ndarray or None

        Returns
        -------
        None
        """
        self.get_significance().data = significance_image

    def scale(self, factor, indices=None):
        """
        Scale the data values and weights by a given factor.

        Parameters
        ----------
        factor : float
        indices : numpy.ndarray (bool), optional

        Returns
        -------
        None
        """
        super().scale(factor, indices=indices)
        self.get_weight_image().scale(1.0 / (factor ** 2), indices=indices)

    def crop(self, ranges):
        """
        Crop the image data.

        Parameters
        ----------
        ranges : numpy.ndarray (int,) or astropy.units.Quantity (numpy.ndarray)
            The ranges to set crop the data to.  Should be of shape
            (n_dimensions, 2) where ranges[0, 0] would give the minimum crop
            limit for the first dimension and ranges[0, 1] would give the
            maximum crop limit for the first dimension.  In this case, the
            'first' dimension is in FITS format.  i.e., (x, y) for a 2-D image.
            If a Quantity is supplied this should contain the min and max
            grid values to clip to in each dimension.

        Returns
        -------
        None
        """
        self.get_weight_image().crop(ranges)
        self.get_exposure_image().crop(ranges)
        super().crop(ranges)

    def accumulate(self, image, weight=1.0, gain=1.0, valid=None):
        """
        Add an observation image.

        Parameters
        ----------
        image : Observation2D
            The observation to add.
        weight : float, optional
            A global weighting factor for the entire image.  Typically the
            scan weight from which the image was derived.
        gain : float, optional
            A gain factor that will be applied to the image values.  It will
            be applied to the weighting factors as g^2 during accumulation.
        valid : numpy.ndarray (bool), optional
            An array where `False` excludes a datum from accumulation.

        Returns
        -------
        None
        """
        image_weight = image.get_weights().data * weight
        times = image.get_exposures().data
        if valid is None:
            valid = image.valid

        self.accumulate_at(image, gain, image_weight, times, indices=valid)

    def accumulate_at(self, image, gains, weights, times, indices=None):
        """
        Accumulate at given indices.

        The data are accumulated as: image * gains * weights
        The weights are accumulated as: weights * gains^2
        The exposures are accumulated as: times

        Parameters
        ----------
        image : FlaggedArray or numpy.ndarray or float
        gains : FlaggedArray or numpy.ndarray or float
        weights : FlaggedArray or numpy.ndarray or float
        times : FlaggedArray or numpy.ndarray or float
        indices : numpy.ndarray (bool or int), optional
            A boolean mask adds to those indices on self.data marked
            as `True`. If so, image/weights/times etc should be the same
            shape as self.data of scalar values.

        Returns
        -------
        None
        """
        if isinstance(image, FlaggedArray):
            image = image.data
        if isinstance(weights, FlaggedArray):
            weights = weights.data
        if isinstance(times, FlaggedArray):
            times = times.data
        if isinstance(gains, FlaggedArray):
            gains = gains.data

        wg = weights * gains
        add_data = wg * image
        add_weight = wg * gains
        add_time = times

        self.add(add_data, indices=indices)
        self.get_weight_image().add(add_weight, indices=indices)
        self.get_exposure_image().add(add_time, indices=indices)

    def merge_accumulate(self, image):
        """
        Merge and accumulate an image onto this one.

        Parameters
        ----------
        image : Observation2D

        Returns
        -------
        None
        """
        super().add(image)
        self.merge_properties_from(image)
        self.get_weight_image().add(image.get_weights())
        self.get_exposure_image().add(image.get_exposures())

    def end_accumulation(self):
        """
        End the accumulation process by dividing the data values by the weight.

        Zero-valued weights are ignored.

        Returns
        -------
        None
        """
        inverse_weight = self.weight.data.copy()
        nzi = np.nonzero(inverse_weight)
        inverse_weight[nzi] = 1.0 / inverse_weight[nzi]
        super().scale(inverse_weight)

    def get_chi2(self, robust=True):
        """
        Return the Chi-squared statistic.

        Parameters
        ----------
        robust : bool, optional
           If `True`, use the 'robust' (median) method.

        Returns
        -------
        float
        """
        significance = self.significance_values()[self.valid]
        if significance.size == 0:
            return np.nan

        variance = significance ** 2
        if robust:
            return numba_functions.smart_median(
                variance, max_dependence=1)[0] / 0.454937
        else:
            return np.nanmean(variance)

    def mean(self, weights=None):
        """
        Return the weighted mean.

        Parameters
        ----------
        weights : numpy.ndarray (float), optional
            An array of weights.

        Returns
        -------
        mean, weight : float, float
        """
        if weights is None and self.weight is not None:
            weights = self.weight.data
        return super().mean(weights=weights)

    def median(self, weights=None):
        """
        Return the weighted median.

        weights : numpy.ndarray (float), optional
            An array of weights.

        Returns
        -------
        median, weight : float, float
        """
        if weights is None and self.weight is not None:
            weights = self.weight.data
        return super().median(weights=weights)

    def reweight(self, robust=True):
        """
        Re-weight the observation

        Parameters
        ----------
        robust : bool, optional
            If `True`, use the 'robust' (median) method to determine the
            chi2 statistic.

        Returns
        -------
        None
        """
        weight_correction = 1.0 / self.get_chi2(robust=robust)
        self.get_weight_image().scale(weight_correction)
        self.noise_rescale *= 1.0 / np.sqrt(weight_correction)

    def unscale_weights(self):
        """
        Undo the weight rescaling.

        Returns
        -------
        None
        """
        self.get_weight_image().scale(self.noise_rescale ** 2)
        self.noise_rescale = 1.0

    def mem_correct_observation(self, model, lg_multiplier):
        """
        Apply a maximum entropy correction given a model.

        Parameters
        ----------
        model : numpy.ndarray or FlaggedArray or None
            The model from which to base MEM correction.  Should be of shape
            (self.shape).
        lg_multiplier : float
            The Lagrange multiplier (lambda) for the MEM correction.

        Returns
        -------
        None
        """
        noise = self.get_noise().data
        if isinstance(model, FlaggedArray):
            model_data = model.data
        else:
            model_data = model
        self.mem_correct(model_data, noise, lg_multiplier)

    def smooth(self, beam_map, reference_index=None, weights=None):
        """
        Smooth the data with a given beam map kernel.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
        reference_index : numpy.ndarray (float), optional
            The reference index (center) of the beam_map kernel.  By default
            this will be set to (beam_map.shape - 1)[::-1] / 2.  Note that the
            reference index should by supplied in (x, y) order for FITS.
        weights : numpy.ndarray (float), optional
            If not supplied, defaults to the observation weights.

        Returns
        -------
        None
        """
        if weights is None:
            weights = self.weight.data

        if reference_index is not None:
            # Reverse this since reference index is passed in as (x, y) order
            # but numpy expects (y, x) order.
            reference_index = reference_index[::-1]

        smoothed, smoothed_weight = self.get_smoothed(
            beam_map, reference_index=reference_index, weights=weights)

        smoothed_exposure, _ = self.get_exposure_image().get_smoothed(
            beam_map, reference_index=reference_index, weights=weights)

        self.set_image(smoothed)
        self.set_weight_image(smoothed_weight)
        self.set_exposure_image(smoothed_exposure)
        self.add_smoothing(
            Gaussian2D.get_equivalent(beam_map, self.grid.resolution))

    def fast_smooth(self, beam_map, steps, reference_index=None, weights=None):
        """
        Smooth the data with a given beam map kernel using fast method.

        Notes
        -----
        This isn't fast compared to standard smooth.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
        steps : numpy.ndarray (int)
            The kernel steps in each dimension.
        reference_index : numpy.ndarray (float), optional
            The reference index (center) of the beam_map kernel.  By default
            this will be set to (beam_map.shape - 1)[::-1] / 2.  Note that the
            reference index should by supplied in (x, y) order for FITS.
        weights : numpy.ndarray (float), optional
            If not supplied, defaults to the observation weights.

        Returns
        -------
        None
        """
        if weights is None:
            weights = self.weight.data

        smoothed, smoothed_weight = self.get_fast_smoothed(
            beam_map, steps, reference_index=reference_index, weights=weights,
            get_weights=True)

        smoothed_exposure = self.get_exposure_image().get_fast_smoothed(
            beam_map, steps, reference_index=reference_index, weights=weights,
            get_weights=False)

        self.set_image(smoothed)
        self.set_weight_image(smoothed_weight)
        self.set_exposure_image(smoothed_exposure)
        self.add_smoothing(
            Gaussian2D.get_equivalent(beam_map, self.grid.resolution))

    def filter_correct(self, underlying_fwhm, reference=None, valid=None):
        """
        Apply filter correction.

        Parameters
        ----------
        underlying_fwhm : astropy.units.Quantity
        reference : FlaggedArray or numpy.ndarray, optional
        valid : numpy.ndarray (bool), optional

        Returns
        -------
        None
        """
        if reference is None:
            reference = self.significance_values()
        super().filter_correct(underlying_fwhm, reference=reference,
                               valid=valid)

    def undo_filter_correct(self, reference=None, valid=None):
        """
        Undo the last filter correction.

        Parameters
        ----------
        reference : FlaggedArray or numpy.ndarray (float), optional
            The data set to determine valid data within the blanking range.
            Defaults to self.data.
        valid : numpy.ndarray (bool), optional
            `True` indicates a data element that may have the filter correction
            factor un-applied.

        Returns
        -------
        None
        """
        if reference is None:
            reference = self.significance_values()
        self.undo_filter_correct(reference=reference, valid=valid)

    def fft_filter_above(self, fwhm, valid=None, weight=None):
        """
        Apply FFT filtering above a given FWHM.

        Parameters
        ----------
        fwhm : astropy.units.Quantity
        valid : numpy.ndarray (bool), optional
        weight : FlaggedArray or numpy.ndarray (float)

        Returns
        -------
        None
        """
        if weight is None:
            weight = self.weight
        super().fft_filter_above(fwhm, valid=valid, weight=weight)

    def resample_from_map(self, map2d, weights=None):
        """
        Resample from one map to another.

        Parameters
        ----------
        map2d : Observation2D
        weights : numpy.ndarray (float), optional

        Returns
        -------
        None
        """
        if weights is None:
            weights = self.weight

        beam = self.get_anti_aliasing_beam_image_for(map2d)
        map_indices = self.get_index_transform_to(map2d)
        self.resample_from(map2d, map_indices, kernel=beam, weights=weights)
        self.copy_processing_from(map2d)

    def get_table_entry(self, name):
        """
        Return a parameter value for a given name.

        Parameters
        ----------
        name : str, optional

        Returns
        -------
        value
        """
        if name == 'depth':
            return self.get_weights().mean()[0] / self.unit.value
        return super().get_table_entry(name)

    def get_hdus(self):
        """
        Return the FITS HDUs for the observation.

        Returns
        -------
        hdus: list (astropy.io.fits.hdu.base.ExtensionHDU)
        """
        hdus = super().get_hdus()
        hdu = self.get_exposures().create_hdu()
        ext_comment = 'Identifier of data contained in this HDU'
        hdu.header['EXTNAME'] = 'Exposure', ext_comment
        self.edit_header(hdu.header)
        hdus.append(hdu)

        hdu = self.get_noise().create_hdu()
        hdu.header['EXTNAME'] = 'Noise', ext_comment
        self.edit_header(hdu.header)
        hdus.append(hdu)

        hdu = self.get_significance().create_hdu()
        hdu.header['EXTNAME'] = 'S/N', ext_comment
        self.edit_header(hdu.header)
        hdus.append(hdu)

        return hdus

    def get_info(self):
        """
        Get a list of info strings for the observation.

        Returns
        -------
        list of str
        """
        info = super().get_info()
        if self.noise_rescale != 1.0:
            info.append(f'Noise Re-scaling: {self.noise_rescale:.2f}x '
                        f'(from image variance).')
        return info

    def index_of_max(self, sign=1, data=None):
        """
        Return the maximum value and index of maximum value.

        Parameters
        ----------
        sign : int or float, optional
            If positive, find the maximum value in the array.  If negative,
            find the minimum value in the array.  If zero, find the maximum
            magnitude in the array.
        data : numpy.ndarray (float), optional
            The data array to examine.  Default is the significance values.

        Returns
        -------
        maximum_value, maximum_index : float, int
        """
        if data is None:
            data = self.significance_values()
        return super().index_of_max(sign=sign, data=data)

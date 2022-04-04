Algorithm
---------

The mapping algorithm for the output source model implements a nearest-pixel method,
whereby each data point is mapped entirely into the map pixel that falls
nearest to the given detector channel :math:`c`, at a given time
:math:`t`. Here,

.. math:: \delta S_{xy} = \frac{\sum\limits_{ct} M_{xy}^{ct} w_c w_t \varkappa_c G_c R_{ct}}{\sum\limits_{ct} M_{xy}^{ct} w_c w_t \varkappa_c^2 G_c^2}

where :math:`M_{xy}^{ct}` associates each sample :math:`\{c,t\}`
uniquely with a map pixel :math:`\{x,y\}`, and is effectively the
transpose of the mapping function defined earlier. :math:`\varkappa_c`
is the point-source filtering (pass) fraction of the pipeline. It can be
thought of as a single scalar version of the transfer function. Its
purpose is to measure how isolated point-source peaks respond to the
various reduction steps, and correct for it. When done correctly, point
source peaks will always stay perfectly cross-calibrated between
different reductions, regardless of what reduction steps were used in
each case. More generally, a reasonable quality of cross-calibration (to
within 10%) extends to compact and slightly extended sources (typically
up to about half of the field-of-view (FoV) in size). While corrections
for more extended structures (:math:`\geq` FoV) are possible to a
certain degree, they come at the price of steeply increasing noise at
the larger scales.

The map-making algorithm should skip over any data that is unsuitable
for quality map-making (such as too-fast scanning that may smear a
source). For formal treatment, we assume that :math:`M_{ct}^{xy} = 0`
for any troublesome data.

Calculating the precise dependence of each map point :math:`S_{xy}` on
the timestream data :math:`R_{ct}` is computationally costly to the
extreme. Instead, the pipeline gets by with the approximation:

.. math:: p_{ct} \approx N_{xy} \cdot \frac{w_t}{\sum\limits_t w_t} \cdot \frac{w_c \varkappa_c^2 G_c}{\sum\limits_c w_c \varkappa_c^2 G_c^2}

This approximation is good as long as most map points are covered with
a representative collection of pixels, and as long as the pixel
sensitivities are more or less uniformly distributed over the field of
view.

We can also calculate the flux uncertainty in the map
:math:`\sigma_{xy}` at each point :math:`\{x,y\}` as:

.. math:: \sigma_{xy}^2 = 1 / \sum_{ct} M_{xy}^{ct} w_c w_t \varkappa_c^2 G_c^2

Source models are first derived from each input scan separately. These
may be despiked and filtered, if necessary, before added to the global
increment with an appropriate noise weight (based on the observed map
noise) if source weighting is desired.

Once the global increment is complete, we can add it to the prior source
model :math:`S_{xy}^{r(0)}` and subject it to further conditioning,
especially in the intermediate iterations. Conditioning operations may
include smoothing, spatial filtering, redundancy flagging, noise or
exposure clipping, signal-to-noise blanking, or explicit source masking.
Once the model is processed into a finalized :math:`S_{xy}'`, we
synchronize the incremental change
:math:`\delta S_{xy}' = S_{xy}' - S_{xy}^{r(0)}` to the residuals:

.. math:: R_{ct} \rightarrow R_{ct} - M_{ct}^{xy} (\delta G_c S_{xy}^{r(0)} + G_c \delta S_{xy}')

Note, again, that :math:`\delta S_{xy}' \neq \delta S_{xy}`. That is,
the incremental change in the conditioned source model is not the same
as the raw increment derived above. Also, since the source gains
:math:`G_c` may have changed since the last source model update, we must
also re-synchronize the prior source model :math:`S_{xy}^{(0)}` with the
incremental source gain changes :math:`\delta G_c` (first term inside
the brackets).

The pipeline operates under the assumption that the point-source
gains :math:`G_c` of the detectors are closely related to the observed
sky-noise gains :math:`g_c` derived from the correlated noise for all
channels. Specifically, it treats the point-source gains as the
product:

.. math:: G_c = \varepsilon_c g_c g_s e^{-\tau}

where :math:`\varepsilon_c` is the point-source coupling efficiency. It
measures the ratio of point-source gains to sky-noise gains (or extended
source gains). Generally, the pipeline will assume :math:`\varepsilon_c = 1`,
unless these values are measured and loaded during the initial scan validation
sequence.

Optionally, the pipeline can also derive :math:`\varepsilon_c` from
the observed response to a source structure, provided the scan pattern
is sufficient to move significant source flux over all detectors. The
source gains also include a correction for atmospheric attenuation, for
an optical depth :math:`\tau`, in-band and in the line of sight.


Point-Source Flux Corrections
-----------------------------

We mentioned point-source corrections in the section above; here, we
explain how these are calculated. First, consider drift removal. Its
effect on point source fluxes is a reduction by a factor:

.. math:: \varkappa_{D,c} \approx 1 - \frac{\tau_{pnt}}{T}

In terms of the 1/f drift removal time constant :math:`T` and the
typical point-source crossing time :math:`\tau_{pnt}`. Clearly, the
effect of 1/f drift removal is smaller the faster one scans across the
source, and becomes negligible when :math:`\tau_{pnt} \ll T`.

The effect of correlated-noise removal, over some group of channels of
mode :math:`i`, is a little more complex. It is calculated as:

.. math:: \varkappa_{(i),c} = 1 - \frac{1}{N_{(i),t}} (P_{(i),c} + \sum_k \Omega_{ck} P_{(i),k})

where :math:`\Omega_{ck}` is the overlap between channels :math:`c` and
:math:`k`. That is, :math:`\Omega_{ck}` is the fraction of the point
source peak measured by channel :math:`c` when the source is centered on
channel :math:`k`. :math:`N_{(i),t}` is the number of correlated
noise-samples that have been derived for the given mode (usually the
same as the number of time samples in the analysis). The correlated
model’s dependence on channel :math:`c` is:

.. math:: P_{(i),c} = \sum_t p_{(i),ct}

Finally, the point-source filter correction due to spectral filtering is
calculated based on the average point-source spectrum produced by the
scanning. Gaussian source profiles with spatial spread
:math:`\sigma_x \approx FWHM / 2.35` produce a typical temporal spread
:math:`\sigma_t \approx \sigma_x / \bar{v}`, in terms of the mean
scanning speed :math:`\bar{v}`. In frequency space, this translates to a
Gaussian frequency spread of :math:`\sigma_f = (2 \pi \sigma_t)^{-1}`,
and thus a point-source frequency profile of:

.. math:: \Psi_f \approx e^{-f^2 / (2\sigma_f^2)}

More generally, :math:`\Psi_f` may be complex-valued (asymmetric beam).
Accordingly, the point-source filter correction due to filtering with
:math:`\phi_f` is generally:

.. math:: \varkappa_{\phi,c} \approx \frac{\sum\limits_f Re(\phi_f \Psi_f \phi_f)}{\sum\limits_f Re(\Psi_f)}

The compound point source filtering effect from :math:`m` model
components is the product of the individual model corrections, i.e.:

.. math:: \varkappa_c = \prod_m \varkappa_{(m),c}


Other Resources
---------------

The scan map reconstruction algorithms are based on a Java pipeline
called CRUSH.  For more information, see:

-  CRUSH paper: `Kovács, A. 2008, Proc. SPIE, 7020,
   45 <http://adsabs.harvard.edu/abs/2008SPIE.7020E..45K>`__

-  CRUSH thesis: `Kovács, A. 2006, PhD Thesis,
   Caltech <http://adsabs.harvard.edu/abs/2006PhDT........28K>`__

-  Online documentation: http://www.sigmyne.com/crush/

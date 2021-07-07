.. currentmodule:: sofia_redux.toolkit.resampling

.. _theory_adaptive:

Theory - Adaptive Weighting
===========================

Inputs
------

Note that all units of distance (:math:`x, v, \alpha`) are in units of the
window :math:`\omega=1`.

Distance Weight
^^^^^^^^^^^^^^^

    .. math::
        w(\delta)_{i,j} = exp \left\{
            - \sum_{k=1}^{M}{
                \frac{\left(x_{i,k} - v_{j,k} \right)^{2}}
                     {\alpha_{k}}
            }
        \right\}

Error Weight
^^^^^^^^^^^^

    .. math::
        w(\epsilon)_{i,j} = \frac{1}{\sigma_{i}}

    if `error_weighting` = `True`, otherwise:

    .. math::
        w(\epsilon)_{i,j} = 1

Fit Weight
^^^^^^^^^^
When fitting point :math:`j`, the weight applied to sample :math:`i` is:

    .. math::
        w_{i,j} = w(\epsilon)_{i,j} w(\delta)_{i,j}


Point Error
^^^^^^^^^^^
The final error of point :math:`j` is:

    .. math::
        e_{j} = \left[ \frac{1}{N_{j}}
            \sum_{i=1}^{N_{j}}{w_{i,j}^{-2}}
        \right]^{-0.5}

This value may be retrieved from the resampler.


Point Distance Weight
^^^^^^^^^^^^^^^^^^^^^
The final distance weight of point :math:`j` is:

    .. math::
        w(\delta)_{j}^{2} =  \sum_{i=1}^{N_{j}}
            {
                {w(\delta)_{i,j}^{2}}
            }


This value may be retrieved from the resampler.


Point Fit Deviation
^^^^^^^^^^^^^^^^^^^
The final fit deviation is given by:

    .. math::
        E^{2}(f_{j}) = \frac
            {\sum_{i=1}^{N_{j}}{
                \left[
                    w_{i,j} \left( d_{i} - f_{j}(x_{i}) \right)
                \right]^{2}}
            }
            {\sum_{i=1}^{N_{j}}{{w_{i,j}}^2}}

This value may be retrieved from the resampler.


Goodness of Fit
^^^^^^^^^^^^^^^
The following parameter is used to describe how well a polynomial fits the
data in the window of point :math:`j`:

    .. math::
        \chi_{j}^{2} = \frac{E^{2}(f_{j})}{N_{j} e_{j}^2}


Local Density
-------------

Volume of an N-Ellipsoid
^^^^^^^^^^^^^^^^^^^^^^^^

The volme of an N-dimensional ellipsoid is given by:

    .. math::
        V(\omega) = \frac
            {\pi^{m/2} \prod_{k=1}^{m}{\omega_{k}}}
            {\Gamma \left( 1 + \frac{m}{2} \right)}

N-dimensional Ellipsoid Gaussian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. math::
        k(\sigma) = A * exp \left\{
           -\sum_{k=1}^{m}{
               \frac{(x_{k} - v_{k})^2}
                    {2 \sigma_{k}^2}
               }
        \right\}

    where,

    .. math::
        A = \frac{1}{(2\pi)^{m/2}\prod_{k=1}^{m}{\sigma_{k}}}

Density Profile
^^^^^^^^^^^^^^^
For point :math:`j`, the Gaussian density profile is:

    .. math::
        \rho(\sigma)_{j} = 2^{m/2} A.{w(\delta)_{j}}^{2}

The density profile over the entire window is:

    .. math::
        \rho(\omega)_{j} = \frac{N_{j}}{V(\omega_{j})}

Define a relative local density profile:

    .. math::
        :nowrap:

        \begin{eqnarray}
        \rho_{j} & = \frac{\rho(\sigma)_{j}}{\rho(\omega)_{j}} \\
        \rho_{j} & = \frac{1}{\Gamma \left( 1 + \frac{m}{2} \right)}.
                     \frac{\prod_{k=1}^{m}{\omega_{j,k}}}
                          {\prod_{k=1}^{m}{\sigma_{j,k}}}.
                     \frac{{w(\delta)_{j}}^{2}}{N_{j}} \\
        \end{eqnarray}


Bandwidth Selection
-------------------
M. Francisco-Fernndez et. al. "A Plug-in Bandwidth Selector for Local
Polynomial Regression Estimator with Correlated Errors"
(`pdf <https://pdfs.semanticscholar.org/3299/0a1103da09de3fba63f8c67759f803068495.pdf/>`_)
has lots of useful stuff, but doesn't work in this case (just here for notes).
Also see notes on `Gaussians <https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/chap1-high-dim-space.pdf>`_.

ADAPTIVE SMOOTHING PARAMETER IN KERNEL DENSITY ESTIMATION
https://www.researchgate.net/publication/336060248

GAUSSIAN SAMPLING
http://www.cs.utah.edu/~gk/MS/html/node29.html

N-BALLS
https://en.wikipedia.org/wiki/Volume_of_an_n-ball

Polynomials
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.8765&rep=rep1&type=pdf

Chi2
https://arxiv.org/pdf/1012.3754.pdf

Indicator function
https://en.wikipedia.org/wiki/Indicator_function

Goodness of fit statistic
https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

Fitting Error
https://iopscience.iop.org/article/10.1088/0026-1394/48/1/002/pdf

Biased and unbiased variance propagation (using biased currently)
https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic

Error propagation
https://pdfs.semanticscholar.org/1928/c3d8bd726b17e71767f28da3920373663134.pdf
https://stats.stackexchange.com/questions/283764/standard-errors-with-weighted-least-squares-regression
https://stats.stackexchange.com/questions/68151/how-to-derive-variance-covariance-matrix-of-coefficients-in-linear-regression
https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=7922&context=etd Local Polynomial Kernel Smoothing with
Correlated Errors
https://stats.stackexchange.com/questions/52704/covariance-of-linear-regression-coefficients-in-weighted-least-squares-method

Kurtosis
https://arxiv.org/pdf/1304.6564v2.pdf


As :math:`N_{j} \rightarrow \infty`, the :math:`\chi^{2}` distribution
asymptotically approaches a normal distribution.  Since this is the basis for
bandwidth selection, it is important to set a sufficiently large window (in
combination with the :math:`fwhm`) when trying to determine an optimal kernel.

The final bandwidth :math:`h` may then be selected according to:

    .. math::
        h_{j} = \chi_{j}^{-\frac{m}{a}}

Likewise, the new weighting parameter :math:`\beta` is given as:

    .. math::
        \beta_{j} = \left( \chi_{j} \rho_{j} \right)^{-\frac{m}{a}}

The optimal value of :math:`\alpha` is estimated using a kernel density
estimator (:math:`\rho`) and the mean integrated square error
(:math:`\chi^{2}`).


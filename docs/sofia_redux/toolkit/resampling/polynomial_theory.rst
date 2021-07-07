.. currentmodule:: sofia_redux.toolkit.resampling

.. _polynomial_theory:

Polynomial Representation of K-dimensional Data
===============================================

The resampler allows polynomial order to vary by dimension, represented by
:math:`o_{k}` for dimension :math:`k` in :math:`K` dimensions.  Let the
polynomial approximation at coordinate :math:`x` be represented as

    .. math::
        f(x) = \sum_{p_{1}=0}^{o_{1}} \sum_{p_{2}=0}^{o_{2}}
               ... \sum_{p_{K}=0}^{o_{K}}
            \lambda_{(p_{1},p_{2},...,p_{K})} a_{(p_{1},p_{2},...,p_{K})}
            x_{1}^{p_{1}} x_{2}^{p_{2}} ... x_{K}^{p_{K}}

where :math:`\lambda_{(p_{1},p_{2},...,p_{K})} \rightarrow \{0,1\}` defining
redundancy where

    .. math::
        :nowrap:

        \begin{eqnarray}
        \lambda_{(p_{1},p_{2},...,p_{K})} = \Biggl \lbrace{
            1, \text{ if } \sum_{k=1}^{K}{p_k} \leq max(o) \atop
            0, \text{ otherwise }
        }
        \end{eqnarray}

Now define a the redundancy set :math:`s` so that :math:`f(x)` may be
represented by the sum over :math:`S` sets when :math:`\lambda=1`.

    .. math::
        :nowrap:

        \begin{eqnarray}
        s     &=& \{ (p_{1}, p_{2},...,p_{K})
                 \; | \; \lambda_{(p_{1}, p_{2},...,p_{K})} = 1 \} \\
        c_{m} &=& a_{s_{m}} \\
        \Phi_{m} &=& \prod_{k=1}^{K}{x_{k}^{s_{(m, k)}}}\\
        f(x)  &=& \sum_{m=1}^{S}{c_{m} \Phi_{m}}
        \end{eqnarray}

The set :math:`s` is in lexographic order such that :math:`{\{0, 1, 1\}}` is
before :math:`{\{1, 0, 0\}}` (for :math:`K=3`).  For example, when :math:`K=2`
and the polynomial order is 2 in all dimensions:

    .. math::
        :nowrap:

        \begin{eqnarray}
        & s_{(p=2,K=2)} & = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)] \\
        & \Phi & = [1, x_{2}, {x_{2}}^{2}, x_{1}, x_{1}x_{2}, {x_{1}}^{2} ] \\
        \therefore & f(x) & = c_{1} + c_{2}x_{2} + c_{3}{x_{2}}^2 + c_{4}x_{1}
                              + c_{5}x_{1}x_{2} + c_{6}{x_{1}}^{2}
        \end{eqnarray}

Likewise, for 3-dimensional (:math:`K=3`) data where :math:`p_{1}=1`,
:math:`p_{2}=2`, and :math:`p_{3}=3`:

    .. math::
        :nowrap:

        \begin{eqnarray}
        s_{(p=[1,2,3], K=3)} = [
        &(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
         (0, 1, 0), (0, 1, 1), (0, 1, 2), \\
        &(0, 2, 0), (0, 2, 1), (1, 0, 1), (1, 0, 2),
         (1, 1, 0), (1, 1, 1), (1, 2, 0) \;]
        \end{eqnarray}

    .. math::
        :nowrap:

        \begin{eqnarray}
        \therefore f(x) =
            &c_{1} + c_{2}x_{3} + c_{3}{x_{3}}^{2} + c_{4}{x_{3}}^{3} +
             c_{5}x_{2} + c_{6}x_{2}x_{3} + &c_{7}x_{2}{x_{3}}^2 + \\
            &c_{8}{x_{2}}^{2} + c_{9}{x_{2}}^{2}x_{3} + c_{10}x_{1}x_{3} +
             c_{11}x_{1}{x_{3}}^2 + c_{12}x_{1}x_{2} + &c_{13}x_{1}x_{2}x_{3} +
             c_{14}x_{1}{x_{2}}^2
        \end{eqnarray}

Polynomial Resampling
=====================
When resampling, we wish to evaluate :math:`f(x)` at coordinate :math:`v`.
To derive :math:`f(x)`, consider a set of :math:`N` samples contained within an
ellipsoid hyper-surface (the window region set :math:`\Omega`) centered about
:math:`v`.

The preliminary step is to first define :math:`s(o)` which for each sample
:math:`i`, allows us to define the set :math:`\Phi_{i}`.  This is then used
to build the design matrix :math:`\Phi` of :math:`N` rows by :math:`S` columns.
:math:`c` may then be found by finding the least-squares solution of

    .. math::
        \Phi c + \varepsilon = y

such that :math:`y_{i}` is the value of sample :math:`i`, with an associated
error of :math:`\varepsilon_{i}` or,

    .. math::

        \begin{bmatrix}
                1 & \Phi_{(1,2)} & \Phi_{(1,3)} & \cdots & \Phi_{(1,S)} \\
                1 & \Phi_{(2,2)} & \Phi_{(2,3)} & \cdots & \Phi_{(2,S)} \\
                1 & \Phi_{(3,2)} & \Phi_{(3,3)} & \cdots & \Phi_{(3,S)} \\
                \vdots & \vdots & \vdots & \ddots & \vdots \\
                1 & \Phi_{(N,2)} & \Phi_{(N,3)} & \cdots & \Phi_{(N,S)}
        \end{bmatrix}
        \begin{bmatrix}
                c_{1} \\
                c_{2} \\
                \vdots \\
                c_{S}
        \end{bmatrix}
        +
        \begin{bmatrix}
                \varepsilon_{1} \\
                \varepsilon_{2} \\
                \varepsilon_{3} \\
                \vdots \\
                \varepsilon_{N}
        \end{bmatrix}
        =
        \begin{bmatrix}
                y_{1} \\
                y_{2} \\
                y_{3} \\
                \vdots \\
                y_{N}
        \end{bmatrix}

Where :math:`\Phi_{(i, m)}` is the :math:`m^{th}` element of set :math:`\Phi`
for sample :math:`i`.

Using least-squares, the estimated values for :math:`c` is given by:

    .. math::
        \hat{c} = (\Phi^{T} \Phi)^{-1} \Phi^{T} y

If performing a weighted fit, :math:`\hat{c}` is given by:

    .. math::
        \hat{c} = (\Phi^{T}W \Phi)^{-1} \Phi^{T} W y


where :math:`W` is an :math:`(N, N)` diagonal weight matrix such that
:math:`W_{i,i}` contains the assigned weighting for sample :math:`i`.

Once :math:`\hat{c}` is known, we can then evaluate :math:`f(v)` as:

    .. math::
        :nowrap:

        \begin{eqnarray}
        &\Phi_{m}^{v} & = \prod_{k=1}^{K}{v_{k}^{s_{(m, k)}}} \\
        &f(v) & = \sum_{m=1}^{S}{\hat{c}_{m} \Phi_{m}^{v}}
        \end{eqnarray}

Implementation
==============

Terminology
-----------
In following sections, the samples at coordinates :math:`x` will be referred
to as *samples*.  The points at which we wish to derive :math:`f(v)` will be
referred to as *points*.  For any set :math:`A`, :math:`|A|` represents the
number of members within :math:`A` which by definition will all be unique.

Unit Conversion and the Window Parameter
----------------------------------------

The window parameter :math:`\omega` defines the semi-principle axes of a
hyper-ellipsoid such that samples :math:`\Omega_{j}` are said to be within
the window region of point :math:`j` if:

    .. math::
        \sum_{k=1}^{K}{\frac{(x_{k}^{\prime} - v_{k, j}^{\prime})^{2}}
                            {\omega_{k}^{2}}} \leq 1

Where :math:`x^{\prime}` and :math:`v^{\prime}` are coordinates before any
unit conversion has been applied.  :math:`\omega` should be supplied in the
same units as :math:`x^{\prime}` and :math:`v^{\prime}` for each dimension.

Coordinates are then converted into units of :math:`\omega` using:

    .. math::
        :nowrap:

        \begin{eqnarray}
        x &= \{
             \frac{x_{1}^{\prime} - min(x_{1}^{\prime})}{\omega_{1}},
             \frac{x_{2}^{\prime} - min(x_{2}^{\prime})}{\omega_{2}}, ...,
             \frac{x_{K}^{\prime} - min(x_{K}^{\prime})}{\omega_{K}},
             \} \\
        v &= \{
             \frac{v_{1}^{\prime} - min(x_{1}^{\prime})}{\omega_{1}},
             \frac{v_{2}^{\prime} - min(x_{2}^{\prime})}{\omega_{2}}, ...,
             \frac{v_{K}^{\prime} - min(x_{K}^{\prime})}{\omega_{K}},
             \}
        \end{eqnarray}

We can then define the set of samples within the window region of point
:math:`j` as

    .. math::
        :nowrap:

        \begin{eqnarray}
        & \Omega_{j} = & \{
            i \; | \; \left\| x_{i} - v_{j} \right\|_2 \leq 1 \; | \;
            i \in \mathbb{Z^{+}} \leq N
        \} \\
        & N_{j} = & | \Omega_{j} |
        \end{eqnarray}

At this point, :math:`\Phi` and :math:`\Phi^{v}` are also calculated for all
samples and points.  This is a fast operation, but does change the memory
requirements from :math:`O(K)` to :math:`O(S)` so that in most cases, memory
requirements are increased.  Generally though, one expects low order
(:math:`p \leq 3`) polynomials to be used such that calculating :math:`\Phi` at
this stage does not introduce significant memory issues.  The alternative would
be to calculate :math:`\Phi` later, for all samples within the window region of
a resampling point leading to many duplicate calculations in situations where
the same sample is within the window of multiple points.

The Search Problem
^^^^^^^^^^^^^^^^^^
Finding each :math:`\Omega_{j}` for all :math:`M` samples using the standard
brute force method of looping through all :math:`N` samples comes at the heavy
computational cost of :math:`O(MN)`.

This problem is overcome in two stages.  The first stage breaks up the full
extent of sample and point space into blocks (hypercubes in :math:`K`
dimensions) where the length of each side of a block is 1 following unit
conversion.

The search problem may then be constrained by recognizing that for each point
within a block, one should only consider membership of samples to that point's
window region from within that same block and neighboring blocks (direct and
diagonal).  Block membership :math:`\square` for sample coordinates :math:`x`
or point coordinates :math:`v` is simply defined by:

    .. math::
        :nowrap:

        \begin{eqnarray}
        & \square(x) = & \lfloor x \rfloor \\
        & \square(v) = & \lfloor v \rfloor
        \end{eqnarray}

The block itself and all neighboring blocks are referred to as a neighborhood
such that for block :math:`l`, the neighborhood is defined as:

    .. math::
        \square_{b}^{hood} = \square_{b} + (-1, 0, 1)^K

where :math:`(-1, 0, 1)^K` indicates all permutations of (-1, 0, 1) over
:math:`K` dimensions.  The algorithm creates two sparse matrices for samples
and points where each row :math:`b` is the block index, and sample or point
indices are stored in the columns.  This allows fast access to all samples
and points contained within each block.  A block will be removed from any
further processing if the point block population
:math:`| \square_{b}(v) | = 0` or the sample neighborhood population
:math:`| \square_{b}^{hood}(x) | = 0`.

Once all valid blocks have been identified, the algorithm may proceed to
process these blocks in parallel.  Having narrowed down the number of samples
to search through for each point in a block to a local neighborhood, euclidean
distances from each point within the block to every sample within the local
neighborhood must be derived in order to find :math:`\Omega`.  There are
multiple ways this could be accomplished, but for this implementation, a
:math:`K` dimensional ball tree is used to directly derive all
:math:`\Omega_{j \in \square_{b}(v)}` efficiently.

Block Processing
----------------
For a single block :math:`b`, a value for :math:`f(v_{j})` is derived in series
for each :math:`j \in \square_{b}(v)`.  However, in order to calculate
:math:`f(v_{j})` we must first ensure that :math:`\hat{c}_{j}` is solvable
via least-squares based on the sample distribution of :math:`\Omega_{j}`.
Let us define the vector :math:`x(j)` such that

    .. math::
        x(j) = x_{i} \; | \; i \in \Omega_{j}

and :math:`x(j)_{k}` are the coordinates of vector :math:`x(j)` along the
:math:`k^{th}` dimension.

Sample Distribution
^^^^^^^^^^^^^^^^^^^
There are three methods available to check whether an attempt to solve
:math:`\hat{c}_{j}` should be made, based on sample distribution.  The most
basic method ("counts") is to require that

    .. math::
        | \Omega_{j} | > \prod_{k=1}^{K}{(o_{k} + 1)}

That is, the number of samples within the window region is greater than that
required to derive a solution assuming zero redundancy
(:math:`\lambda_{(p_{1},p_{2},...,p_{K})} \equiv 1`).  This can be dangerous as
we are only placing a limit on the number of samples, not how they are
distributed.  For example, if the samples are co-linear in any dimension,
a solution is not possible.  Therefore, while fast, "counts" should only be
used when it is assured that samples are uniformly distributed (e.g. an image)
and :math:`\omega` is sufficiently large.

The second method, "extrapolate" is robust, but does not guarantee that
:math:`v_{j}` is bounded by :math:`x(j)`.  As such, it is possible
that any solution may deviate significantly, especially for higher order
polynomials.  The "extrapolate" method requirement is

    .. math::
        | \{ x(j){k} \} | > o_{k} + 1, \forall k = [1, 2, ..., K]

or that in each dimension, there must be enough uniquely valued coordinates
:math:`x` to solve for :math:`\hat{c}` given zero redundancy.  There
may be circumstances in which the user wishes to attain extrapolated values
of a polynomial fit.  For example, when deriving :math:`f(v)` near the edge
of an image.

Finally, "edges" (default) is the most robust of the three methods, similar to
"extrapolate" with the additional requirement that :math:`v_{j}` is bounded by
:math:`x(j)`:

    .. math::
        min(| \{ x(j)_{k} < v_{k} \} |,
            | \{ x(j)_{k} > v_{k} \} |)
        > o_{k} + 1, \; \forall k = [1, 2, ..., K]

or that in each dimension there are more than :math:`o + 1` uniquely valued
sample coordinates to the "left" and "right" of :math:`v`.

If a sample distribution fails the above check, no value will be derived for
:math:`f(v_{j})`.  However, in certain cases when the polynomial order is
symmetrical across all dimensions (:math:`|\{p\}| = 1`), the user may
allow :math:`p` to decrease until the condition is met.  While this is
theoretically possible for asymmetric polynomial orders, doing so would require
recalculating :math:`\Phi` with potentially significant overhead.

Edge Clipping
@@@@@@@@@@@@@

An additional check may be performed on the sample distribution :math:`x(j)`
based on how close :math:`v_{j}` is to the "edge" of :math:`x(j)`.  This edge
is defined by the parameter :math:`\epsilon_{k} \; | \; k = [1, 2, ..., K]`
where :math:`0 < \epsilon_{k} < 1` and one of the three definitions of
an edge.

The "range" definition of the edge requires that point :math:`j` satisfies

    .. math::
        \exists (x(j)_{k} - v_{j, k} \leq -\epsilon_{k}) \land
        \exists (x(j)_{k} - v_{j, k} \geq \epsilon_{k}) \; \forall
        k, k = [1, 2, ..., M]

or that there is at least one sample to both the "left" and "right" of a
resampling point in each dimension at a distance of at least :math:`\epsilon`.

The "com_feature" edge clipping mode requires that

    .. math::
        \frac{1}{N_{j}} \sum_{i=1}^{N_{j}}{
        \frac{| x(j)_{(i,k)} - v_{(j,k)} |}{1 - \epsilon_{k}}} \leq 1
        \; \forall{k}, k = [1, 2, ..., M]

Finally, the "com_distance" edge clipping mode (default) requires that

    .. math::
        \frac{1}{N_{j}} \sum_{i=1}^{N_{j}} {
            \left[ \sum_{k=1}^{K} {
                \left(\frac{ x(j)_{(i,k)} - v_{(j,k)} }
                           { 1 - \epsilon_{k} }\right)^{2} }
            \right]^{0.5}
        } \leq 1

In all cases, as :math:`\epsilon \to 1`, edge clipping effects will become
increasingly severe.

Weighting
---------
The solution to :math:`f(v)` may be derived by placing optional weights on
each sample based on associated measurement error in :math:`y` and/or the
proximity of :math:`x` from :math:`v`.  The weight matrix :math:`W`
is a diagonal matrix :math:`(N \times N)` in which we define the weight for
sample :math:`i` as :math:`w_{i} = W_{(i,i)}` and

    .. math::
        w_{i} = w_{i}^{\varepsilon} w_{i}^{\delta}

For each point :math:`j`, the vector of weights is

    .. math::
        w(j) = w(j)^{\varepsilon} w(j)^{\delta}

where :math:`w^{\delta}` is the distance weighting and :math:`w^{\varepsilon}`
is the error weighting.  If error weighting is disabled then
:math:`w^{\varepsilon} = 1`, and if distance weighting is disabled,
:math:`w^{\delta} = 1`.

Error Weighting
^^^^^^^^^^^^^^^
For error weighting to be applied, :math:`\varepsilon` must be supplied by the
user.

    .. math::
        :nowrap:

        \begin{eqnarray}
        w_{i}^{\varepsilon} & = &  \frac{1}{{\varepsilon_{i}}^{2}} \\
        w^{\varepsilon}(j) & = & w_{i}^{\varepsilon}
            \; | \; i \in \Omega_{j}
        \end{eqnarray}

Distance Weighting
^^^^^^^^^^^^^^^^^^
Distance weights use a Gaussian function of the euclidean distance of samples
from a point.  The user may either supply a smoothing parameter :math:`\alpha`,
or the Gaussian sigma in the original coordinate units :math:`\alpha^{\prime}`.

    .. math::
        \alpha_{k} = \frac{2 {\alpha_{k}^{\prime}}^{2}}{{\omega_k}^{2}},
        k = [1, 2, ..., K]

The vector of distance weights applied to point :math:`j` is

    .. math::
        w^{\delta}(j) = exp \left(
        -\sum_{k=1}^{K}{
            \frac{(x(j)_{k} - v_{(j,k)})^{2}}{\alpha_{k}}
        }
        \right)

In the initial units (marked by :math:`\prime`), this is equivalent to

    .. math::
        w^{\delta}(j) = exp \left(
        -\sum_{k=1}^{K}{
            \frac{(x^{\prime}(j)_{k} - v^{\prime}_{(j,k)})^{2}}
                 {2 {\alpha_{k}^{\prime}}^{2}}
        }
        \right)


Deriving Point Solutions
------------------------

At point :math:`j`, define

    .. math::
        :nowrap:

        \begin{eqnarray}
        X & = & [\Phi_{i} \; \forall \; i \in \Omega_{j}] \\
        W & = & \text{ diagonal matrix } \; | \; diag(W) = w(j) \\
        Y & = & [y_{i} \; \forall \; i \in \Omega_{j}] \\
        Z & = & \text{ diagonal matrix } \; | \; diag(Z) = \varepsilon(j)
        \end{eqnarray}

The estimated polynomial coefficients :math:`\hat{c}` are then solved for
by finding the least-squares solution of

    .. math::
        :nowrap:

        \begin{eqnarray}
        & (X^{T} W X) \hat{c_{j}} = X^{T} W Y \\
        & \hat{c_{j}} = (X^{T} W X)^{-1} X^{T} W Y
        \end{eqnarray}

:math:`f(v_{j})` can then be fitted for by

    .. math::
        f(v_{j}) = \sum_{m=1}^{S}{\hat{c}_{m} \Phi_{j, m}^{v}}


Error Estimates
^^^^^^^^^^^^^^^

If errors are to be propagated through the system, the estimated
covariance-variance matrix :math:`C` is

    .. math::

        C = (X^{T} W X)^{-1} X^{T}W Z^{T} Z W^{T} X (X^{T} W X)^{-1}


If no errors are provided, but an estimate for the error in the fit is
required, the covariance-variance matrix is derived from the residuals
(:math:`r`) on the fit.

    .. math::
        :nowrap:

        \begin{eqnarray}
        r & = & Y - f(v_{j}) \\
        C & = & (X^{T} W X)^{-1} X^{T} W r^{T} r W^{T} X (X^{T} W X)^{-1}
        \end{eqnarray}

The error :math:`\sigma_{j}` is then given by

    .. math::
        \sigma_{j}^{2} = \frac{\sum_{s1=1}^{S} \sum_{s2=1}^{S}
                               \Phi_{j,s1}^{v} C_{s1, s2} \Phi_{j,s2}^{v}}
                              {N_{j} - rank(X^{T} W X)}

Symmetric Order Zero
^^^^^^^^^^^^^^^^^^^^
If :math:`o_{k} = 0 \; \forall k` then the weighted mean is used instead of
a polynomial fit.  In this case

    .. math::
        :nowrap:

        \begin{eqnarray}
        f(v_{j}) & = & \frac{tr(W Y)}{tr(W)} \\
        \sigma_{j}^{2} & = & \frac{tr(W Z Z^{T} W^{T})}{tr(W^{T} W)}
        \end{eqnarray}

Goodness of Fit
^^^^^^^^^^^^^^^
If needed, a measure of how well the fit matched the data is given by the
reduced :math:`\chi^2`.

    .. math::
        :nowrap:

        \begin{eqnarray}
        R & = & Z^{-1} (Y - f(v_{j})) \\
        \chi_{r}^{2} & = & \frac{tr(R^{T} W R)}{tr(W)}
                           \frac{N_{j}}{N_{j} - rank(X^{T} W X)}
        \end{eqnarray}

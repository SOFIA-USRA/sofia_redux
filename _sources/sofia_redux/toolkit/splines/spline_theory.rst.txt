.. currentmodule:: sofia_redux.toolkit.resampling.splines

Definition
==========
A spline function :math:`s(x)` of degree :math:`k > 0` (order :math:`k+1`) is
defined on the finite interval :math:`[a,b]` containing monotonically increasing
knots :math:`\lambda_j` where :math:`j=0,1,\cdots,g+1`, :math:`\lambda_0 = a`,
and :math:`\lambda_{g+1}=b`.  For each knot interval
:math:`[\lambda_j, \lambda_{j+1}]` the spline function is given by a polynomial
of degree :math:`\leq k`, and must be continues along with all its derivatives
up to :math:`k-1` on :math:`[a,b]`.  For further information on the theory
and computational implementation for univariate and bivariate spline fitting,
please see Paul Dierckx, Curve and Surface Fitting with Splines, Oxford
University Press, 1993.

Implementation - Spline representation
======================================
Spline coefficients and knots are derived iteratively starting with only two
knots per dimension by default at :math:`[min(x), max(x)]`.  At each iteration,
a new knot will be added until:

    .. math::
        :nowrap:

        \begin{eqnarray}
        \left( \sum_i^n{(d_i - s(x_i))^2} \right) - \alpha \leq
            \alpha * tolerance
        \end{eqnarray}

where :math:`d_i` is the sample value at point :math:`i` and :math:`\alpha`
is the smoothing factor for the reduction.  When :math:`\alpha = 0`, the
reduction will terminate once the resulting function represents an interpolating
spline.  The :math:`tolerance` parameter is a user defined value, typically of
the order :math:`10^{-3}`.

During each new reduction, the current knots are used to divide the entire
sample space into panels.  Each panel is defined as an n-dimensional volume
with neighbouring knots at each vertex.  Fits are performed to all samples in a
panel, and the panel with the greatest :math:`w = \sum{(d - s(x))^2}` is taken
as the candidate in which to place a new knot.  If no valid knot can be placed
at that location, the algorithm moves onto the next greatest panel :math:`w`
and so on.

Note that in rare circumstances, iterations will halt when the location of the
new knot coincides with an existing one or the maximum number of knots has been
reached.  Iterations may also terminate once the maximum number of iterations
has been reached, the solution no longer converges, or the required storage
space exceeds a certain limit.

Implementation - Spline evaluation
==================================
Once a spline representation has been derived, it may be evaluated at any
given coordinate :math:`x` in the range :math:`a \leq x \leq b`.  B-splines
are generated for :math:`x` in each dimension using the Cox-de Boor recursion
formula (2007):

    .. math::
        :nowrap:

        \begin{eqnarray}
            B_{j,0}(x) & = & \left\{
                \begin{matrix}
                    1 & \text{if} \quad \lambda_j \leq x < \lambda_{j+1} \\
                    0 & \text{otherwise}
                \end{matrix} \right\} \\
            B_{j,k}(x) & = & \frac{x - \lambda_j}
                              {\lambda_{j+k} - \lambda_j}
                         B_{j,k-1}(x) +
                         \frac{\lambda_{j+k+1} - x}
                              {\lambda_{j+k+1} - \lambda_{j+1}}
                         B_{j+1,k-1}(x)
        \end{eqnarray}

If :math:`j` defines the knot interval :math:`\lambda_j \leq x < \lambda_{j+1}`,
a solution at :math:`x` is given by:

    .. math::
        :nowrap:

        \begin{eqnarray}
            s(x) = \sum_{i=j-k}^{j} c_i B_{i,k}(x)
        \end{eqnarray}

where :math:`c` are the spline coefficients which are determined during the
spline representation phase.

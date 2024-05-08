import functools
from typing import Callable, Tuple

import numpy as np


def memoize(func):
    """
    The `memoize` function is a decorator that caches the results of a function based on its input
    arguments to improve performance by avoiding redundant calculations.

    :param func: The `func` parameter in the `memoize` function is a function that you want to apply
    memoization to. It is the function whose return values you want to cache for future use, in order to
    avoid redundant computations
    :return: The `memoize` function is returning the `memoized_func` function, which is a wrapper
    function that caches the results of the original function `func` based on the input arguments `args`
    and `kwargs`.
    """
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func


def get_future_price_mean_var(
    x: float,
    t: float,
    delta_t: float,
    lognormal: bool,  # whether dispersion is multiplied by x or not
    rate_int_func: Callable[[float], float],  # ir integral
    sigma2_int_func: Callable[[float], float],  # sigma^2 integral
) -> Tuple[float, float]:
    """
    :param x: represents underlying price at time t (= x_t)
    :param t: represents current time t
    :param delta_t: represents interval of time beyond t at which
    we want the future price, i.e., at time t + delta_t
    :param lognormal: this indicates whether dispersion func is
    multiplied by x or not (i.e., whether lognormal or normal)
    :param rate_int_func: this is ir(t) func
    :param sigma2_int_func: this is isig(t) func
    :return: mean and variance of x_{t+delta_t} if
    lognormal == True, else return mean and variance of
    log(x_{t+delta_t})

    rate_int_func is ir(t) = int_0^t r(u) du

    If lognormal == True, we have generalized GBM
    dx_t = r(t) x_t dt + sigma(t) x_t dz_t
    The solution is (denoting t + delta_t as t1):
    x_{t1} = x_t . e^{int_t^{t1} (r(u) -
     sigma^2(u)/2) du + int_t^{t1} sigma(u) dz_u}
     So, log(x_{t1}) is normal with:
    Mean[log(x_{t1})] = log(x_t) + int_t^{t1} (r(u) - sigma^2(u)/2) du
    Variance[log(x_{t1}] = int_t^{t1} sigma^2(u) du
    In the case that lognormal == True, sigma2_int_func
    = isig(t) = int_0^t sigma^2(u) du
    Therefore, in the case that lognormal == True,
    log(x_{t1}) is normal with:
    Mean[log(x_{t1})] = log(x_t) + ir(t1) - ir(t) + (isig(t) - isig(t1)) / 2
    Variance[log(x_{t1})] = isig(t1) - isig(t)

    If lognormal == False, we have generalize OU with mean-reversion to 0
    dx_t = r(t) x_t dt + sigma(t) dz_t
    The solution is (denoting t + delta_t as t1)
    x_{t1} = x_t e^{int_t^{t1} r(u) du} +
     (e^{int_0^{t1} r(u) du}) . (int_t^t1 sigma(u) e^{-int_0^u r(s) ds} d_zu)
     So, x_{t1} is normal with:
    Mean[x_{t1}] = x_t . e^{int_t^{t1} r(u) du}
    Variance[x_{t1}] = (e^{int_0^{t1} 2 r(u) du})) .
    (int_t^t1 sigma^2(u) e^{-int_0^u 2 r(s) ds} du)
    In the case that lognormal == False, sigma2_int_func
    = isig(t) = int_0^t sigma^2(u) . e^{-int_0^u 2 r(s) ds} . du
    Therefore, in the case that lognormal == False,
    x_{t1} is normal with:
    Mean[x_{t1}] = x_t . e^{ir(t1) - ir(t)}
    Variance[x_{t1}] = e^{2 ir(t1)} . (isig(t1) - isig(t))
    """
    ir_t = rate_int_func(t)
    ir_t1 = rate_int_func(t + delta_t)
    isig_t = sigma2_int_func(t)
    isig_t1 = sigma2_int_func(t + delta_t)
    ir_diff = ir_t1 - ir_t
    isig_diff = isig_t1 - isig_t

    if lognormal:
        mean = np.log(x) + ir_diff - isig_diff / 2.0
        var = isig_diff
    else:
        mean = x * np.exp(ir_diff)
        var = np.exp(2.0 * ir_t1) * isig_diff
    return mean, var


def lagval(x, c, tensor=True):
    """
    Evaluate a Laguerre series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

        .. versionadded:: 1.7.0

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    lagval2d, laggrid2d, lagval3d, laggrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval
    >>> coef = [1,2,3]
    >>> lagval(1, coef)
    -0.5
    >>> lagval([[1,2],[3,4]], coef)
    array([[-0.5, -4. ],
           [-4.5, -2. ]])

    """
    c = np.array(c, ndmin=1, copy=False)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
    return c0 + c1 * (1 - x)

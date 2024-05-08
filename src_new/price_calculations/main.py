"""
    PRE USAGE SETUP:
        pip install -r requirements.txt

    EXAMPLE USAGE:

        cd src
        python -m price_calculations.main

"""

# Std libraries
from pathlib import Path
from typing import Set

# External libraries
import numpy as np

# Local modules
from price_calculations.bs_pricing import EuropeanBSPricing
from price_calculations.american_pricing import AmericanPricing
from price_calculations.utils import lagval

is_call_val = False
spot_price_val = 80.0
strike_val = 80.0
expiry_val = 1.0
lognormal_val = True
r_val = 0.03
sigma_val = 0.3

##################################################

"""
    CONSTANTS DICTATING PROCESSING THREADS.
"""

num_dt_val = 100

# initial defualt value '10000'
# CURRENT num_paths_val
num_paths_val = 1000

"""
# num_paths_val = 50000
European Price = 8.262
LSPI Price = 8.316
FQI Price = ?
Longstaff-Schwartz Price = ?
"""

"""
# num_paths_val = 10000
European Price = 8.262
LSPI Price = 8.337
FQI Price = 8.361
Longstaff-Schwartz Price = 8.596
"""

"""
# num_paths_val = 1000
European Price = 8.262
LSPI Price = 8.187
FQI Price = 8.143
Longstaff-Schwartz Price = 8.588
"""

"""
# num_paths_val = 100
European Price = 8.262
LSPI Price = 7.173
FQI Price = 7.030
Longstaff-Schwartz Price = 9.460
"""

"""
# num_paths_val = 10
European Price = 8.262
LSPI Price = 2.569
FQI Price = 2.569
Longstaff-Schwartz Price = 8.371
"""

num_laguerre_val = 3
num_iters_val = 15

####################################################

epsilon_val = 1e-3
spot_pct_noise_val = 0.25


if __name__ == "__main__":

    # This is creating an instance of the `EuropeanBSPricing` class with
    # specific parameter values such as `is_call`, `spot_price`, `strike`, `expiry`, `r`, and `sigma`.
    # After creating the instance, it calculates the option price using the `option_price` attribute
    # of the `ebsp` object and then prints the European option price with 3 decimal places using the
    # `print` statement.
    ebsp = EuropeanBSPricing(
        is_call=is_call_val,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
        r=r_val,
        sigma=sigma_val,
    )
    print("European Price = %.3f" % ebsp.option_price)
    del ebsp

    # The `# noinspection PyShadowingNames` comments are used to suppress warnings about shadowing
    # variable names in Python. In this specific context, the lambda functions `ir_func` and
    # `isig_func` are defined using default arguments `r_val=r_val` and `sigma_val=sigma_val`
    # respectively.
    # noinspection PyShadowingNames
    ir_func = lambda t, r_val=r_val: r_val * t
    # noinspection PyShadowingNames
    isig_func = lambda t, sigma_val=sigma_val: sigma_val * sigma_val * t

    def vanilla_american_payoff(_: float, x: np.ndarray) -> float:
        """
        This Python function calculates the payoff for a vanilla American option based on the option
        type (call or put) and the underlying asset price at expiration.

        :param _: The underscore (_) is typically used as a placeholder variable in Python when you want
        to ignore the value. In this function, it seems that the underscore (_) is not being used for
        any specific purpose and can be safely ignored
        :type _: float
        :param x: The parameter `x` is expected to be a numpy array
        representing the possible future prices of the underlying asset
        :type x: np.ndarray
        :return: The function `vanilla_american_payoff` is returning the payoff value of an American
        option based on the input parameters. If it is a call option (`is_call_val` is True), the
        function calculates the payoff as the maximum of the difference between the final stock price
        (`x[-1]`) and the strike price (`strike_val`) or 0.0. If it is a put
        """
        if is_call_val:
            ret = max(x[-1] - strike_val, 0.0)
        else:
            ret = max(strike_val - x[-1], 0.0)
        return ret

    # noinspection PyShadowingNames
    amp = AmericanPricing(
        spot_price=spot_price_val,
        payoff=lambda t, x: vanilla_american_payoff(t, x),
        expiry=expiry_val,
        lognormal=lognormal_val,
        ir=ir_func,
        isig=isig_func,
    )

    # Creating an identity matrix of size
    # `num_laguerre_val` using NumPy.
    ident = np.eye(num_laguerre_val)

    # noinspection PyShadowingNames
    def laguerre_feature_func(x: float, i: int) -> float:
        """
        The function `laguerre_feature_func` calculates a Laguerre feature based on the input `x` and
        index `i`.

        :param x: The parameter `x` is a float value that is being passed to the `laguerre_feature_func`
        function
        :type x: float
        :param i: The parameter `i` in the `laguerre_feature_func` function is an integer representing
        the index used to access a specific element in the `ident` list
        :type i: int
        :return: The function `laguerre_feature_func` is returning the result of the expression
        `np.exp(-xp / 2) * lagval(xp, ident[i])`.
        """
        # noinspection PyTypeChecker
        xp = x / strike_val
        return np.exp(-xp / 2) * lagval(xp, ident[i])

    def rl_feature_func(ind: int, x: float, i: int) -> float:
        """
        The function `rl_feature_func` calculates a feature value based on the input parameters `ind`,
        `x`, and `i` using conditional logic.

        :param ind: The parameter `ind` represents the index of the feature being
        calculated.
        :type ind: int
        :param x: The `x` parameter in the `rl_feature_func` function, represents a value that is
        used in the calculation of the feature function. It is passed as an argument to the function and
        used in the `laguerre_feature_func` call and other calculations within the function.
        :type x: float
        :param i: The parameter `i` in the `rl_feature_func` function represents the index of the
        feature being calculated. It is used to determine which feature to compute based on the
        conditions specified in the function
        :type i: int
        :return: The function `rl_feature_func` returns a float value based on the conditions specified
        in the function. The value returned depends on the input parameters `ind`, `x`, and `i`, as well
        as the variables `expiry_val`, `num_dt_val`, `num_laguerre_val`, and the function
        `laguerre_feature_func`.
        """
        dt = expiry_val / num_dt_val
        t = ind * dt
        if i == 0:
            ret = 1.0
        elif i < num_laguerre_val + 1:
            ret = laguerre_feature_func(x, i - 1)
        elif i == num_laguerre_val + 1:
            ret = np.sin(-t * np.pi / (2.0 * expiry_val) + np.pi / 2.0)
        elif i == num_laguerre_val + 2:
            ret = np.log(expiry_val - t)
        else:
            rat = t / expiry_val
            ret = rat * rat
        return ret

    lspi_price = amp.get_lspi_price(
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        feature_funcs=[
            lambda t, x, i=i: rl_feature_func(t, x[-1], i)
            for i in range(num_laguerre_val + 4)
        ],
        num_iters=num_iters_val,
        epsilon=epsilon_val,
        spot_pct_noise=spot_pct_noise_val,
    )
    print("LSPI Price = %.3f" % lspi_price)

    fqi_price = amp.get_fqi_price(
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        feature_funcs=[
            lambda t, x, i=i: rl_feature_func(t, x[-1], i)
            for i in range(num_laguerre_val + 4)
        ],
        num_iters=num_iters_val,
        epsilon=epsilon_val,
        spot_pct_noise=spot_pct_noise_val,
    )
    print("FQI Price = %.3f" % fqi_price)

    ls_price = amp.get_ls_price(
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        feature_funcs=[lambda _, x: 1.0]
        + [
            (lambda _, x, i=i: laguerre_feature_func(x[-1], i))
            for i in range(num_laguerre_val)
        ],
    )
    print("Longstaff-Schwartz Price = %.3f" % ls_price)

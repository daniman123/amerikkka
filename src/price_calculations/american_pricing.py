from pathlib import Path
from typing import Callable, Sequence, Set, Tuple

import numpy as np

from price_calculations.utils import get_future_price_mean_var, memoize

StateType = Tuple[int, np.ndarray]
ActionType = bool


class AmericanPricing:
    """
        In the risk-neutral measure, the underlying price x_t
        follows the Ito process: dx_t = r(t) x_t dt + dispersion(t, x_t) dz_t
        spot_price is x_0
        In this module, we only allow two types of dispersion functions,
        Type 1 (a.k.a. "lognormal") : dx_t = r(t) x_t dt + sigma(t) x_t dz_t
        Type 2 (a.k.a. "normal"): dx_t = r(t) x_t dt + sigma(t) dz_t
        payoff is a function from (t, x_t) to payoff (eg: x_t - K)
        expiry is the time to expiry of american option (in years)
        lognormal is a bool that defines whether our dispersion function
        amounts to Type 1(lognormal) or Type 2(normal)
        r(t) is a function from (time t) to risk-free rate
        sigma(t) is a function from (time t) to (sigma at time t)
        We don't provide r(t) and sigma(t) as arguments
        Instead we provide their appropriate integrals as arguments
        Specifically, we provide ir(t) and isig(t) as arguments (as follows):
        ir(t) = \int_0^t r(u) du, so discount D_t = e^{- ir(t)}
        isig(t) = \int 0^t sigma^2(u) du if lognormal == True
        else \int_0^t sigma^2(u) e^{-\int_0^u 2 r(s) ds} du
    """

    def __init__(
        self,
        spot_price: float,
        payoff: Callable[[float, np.ndarray], float],
        expiry: float,
        lognormal: bool,
        ir: Callable[[float], float],
        isig: Callable[[float], float]
    ) -> None:
        self.spot_price: float = spot_price
        self.payoff: Callable[[float, np.ndarray], float] = payoff
        self.expiry: float = expiry
        self.lognormal: bool = lognormal
        self.ir: Callable[[float], float] = ir
        self.isig: Callable[[float], float] = isig

    @memoize
    def get_all_paths(
        self,
        spot_pct_noise: float,
        num_paths: int,
        num_dt: int
    ) -> np.ndarray:

        """
            Standard use path defintions - GBM, method of simulating stock prices.
        """
        dt = self.expiry / num_dt
        paths = np.empty([num_paths, num_dt + 1])
        spot = self.spot_price
        for i in range(num_paths):
            start = max(0.001, np.random.normal(spot, spot * spot_pct_noise))
            paths[i, 0] = start
            for t in range(num_dt):
                m, v = get_future_price_mean_var(
                    paths[i, t],
                    t,
                    dt,
                    self.lognormal,
                    self.ir,
                    self.isig
                )
                norm_draw = np.random.normal(m, np.sqrt(v))
                paths[i, t + 1] = np.exp(norm_draw) if self.lognormal else norm_draw
        return paths

    def get_ls_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]]
    ) -> float:
        paths = self.get_all_paths(0.0, num_paths, num_dt)
        cashflow = np.array([max(self.payoff(self.expiry, paths[i, :]), 0.)
                             for i in range(num_paths)])
        dt = self.expiry / num_dt

        stprcs = np.arange(100.)
        final = [(p, self.payoff(self.expiry, np.append(np.zeros(num_dt), p))) for p in stprcs]
        ex_boundary = [max(p for p, e in final if e > 0)]

        for step in range(num_dt - 1, 0, -1):
            """
            For each time slice t
            Step 1: collect X as features of (t, [S_0,.., S_t]) for those paths
            for which payoff(t, [S_0, ...., S_t]) > 0, and corresponding Y as
            the time-t discounted future actual cash flow on those paths.
            Step 2: Do the (X,Y) regression. Denote Y^ as regression-prediction.
            Compare Y^ versus payoff(t, [S_0, ..., S_t]). If payoff is higher,
            set cashflow at time t on that path to be the payoff, else set 
            cashflow at time t on that path to be the time-t discounted future
            actual cash flow on that path.
            """
            t = step * dt
            disc = np.exp(self.ir(t) - self.ir(t + dt))
            cashflow = cashflow * disc
            payoff = np.array([self.payoff(t, paths[i, :(step + 1)]) for
                               i in range(num_paths)])
            indices = [i for i in range(num_paths) if payoff[i] > 0]
            if len(indices) > 0:
                x_vals = np.array([[f(step, paths[i, :(step + 1)]) for f in
                                    feature_funcs] for i in indices])
                y_vals = np.array([cashflow[i] for i in indices])
                weights = np.linalg.lstsq(x_vals, y_vals, rcond=None)[0]
                estimate = x_vals.dot(weights)

                for i, ind in enumerate(indices):
                    if payoff[ind] > estimate[i]:
                        cashflow[ind] = payoff[ind]

                prsqs = [np.append(np.zeros(step), s) for s in stprcs]
                cp = [weights.dot([f(step, prsq) for f in feature_funcs]) for prsq in prsqs]
                ep = [self.payoff(t, prsq) for prsq in prsqs]
                ll = [p for p, c, e in zip(stprcs, cp, ep) if e > c]
                if len(ll) == 0:
                    num = 0.
                else:
                    num = max(ll)
                ex_boundary.append(num)

        return max(
            self.payoff(0., np.array([self.spot_price])),
            np.average(cashflow * np.exp(-self.ir(dt)))
        )


    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_dt: int
    ) -> Tuple[StateType, float]:
        ind, price_arr = state
        delta_t = self.expiry / num_dt
        t = ind * delta_t
        reward = (np.exp(-self.ir(t)) * self.payoff(t, price_arr)) if\
            (action and ind <= num_dt) else 0.
        m, v = get_future_price_mean_var(
            price_arr[-1],
            t,
            delta_t,
            self.lognormal,
            self.ir,
            self.isig
        )
        norm_draw = np.random.normal(m, np.sqrt(v))
        next_price = np.exp(norm_draw) if self.lognormal else norm_draw
        price1 = np.append(price_arr, next_price)
        next_ind = (num_dt if action else ind) + 1
        return (next_ind, price1), reward

    def get_price_from_paths_and_params(
        self,
        paths: np.ndarray,
        params: np.ndarray,
        num_dt: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]]
    ) -> float:
        num_paths = paths.shape[0]
        prices = np.zeros(num_paths)
        dt = self.expiry / num_dt
        for path_num, path in enumerate(paths):
            step = 0
            while step <= num_dt:
                t = dt * step
                price_seq = path[:(step + 1)]
                exercise_price = self.payoff(t, price_seq)
                if step == num_dt:
                    continue_price = 0.
                else:
                    continue_price = params.dot([f(step, price_seq) for f in
                                                 feature_funcs])
                step += 1
                if exercise_price > continue_price:
                    prices[path_num] = np.exp(-self.ir(t)) * exercise_price
                    step = num_dt + 1


        return np.average(prices)

    def get_lspi_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]],
        num_iters: int,
        epsilon: float,
        spot_pct_noise: float
    ) -> float:
        features = len(feature_funcs)
        params = np.zeros(features)
        paths = self.get_all_paths(spot_pct_noise, num_paths, num_dt)
        iter_steps = num_paths * num_dt
        dt = self.expiry / num_dt

        for _ in range(num_iters):
            a_mat = np.zeros((features, features))
            b_vec = np.zeros(features)

            for path_num, path in enumerate(paths):

                for step in range(num_dt):
                    t = step * dt
                    disc = np.exp(self.ir(t) - self.ir(t + dt))
                    phi_s = np.array([f(step, path[:(step + 1)]) for f in
                                      feature_funcs])
                    local_path = path[:(step + 2)]
                    phi_sp = np.zeros(features)
                    reward = 0.
                    next_payoff = self.payoff(t + dt, local_path)

                    if step == num_dt - 1:
                        reward = next_payoff
                    else:
                        next_phi = np.array([f(step + 1, local_path)
                                             for f in feature_funcs])
                        if next_payoff > params.dot(next_phi):
                            reward = next_payoff
                        else:
                            phi_sp = next_phi

                    a_mat += np.outer(
                        phi_s,
                        phi_s - phi_sp * disc
                    )
                    b_vec += reward * disc * phi_s

            a_mat /= iter_steps
            a_mat += epsilon * np.eye(features)
            b_vec /= iter_steps
            params = np.linalg.inv(a_mat).dot(b_vec)
            # print(params)

        return self.get_price_from_paths_and_params(
            self.get_all_paths(0.0, num_paths, num_dt),
            params,
            num_dt,
            feature_funcs
        )

    def get_fqi_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]],
        num_iters: int,
        epsilon: float,
        spot_pct_noise: float
    ) -> float:
        features = len(feature_funcs)
        params = np.zeros(features)
        paths = self.get_all_paths(spot_pct_noise, num_paths, num_dt)
        iter_steps = num_paths * num_dt
        dt = self.expiry / num_dt

        for _ in range(num_iters):
            a_mat = np.zeros((features, features))
            b_vec = np.zeros(features)

            for path_num, path in enumerate(paths):

                for step in range(num_dt):
                    t = step * dt
                    disc = np.exp(self.ir(t) - self.ir(t + dt))
                    phi_s = np.array([f(step, path[:(step + 1)]) for f in
                                      feature_funcs])
                    local_path = path[:(step + 2)]

                    next_payoff = self.payoff(t + dt, local_path)
                    if step == num_dt - 1:
                        max_val = next_payoff
                    else:
                        next_phi = np.array([f(step + 1, local_path)
                                             for f in feature_funcs])
                        max_val = max(next_payoff, params.dot(next_phi))

                    a_mat += np.outer(phi_s, phi_s)
                    b_vec += phi_s * disc * max_val

            a_mat /= iter_steps
            a_mat += epsilon * np.eye(features)
            b_vec /= iter_steps
            params = np.linalg.inv(a_mat).dot(b_vec)
            # print(params)

        return self.get_price_from_paths_and_params(
            self.get_all_paths(0.0, num_paths, num_dt),
            params,
            num_dt,
            feature_funcs
        )

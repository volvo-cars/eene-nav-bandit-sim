# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

import numpy as np


def mc_estimate_of_mean_charge_power_reciprocal(charge_alpha, charge_beta, min_power, max_power,
                                                number_of_samples=1000, scale=300., rng=np.random.default_rng()):
    """ Monte Carlo estimate of the expected reciprocal of the charging power (assumed to be gamma-distributed)

    :param charge_alpha: Alpha parameter of the charging power distribution
    :param charge_beta: Beta parameter of the charging power distribution
    :param min_power: Minimum allowed power
    :param max_power: Maximum allowed power
    :param number_of_samples: Number of Monte Carlo samples
    :param scale: Scale factor for the charging power distribution
    :param rng: NumPy random number generator
    :return: Estimate of the expected reciprocal of the charging power
    """
    gamma_sample = rng.gamma(float(charge_alpha),
                             scale=1. / float(charge_beta),
                             size=number_of_samples)
    charging_power_sample = np.maximum(min_power, max_power - (gamma_sample * scale))
    avg_charging_power_reciprocal = (1. / charging_power_sample).mean()
    return avg_charging_power_reciprocal


class MillerDummyModeSampler:
    """ A dummy sampler implementing the "rvs" method in the same way as the Transformed Density Rejection
    random sampler from SciPy, but returning a fixed value (e.g., the mode) instead.
    """

    def __init__(self, mode):
        """ Constructor of the dummy sampler

        :param mode: Fixed (e.g., mode) value to be return by the "rvs" method.
        """
        self.mode = mode

    def rvs(self, size):
        """ Generate dummy random values in an array of the specified size.

        :param size: Size of the returned array.
        :return: Dummy random values (each element is the specified fixed mode value).
        """
        return self.mode * np.ones(size)

import numpy as np


def mc_estimate_of_mean_charge_power_reciprocal(charge_alpha, charge_beta, min_power, max_power,
                                                number_of_samples=1000, scale=300.):
    gamma_sample = np.random.gamma(float(charge_alpha),
                                   scale=1. / float(charge_beta),
                                   size=number_of_samples)
    charging_power_sample = np.maximum(min_power, max_power - (gamma_sample * scale))
    avg_charging_power_reciprocal = (1. / charging_power_sample).mean()
    return avg_charging_power_reciprocal


class MillerDummyModeSampler:

    def __init__(self, mode):
        self.mode = mode

    def rvs(self, size):
        return self.mode * np.ones(size)

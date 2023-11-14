# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

import numpy as np
import scipy.special as sc
import scipy.optimize as so

EPSILON = np.finfo(float).eps


def gamma_prior_alpha_mode(ln_p, q, r, s, center=1.0):
    """ Compute the mode of the gamma distribution alpha parameter conjugate prior distribution (i.e., the GamCon-II
    distribution) numerically.

    :param ln_p: Distribution parameter "ln_p".
    :param q: Distribution parameter "q".
    :param r: Distribution parameter "r".
    :param s: Distribution parameter "s".
    :param center: Approximate center of the distribution.
    :return: Mode of the distribution.
    """
    fun = lambda alpha: -gamma_prior_alpha_lpdf(alpha, ln_p, q, r, s)
    jac = lambda alpha: -gamma_prior_alpha_dlpdf(alpha, ln_p, q, r, s)
    x0 = np.array(center)
    bounds = [(EPSILON, np.inf)]
    result = so.minimize(fun, x0=x0, bounds=bounds, jac=jac)
    return result.x[0]


def gamma_prior_alpha_pdf(alpha, ln_p, q, r, s):
    """ Probability density function of the gamma distribution alpha parameter conjugate prior distribution (i.e., the
    GamCon-II distribution).

    :param alpha: "Alpha" value.
    :param ln_p: Distribution parameter "ln_p".
    :param q: Distribution parameter "q".
    :param r: Distribution parameter "r".
    :param s: Distribution parameter "s".
    :return: Probability density of distribution at "alpha".
    """
    return np.exp(gamma_prior_alpha_lpdf(alpha, ln_p, q, r, s))


def gamma_prior_alpha_dpdf(alpha, ln_p, q, r, s):
    """ Derivative of probability density function of the gamma distribution alpha parameter conjugate prior
    distribution (i.e., the GamCon-II distribution).

    :param alpha: "Alpha" value.
    :param ln_p: Distribution parameter "ln_p".
    :param q: Distribution parameter "q".
    :param r: Distribution parameter "r".
    :param s: Distribution parameter "s".
    :return: Derivative of the probability density function at "alpha".
    """
    pdf = gamma_prior_alpha_pdf(alpha, ln_p, q, r, s)
    return pdf * gamma_prior_alpha_dlpdf(alpha, ln_p, q, r, s)


def gamma_prior_alpha_lpdf(alpha, ln_p, q, r, s):
    """ Natural logarithm of the probability density function of the gamma distribution alpha parameter conjugate
    prior distribution (i.e., the GamCon-II distribution).

    :param alpha: "Alpha" value.
    :param ln_p: Distribution parameter "ln_p".
    :param q: Distribution parameter "q".
    :param r: Distribution parameter "r".
    :param s: Distribution parameter "s".
    :return: Natural logarithm of the probability density of distribution at "alpha".
    """
    first_term = alpha*ln_p
    second_term = s*alpha*np.log(q)
    third_term = r*sc.gammaln(alpha)
    fourth_term = sc.gammaln(s * alpha)
    return first_term - second_term - third_term + fourth_term


def gamma_prior_alpha_dlpdf(alpha, ln_p, q, r, s):
    """ Derivative of the natural logarithm of the probability density function of the gamma distribution alpha
    parameter conjugate prior distribution (i.e., the GamCon-II distribution).

    :param alpha: "Alpha" value.
    :param ln_p: Distribution parameter "ln_p".
    :param q: Distribution parameter "q".
    :param r: Distribution parameter "r".
    :param s: Distribution parameter "s".
    :return: Derivative of the natural logarithm of the probability density function at "alpha".
    """
    return ln_p - s*np.log(q) - r*sc.digamma(alpha) + s*sc.digamma(s*alpha)


class GammaConjugatePriorAlphaDist:
    """ Gamma distribution alpha parameter conjugate prior distribution (i.e., the GamCon-II distribution).

    The method signatures of this class match the expected signatures of the distribution argument of the SciPy
    Transformed Density Rejection random sampler.
    """

    def __init__(self, ln_p, q, r, s):
        """ Constructor of the distribution.

        :param ln_p: Distribution parameter "ln_p".
        :param q: Distribution parameter "q".
        :param r: Distribution parameter "r".
        :param s: Distribution parameter "s".
        """
        self.ln_p = ln_p
        self.q = q
        self.r = r
        self.s = s

    def pdf(self, x):
        """ Probability density function.

        :param x: Argument.
        :return: Probability density at "x".
        """
        return gamma_prior_alpha_pdf(x, self.ln_p, self.q, self.r, self.s)

    def dpdf(self, x):
        """ Derivative of probability density function.

        :param x: Argument.
        :return: Derivative of probability density function at "x".
        """
        return gamma_prior_alpha_dpdf(x, self.ln_p, self.q, self.r, self.s)

    def support(self):
        """ Support of distribution.

        :return: Support of distribution (as two-element tuple).
        """
        return EPSILON, np.inf

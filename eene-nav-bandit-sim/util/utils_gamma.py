
import numpy as np
import scipy.special as sc
import scipy.optimize as so

EPSILON = np.finfo(float).eps


def gamma_prior_alpha_mode(ln_p, q, r, s, center=1.0):
    fun = lambda alpha: -gamma_prior_alpha_lpdf(alpha, ln_p, q, r, s)
    jac = lambda alpha: -gamma_prior_alpha_dlpdf(alpha, ln_p, q, r, s)
    x0 = np.array(center)
    bounds = [(EPSILON, np.inf)]
    result = so.minimize(fun, x0=x0, bounds=bounds, jac=jac)
    return result.x[0]


def gamma_prior_alpha_lpdf(alpha, ln_p, q, r, s):
    first_term = alpha*ln_p
    second_term = s*alpha*np.log(q)
    third_term = r*sc.gammaln(alpha)
    fourth_term = sc.gammaln(s * alpha)
    return first_term - second_term - third_term + fourth_term


def gamma_prior_alpha_dlpdf(alpha, ln_p, q, r, s):
    return ln_p - s*np.log(q) - r*sc.digamma(alpha) + s*sc.digamma(s*alpha)


def gamma_prior_alpha_pdf(alpha, ln_p, q, r, s):
    return np.exp(gamma_prior_alpha_lpdf(alpha, ln_p, q, r, s))


def gamma_prior_alpha_dpdf(alpha, ln_p, q, r, s):
    pdf = gamma_prior_alpha_pdf(alpha, ln_p, q, r, s)
    return pdf * gamma_prior_alpha_dlpdf(alpha, ln_p, q, r, s)


class GammaConjugatePriorAlphaDist:

    def __init__(self, ln_p, q, r, s):
        self.ln_p = ln_p
        self.q = q
        self.r = r
        self.s = s

    def pdf(self, x):
        return gamma_prior_alpha_pdf(x, self.ln_p, self.q, self.r, self.s)

    def dpdf(self, x):
        return gamma_prior_alpha_dpdf(x, self.ln_p, self.q, self.r, self.s)

    def support(self):
        return EPSILON, np.inf

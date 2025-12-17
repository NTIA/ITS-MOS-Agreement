import os
import yaml

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.stats as stats

from scipy.special import binom


class UniformPDF:
    def __init__(self, a, b):
        """
        UniformPDF

        Wrapper class to for uniform random distribution that ensures better consistency
        with other distributions. Used only to generate plots in corresponding paper.

        Parameters
        ----------
        a : float
            Minimum value of uniform distribution domain of support.
        b : float
            Maximum value of uniform distribution domain of support.
        """
        self.a = a
        self.b = b
        self.scale = self.b - self.a

    def __call__(self, x):
        return self.pdf(x)

    def pdf(self, x):
        return stats.uniform.pdf(x, loc=self.a, scale=self.scale)

    def stat(self, moment):
        return stats.uniform.stats(moments=moment, loc=self.a, scale=self.scale)

    def mean(self):
        return self.stat("m")

    def var(self):
        return self.stat("v")


class TriangularPDF:
    def __init__(self, a, b, m):
        """
        TriangularPDF

        Wrapper class to for triangular random distribution that ensures better
        consistency with other distributions. Used only to generate plots in
        corresponding paper.

        Parameters
        ----------
        a : float
            Minimum value of triangular distribution domain of support.
        b : float
            Maximum value of triangular distribution domain of support.
        m : float
            Value associated with the maximum of the PDF for the triangular
            distribution.
        """
        self.a = a
        self.b = b
        self.m = m

    def __call__(self, x):
        return self.pdf(x)

    def pdf(self, x):
        loc = self.a
        scale = self.b - self.a
        c = (self.m - self.a) / (self.b - self.a)
        return stats.triang.pdf(x, c=c, loc=loc, scale=scale)

    def stat(self, moment):
        loc = self.a
        scale = self.b - self.a
        c = (self.m - self.a) / (self.b - self.a)
        return stats.triang.stats(moments=moment, c=c, loc=loc, scale=scale)

    def mean(self):
        return self.stat("m")

    def var(self):
        return self.stat("v")


class BetaPDF:
    def __init__(self, alpha, beta, a, b):
        """
        BetaPDF

        Wrapper class to for beta random distribution that ensures better consistency
        with other distributions. Used only to generate plots in corresponding paper.

        Parameters
        ----------
        alpha : float
            Alpha parameter of beta distribution.
        beta : float
            Beta parameter of beta distribution.
        a : float
            Shift parameter to shift the distribution. Defines the minimum value of the
            domain of support.
        b : float
            Defines the maximum value of the domain of support. Scale factor is computed
            as (b - a).
        """
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.pdf(x)

    def pdf(self, x):
        scale = self.b - self.a
        beta = stats.beta.pdf(x, a=self.alpha, b=self.beta, loc=self.a, scale=scale)
        return beta

    def stat(self, moment):
        scale = self.b - self.a
        return stats.beta.stats(
            moments=moment, a=self.alpha, b=self.beta, loc=self.a, scale=scale
        )

    def mean(self):
        return self.stat("m")

    def var(self):
        return self.stat("v")


class BinoVotes:
    def __init__(self, v_L, v_H, n_s, n_v, quality_pdf):
        """
        BinoVotes

        Class to for BinoVotes random distribution that ensures consistency with other
        distributions. Used only to generate plots in corresponding paper.

        Parameters
        ----------
        v_L : float
            Minimum value on rating scale.
        v_H : float
            Maximum value on rating scale.
        n_s : int
            Number of values in the rating scale.
        n_v : int
            Number of votes per file.
        quality_pdf : RandomPDF
            One of the random variable PDF classes defined here that describes the
            underlying quality distribution.
        """
        self.v_L = v_L
        self.v_H = v_H
        self.n_s = n_s
        self.n_v = n_v

        self.len = self.n_v * (self.n_s - 1)

        self.quality_pdf = quality_pdf

    def pmf(self, x):
        k = (x - self.v_L) * (self.n_v * (self.n_s - 1)) / (self.v_H - self.v_L)
        if np.abs(k - np.round(k)) > 1e-12:
            raise ValueError(f"x value {x} not in sample space of BinoVotes.")
        else:
            k = np.round(k)
        coef = binom(self.len, k) / (self.v_H - self.v_L) ** (self.n_v * (self.n_s - 1))
        inte, int_err = integrate.quad(self.integral, args=(k), a=self.v_L, b=self.v_H)
        out = coef * inte
        return out

    def integral(self, y, k):
        return (
            (y - self.v_L) ** k
            * (self.v_H - y) ** (self.n_v * (self.n_s - 1) - k)
            * self.quality_pdf(y)
        )

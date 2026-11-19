import os
import yaml

import numpy as np
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


class MaxUnimodalPDF:
    def __init__(self, v_L=1, v_H=5, n_s=5):
        """
        MaxUnimodalPDF

        Wrapper class for a maximum variance unimodal (MVU) discrete random
        distribution. Used only to generate plots in corresponding paper.
        """
        self.v_L = v_L
        self.v_H = v_H
        self.n_s = n_s
        # machine epsilon to avoid floating point issues
        eps = np.finfo(float).eps
        self.step = (self.v_H - self.v_L) / (self.n_s - 1)
        self.ratings = np.arange(self.v_L, self.v_H + self.step - eps, self.step)

    def __call__(self, x):
        return self.pmf(x)

    def pmf(self, x):
        return self.pdf(x)

    def pdf(self, x):
        """
        pmf

        Generate the probability density function for the MVU.

        Only operates on the scale [1, 2, 3, 4, 5].
        For a value x s.t. 1 <= x <=5 the pdf is computed using a linear system to
        satisfy the mean constraint.

        For example if x=2.2 then the pdf has the form P=[a, a, b, b, b] such that
        * sum(P) == 1 (valid probability distribution)
        * sum(P * [1, 2, 3, 4, 5]) == x (correct mean)

        Parameters
        ----------
        x : float
            True quality

        Returns
        -------
        list of float
            Probability density function evaluated at x.
        """
        if np.isnan(x):
            return [np.nan] * self.n_s

        if x < self.v_L or x > self.v_H:
            # If x outside of range pmf is all 0
            return [0] * self.n_s
        if x == self.v_L:
            # If x is at the lower bound, pmf is all 0 except for the first value
            pmf = [1] + [0] * (self.n_s - 1)
            return pmf
        if x == self.v_H:
            # If x is at the upper bound, pmf is all 0 except for the last value
            pmf = [0] * (self.n_s - 1) + [1]
            return pmf
        # Which values have probability a
        a_vals = self.ratings[self.ratings <= x]
        # Which values have probability b
        b_vals = self.ratings[self.ratings > x]
        # Set up for the expected value (sum of a values, sum of b values)
        exp_val = [np.sum(a_vals), np.sum(b_vals)]
        # Set up for the probability sum (number of a values, number of b values)
        prob_sum = [len(a_vals), len(b_vals)]
        # Set up the least squares matrix
        A = np.array([exp_val, prob_sum])
        b = [x, 1]
        # Get the probabilities a and b
        probs = np.linalg.lstsq(
            A,
            b,
        )[0]
        a = probs[0]
        b = probs[1]
        pmf = [a for _ in a_vals] + [b for _ in b_vals]
        return pmf

    def pdf_many(self, x):
        """Return one MVU PMF per quality, with shape ``x.shape + (n_s,)``.

        For each interior quality, ratings at or below it share one probability,
        and ratings above it share another. The two probabilities satisfy the
        normalization and expected-quality constraints.
        """
        quality = np.asarray(x, dtype=float)
        flat_quality = quality.ravel()
        pmfs = np.zeros((flat_quality.size, self.n_s), dtype=float)

        # Match the scalar PMF's handling of NaNs, endpoints, and out-of-range values.
        nan_mask = np.isnan(flat_quality)
        pmfs[nan_mask] = np.nan

        low_mask = flat_quality == self.v_L
        high_mask = flat_quality == self.v_H
        pmfs[low_mask, 0] = 1
        pmfs[high_mask, -1] = 1

        interior_mask = (flat_quality > self.v_L) & (flat_quality < self.v_H)
        if np.any(interior_mask):
            interior_quality = flat_quality[interior_mask]
            # Prefix sums give each quality's lower- and upper-rating groups.
            lower_count = np.searchsorted(
                self.ratings, interior_quality, side="right"
            )
            ratings_cumsum = np.concatenate(([0], np.cumsum(self.ratings)))
            lower_sum = ratings_cumsum[lower_count]
            upper_sum = np.sum(self.ratings) - lower_sum
            upper_count = self.n_s - lower_count
            # Solve normalization and mean constraints for both group probabilities.
            determinant = lower_sum * upper_count - upper_sum * lower_count
            lower_probability = (
                interior_quality * upper_count - upper_sum
            ) / determinant
            upper_probability = (
                lower_sum - lower_count * interior_quality
            ) / determinant
            pmfs[interior_mask] = np.where(
                self.ratings[None, :] <= interior_quality[:, None],
                lower_probability[:, None],
                upper_probability[:, None],
            )

        return pmfs.reshape(quality.shape + (self.n_s,))

    def mean(self, x):
        return np.sum(self.ratings * self.pmf(x))

    def var(self, x):
        mean = self.mean(x)
        pmf = self.pmf(x)
        xx_val = np.sum(self.ratings**2 * pmf)
        var = xx_val - mean**2
        return var

    def var_many(self, x):
        """Return MVU variances with the same shape as the quality input ``x``."""
        pmfs = self.pdf_many(x)
        mean = np.sum(pmfs * self.ratings, axis=-1)
        second_moment = np.sum(pmfs * self.ratings**2, axis=-1)
        return second_moment - mean**2


class MinVariancePDF:
    def __init__(self):
        """
        MinVariancePDF

        Wrapper class for a minimum variance discrete voting model, also called the
        adjacent two-choice (ATC) voting model. Used only to generate plots in
        corresponding paper.
        """
        self.v_L = 1
        self.v_H = 5
        self.n_s = 5
        self.ratings = np.arange(self.v_L, self.v_H + 1)

    def __call__(self, x):
        return self.pmf(x)

    def pdf(self, x):
        return self.pmf(x)

    def pmf(self, x):
        xf = int(np.floor(x) - self.v_L)
        xc = int(np.ceil(x) - self.v_L)
        p = x - np.floor(x)
        pmf = [0] * self.n_s
        pmf[xf] = 1 - p
        pmf[xc] += p
        return pmf

    def mean(self, x):
        return np.sum(self.ratings * self.pmf(x))

    def var(self, x):
        mean = self.mean(x)
        pmf = self.pmf(x)
        xx_val = np.sum(self.ratings**2 * pmf)
        var = xx_val - mean**2
        return var


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

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

import mos_agreement as ma


class TestModelFunctions(unittest.TestCase):
    def test_mos_data_binovotes_average_vote_var(self):
        mos_mean = 3.0
        mos_var = 0.5
        n_v = 10
        s_L = 1
        s_H = 5
        n_s = 5

        # Compute expected according to function's formula
        n_m = n_v * (n_s - 1)
        scale_factor = n_v / (n_m - 1)
        expected = scale_factor * ((mos_mean - s_L) * (s_H - mos_mean) - mos_var)

        got = ma.mos_data_binovotes_average_vote_var(
            mos_mean=mos_mean, mos_var=mos_var, n_v=n_v, s_L=s_L, s_H=s_H, n_s=n_s
        )
        self.assertAlmostEqual(expected, got, places=12)

    def test_mos_data_bounds(self):
        mos_var = 1.0
        average_vote_var = 0.5
        n_v = 4

        expected_quality_var = mos_var - average_vote_var / n_v

        expected_rmse = np.sqrt(average_vote_var / n_v)
        expected_corr = np.sqrt(
            expected_quality_var / (expected_quality_var + average_vote_var / n_v)
        )

        rmse, corr = ma.mos_data_bounds(
            mos_var=mos_var, average_vote_var=average_vote_var, n_v=n_v
        )

        self.assertAlmostEqual(expected_rmse, rmse, places=12)
        self.assertAlmostEqual(expected_corr, corr, places=12)

    def test_quality_distribution_binovotes_bounds(self):
        quality_mean = 3.0
        quality_var = 1.0
        n_v = 4
        s_L = 1
        s_H = 5
        n_s = 5

        numerator = (quality_mean - s_L) * (s_H - quality_mean) - quality_var
        denominator = n_v * (n_s - 1)
        mse = numerator / denominator
        expected_rmse = np.sqrt(mse)
        expected_corr = np.sqrt(quality_var / (quality_var + mse))

        rmse, corr = ma.quality_distribution_binovotes_bounds(
            quality_mean=quality_mean,
            quality_var=quality_var,
            n_v=n_v,
            s_L=s_L,
            s_H=s_H,
            n_s=n_s,
        )

        self.assertAlmostEqual(expected_rmse, rmse, places=12)
        self.assertAlmostEqual(expected_corr, corr, places=12)

    def test_binovotes_and_binomos_with_mocked_rng(self):
        # Prepare deterministic fake RNG so outputs are predictable
        class FakeRNG:
            def binomial(self, n, p, size):
                # return zeros so votes equal s_L after scaling
                return np.zeros(size, dtype=int)

        quality = np.array([3.0])
        n_v = 5

        # Patch the RNG used inside ma.binovotes
        with patch("numpy.random.default_rng", return_value=FakeRNG()):
            votes = ma.binovotes(
                quality=quality, n_v=n_v, step=1, s_L=1, s_H=5, seed=123
            )

        # Expect shape (n_v, quality.size) and all values equal to s_L (1)
        self.assertEqual(votes.shape, (n_v, quality.size))
        self.assertTrue(np.all(votes == 1))

        # Test binomos wrapper returns MOS (mean across axis 0) when mos=True
        with patch("numpy.random.default_rng", return_value=FakeRNG()):
            mos = ma.binomos(
                mos=True, quality=quality, n_v=n_v, step=1, s_L=1, s_H=5, seed=0
            )
        # MOS should be an array with value 1.0
        self.assertEqual(mos.shape, quality.shape)
        self.assertAlmostEqual(float(mos[0]), 1.0, places=12)

        # And when mos=False binomos should return the vote matrix
        with patch("numpy.random.default_rng", return_value=FakeRNG()):
            votes_matrix = ma.binomos(
                mos=False, quality=quality, n_v=n_v, step=1, s_L=1, s_H=5, seed=0
            )
        self.assertEqual(votes_matrix.shape, (n_v, quality.size))


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

import mos_agreement as ma


class TestBoundsModule(unittest.TestCase):
    """Test cases for bounds.py module"""

    def test_mos_data_binovotes_average_vote_var(self):
        """Test calculation of average vote variance under BinoVotes model"""
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
        """Test RMSE and correlation bounds from MOS data statistics"""
        mos_var = 1.0
        vote_var = 0.5  # Can be array or scalar
        n_v = 4

        # Compute SE2 from vote_var and n_v
        vote_se2 = vote_var / n_v
        expected_quality_var = mos_var - vote_se2

        expected_rmse = np.sqrt(vote_se2)
        expected_corr = np.sqrt(
            expected_quality_var / (expected_quality_var + vote_se2)
        )

        rmse, corr = ma.mos_data_bounds(
            mos_var=mos_var, vote_var=vote_var, n_v=n_v
        )

        self.assertAlmostEqual(expected_rmse, rmse, places=12)
        self.assertAlmostEqual(expected_corr, corr, places=12)

    def test_mos_data_bounds_with_array_inputs(self):
        """Test mos_data_bounds with array inputs for vote_var and n_v"""
        mos_var = 1.0
        vote_var = np.array([0.4, 0.5, 0.6])
        n_v = np.array([3, 4, 5])

        # Calculate expected values using the bounds logic
        vote_se2 = vote_var / n_v
        average_vote_se2 = np.mean(vote_se2)
        expected_quality_var = mos_var - average_vote_se2

        expected_rmse = np.sqrt(average_vote_se2)
        expected_corr = np.sqrt(
            expected_quality_var / (expected_quality_var + average_vote_se2)
        )

        rmse, corr = ma.mos_data_bounds(
            mos_var=mos_var, vote_var=vote_var, n_v=n_v
        )

        self.assertAlmostEqual(expected_rmse, rmse, places=12)
        self.assertAlmostEqual(expected_corr, corr, places=12)

    def test_mos_data_binovotes_bounds(self):
        """Test performance bounds from MOS data under BinoVotes model"""
        mos_mean = 3.0
        mos_var = 1.0
        n_v = 10
        s_L = 1
        s_H = 5
        n_s = 5

        rmse, corr = ma.mos_data_binovotes_bounds(
            mos_mean=mos_mean,
            mos_var=mos_var,
            n_v=n_v,
            s_L=s_L,
            s_H=s_H,
            n_s=n_s,
        )

        # Results should be positive
        self.assertGreater(rmse, 0)
        self.assertGreater(corr, 0)
        # Correlation should be between 0 and 1
        self.assertLess(corr, 1)

    def test_quality_distribution_bounds(self):
        """Test performance bounds from quality distribution statistics"""
        quality_var = 1.0
        vote_se2 = 0.25

        rmse, corr = ma.quality_distribution_bounds(
            quality_var=quality_var, vote_se2=vote_se2
        )

        expected_rmse = np.sqrt(vote_se2)
        expected_corr = np.sqrt(quality_var / (quality_var + vote_se2))

        self.assertAlmostEqual(expected_rmse, rmse, places=12)
        self.assertAlmostEqual(expected_corr, corr, places=12)

    def test_quality_distribution_binovotes_bounds(self):
        """Test performance bounds from quality distribution under BinoVotes"""
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


class TestSimVotesModule(unittest.TestCase):
    """Test cases for sim_votes.py module"""

    def test_sim_setup_with_float(self):
        """Test sim_setup with float input"""
        quality = 3.5
        seed = 42
        rng, q = ma.sim_setup(quality, seed)
        
        self.assertIsNotNone(rng)
        self.assertTrue(isinstance(q, np.ndarray))
        self.assertEqual(q.shape, (1,))
        self.assertAlmostEqual(q[0], 3.5)

    def test_sim_setup_with_list(self):
        """Test sim_setup with list input"""
        quality = [1.0, 2.5, 3.5, 4.0]
        seed = 42
        rng, q = ma.sim_setup(quality, seed)
        
        self.assertTrue(isinstance(q, np.ndarray))
        self.assertEqual(q.shape, (4,))
        np.testing.assert_array_almost_equal(q, quality)

    def test_sim_setup_with_array(self):
        """Test sim_setup with array input"""
        quality = np.array([1.0, 2.5, 3.5, 4.0])
        seed = 42
        rng, q = ma.sim_setup(quality, seed)
        
        self.assertTrue(isinstance(q, np.ndarray))
        self.assertEqual(q.shape, (4,))
        np.testing.assert_array_equal(q, quality)

    def test_binovotes_output_shape(self):
        """Test binovotes output shape and range"""
        quality = np.array([1.0, 3.0, 5.0])
        n_v = 10
        s_L = 1
        s_H = 5

        votes = ma.binovotes(quality=quality, n_v=n_v, s_L=s_L, s_H=s_H, seed=42)

        # Check shape: (n_v, len(quality))
        self.assertEqual(votes.shape, (n_v, len(quality)))
        # All votes should be in valid range [s_L, s_H]
        self.assertTrue(np.all(votes >= s_L))
        self.assertTrue(np.all(votes <= s_H))
        # All votes should be integers (multiple of step=1)
        self.assertTrue(np.all(votes == votes.astype(int)))

    def test_binovotes_deterministic_with_seed(self):
        """Test that binovotes is deterministic with same seed"""
        quality = np.array([2.5, 3.5])
        n_v = 5
        
        votes1 = ma.binovotes(quality=quality, n_v=n_v, seed=123)
        votes2 = ma.binovotes(quality=quality, n_v=n_v, seed=123)
        
        np.testing.assert_array_equal(votes1, votes2)

    def test_binomos_wrapper(self):
        """Test binomos wrapper returns MOS when mos=True"""
        quality = np.array([2.0, 3.0, 4.0])
        n_v = 20
        seed = 42

        # Test mos=True (default)
        mos_scores = ma.binomos(mos=True, quality=quality, n_v=n_v, seed=seed)
        self.assertEqual(mos_scores.shape, quality.shape)
        self.assertTrue(np.all(mos_scores >= 1))
        self.assertTrue(np.all(mos_scores <= 5))

    def test_binomos_returns_votes(self):
        """Test binomos returns votes when mos=False"""
        quality = np.array([2.0, 3.0])
        n_v = 10
        seed = 42

        votes = ma.binomos(mos=False, quality=quality, n_v=n_v, seed=seed)
        self.assertEqual(votes.shape, (n_v, len(quality)))

    def test_adjacent_two_choice_output_shape(self):
        """Test adjacent_two_choice output shape"""
        quality = np.array([1.5, 2.5, 3.5, 4.5])
        n_v = 8

        votes = ma.adjacent_two_choice(quality=quality, n_v=n_v, seed=42)

        self.assertEqual(votes.shape, (n_v, len(quality)))
        self.assertTrue(np.all(votes >= 1))
        self.assertTrue(np.all(votes <= 5))

    def test_maximum_variance_unimodal_output_shape(self):
        """Test maximum_variance_unimodal output shape"""
        quality = np.array([1.5, 3.0, 4.5])
        n_v = 10

        votes = ma.maximum_variance_unimodal(quality=quality, n_v=n_v, seed=42)

        self.assertEqual(votes.shape, (n_v, len(quality)))
        self.assertTrue(np.all(votes >= 1))
        self.assertTrue(np.all(votes <= 5))

    def test_binovotes_variance_function(self):
        """Test binovotes_variance calculation"""
        quality = 3.0
        s_L = 1
        s_H = 5

        var = ma.binovotes_variance(quality=quality, s_L=s_L, s_H=s_H)

        # Variance should be positive for intermediate values
        self.assertGreater(var, 0)
        # At extremes, variance should be 0
        var_at_low = ma.binovotes_variance(quality=1.0, s_L=1, s_H=5)
        var_at_high = ma.binovotes_variance(quality=5.0, s_L=1, s_H=5)
        self.assertAlmostEqual(var_at_low, 0.0, places=10)
        self.assertAlmostEqual(var_at_high, 0.0, places=10)

    def test_minimum_vote_variance_function(self):
        """Test minimum_vote_variance calculation"""
        # At integer values, minimum variance should be 0
        min_var_int = ma.minimum_vote_variance(3.0)
        self.assertAlmostEqual(min_var_int, 0.0, places=10)

        # At midpoint between integers, variance should be 0.25
        min_var_mid = ma.minimum_vote_variance(2.5)
        self.assertAlmostEqual(min_var_mid, 0.25, places=10)

    def test_maximum_unimodal_vote_variance_scalar(self):
        """Test maximum_unimodal_vote_variance with scalar input"""
        quality = 3.0
        var = ma.maximum_unimodal_vote_variance(quality=quality)
        self.assertGreater(var, 0)

    def test_maximum_unimodal_vote_variance_array(self):
        """Test maximum_unimodal_vote_variance with array input"""
        quality = np.array([1.0, 2.5, 3.0, 4.5, 5.0])
        var = ma.maximum_unimodal_vote_variance(quality=quality)
        self.assertEqual(var.shape, quality.shape)
        self.assertTrue(np.all(var >= 0))

    def test_effective_votes(self):
        """Test effective_votes calculation"""
        n_bino = 10
        n_mos = 10
        n_s = 5

        effective = ma.effective_votes(n_bino=n_bino, n_mos=n_mos, n_s=n_s)

        # Effective votes should be positive
        self.assertGreater(effective, 0)
        # For equal inputs, should be less than either input
        self.assertLess(effective, n_bino)

    def test_bin_vars_output_structure(self):
        """Test bin_vars output structure and content"""
        means = np.array([1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0])
        vars = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
        n_votes = np.array([5, 5, 5, 5, 5, 5, 5])

        binned = ma.bin_vars(means=means, vars=vars, n_votes=n_votes, step=0.5)

        # Should return a list
        self.assertIsInstance(binned, list)
        # Each item should be a dict with required keys
        for item in binned:
            self.assertIsInstance(item, dict)
            self.assertIn("mos", item)
            self.assertIn("data var", item)
            self.assertIn("n files", item)
            self.assertIn("n_v", item)

    def test_mixed_behavior_votes_output_shape(self):
        """Test mixed_behavior_votes output shape and range"""
        quality = np.array([2.0, 3.0, 4.0])
        vars = np.array([0.3, 0.4, 0.3])
        n_v = 10

        votes = ma.mixed_behavior_votes(quality=quality, vars=vars, n_v=n_v, seed=42)

        self.assertEqual(votes.shape, (n_v, len(quality)))
        self.assertTrue(np.all(votes >= 1))
        self.assertTrue(np.all(votes <= 5))


class TestFitDataModule(unittest.TestCase):
    """Test cases for fit_data.py module"""

    def test_fourth_degree_least_squares_with_weights(self):
        """Test fourth_degree_least_squares with explicit weights"""
        mos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vars = np.array([0.1, 0.4, 0.5, 0.4, 0.1])
        weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0])

        scale, shift = ma.fourth_degree_least_squares(
            mos=mos, vars=vars, weights=weights
        )

        # Results should be real numbers
        self.assertIsInstance(scale, (float, np.floating))
        self.assertIsInstance(shift, (float, np.floating))

    def test_least_squares_weights(self):
        """Test least_squares_weights calculation"""
        in_edges = np.array([True, False, False, False, True])
        target_edge_frac = 0.5

        weights = ma.least_squares_weights(in_edges=in_edges, target_edge_frac=target_edge_frac)

        # Weights should sum properly
        edge_weight = np.sum(weights[in_edges])
        middle_weight = np.sum(weights[~in_edges])
        
        self.assertGreater(edge_weight, 0)
        self.assertGreater(middle_weight, 0)
        # Edge weights should be higher than middle weights for this target
        self.assertGreater(np.mean(weights[in_edges]), np.mean(weights[~in_edges]))

    def test_check_minimum_variance_violations(self):
        """Test check_minimum_variance_violations function"""
        # Case where there should be no violation
        scale = 0.1
        shift = 0.5
        violation = ma.check_minimum_variance_violations(scale=scale, shift=shift)
        self.assertFalse(violation)

        # Case where violation might occur
        scale = -10.0
        shift = -10.0
        violation = ma.check_minimum_variance_violations(scale=scale, shift=shift)
        # Result depends on the specific values, just check it returns boolean
        self.assertIsInstance(violation, (bool, np.bool_))

    def test_fit_fourth_degree_poly_basic(self):
        """Test fit_fourth_degree_poly basic functionality"""
        mos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vars = np.array([0.1, 0.4, 0.5, 0.4, 0.1])

        scale, shift, target_edge_frac = ma.fit_fourth_degree_poly(
            mos=mos, vars=vars, ignore_violation=True
        )

        self.assertIsInstance(scale, (float, np.floating))
        self.assertIsInstance(shift, (float, np.floating))
        # target_edge_frac is None when no bisection is used
        self.assertTrue(target_edge_frac is None or isinstance(target_edge_frac, (float, np.floating)))

    def test_fit_fourth_degree_poly_with_violation_handling(self):
        """Test fit_fourth_degree_poly with potential minimum variance violation"""
        mos = np.array([1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 4.5, 4.8, 5.0])
        vars = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 0.3, 0.1, 0.05, 0.01])

        # Without ignore_violation, should handle violations properly
        scale, shift, target_edge_frac = ma.fit_fourth_degree_poly(
            mos=mos, vars=vars, ignore_violation=False
        )

        # Results should be valid numbers
        self.assertIsInstance(scale, (float, np.floating))
        self.assertIsInstance(shift, (float, np.floating))
        # If bisection was used, target_edge_frac should be set
        if not (target_edge_frac is None):
            self.assertGreater(target_edge_frac, 0)
            self.assertLessEqual(target_edge_frac, 1)


class TestDistributionsModule(unittest.TestCase):
    """Test cases for distributions.py module"""

    def test_uniform_pdf_initialization(self):
        """Test UniformPDF initialization and basic methods"""
        pdf = ma.UniformPDF(a=1, b=5)
        self.assertEqual(pdf.a, 1)
        self.assertEqual(pdf.b, 5)
        self.assertEqual(pdf.scale, 4)

    def test_uniform_pdf_eval(self):
        """Test UniformPDF evaluation"""
        pdf = ma.UniformPDF(a=1, b=5)
        # Within range
        val = pdf.pdf(3.0)
        self.assertAlmostEqual(val, 0.25)  # 1/4

    def test_triangular_pdf_initialization(self):
        """Test TriangularPDF initialization"""
        pdf = ma.TriangularPDF(a=1, b=5, m=3)
        self.assertEqual(pdf.a, 1)
        self.assertEqual(pdf.b, 5)
        self.assertEqual(pdf.m, 3)

    def test_triangular_pdf_eval(self):
        """Test TriangularPDF evaluation"""
        pdf = ma.TriangularPDF(a=1, b=5, m=3)
        val = pdf.pdf(3.0)
        self.assertGreater(val, 0)

    def test_beta_pdf_initialization(self):
        """Test BetaPDF initialization"""
        pdf = ma.BetaPDF(alpha=2, beta=2, a=1, b=5)
        self.assertEqual(pdf.alpha, 2)
        self.assertEqual(pdf.beta, 2)
        self.assertEqual(pdf.a, 1)
        self.assertEqual(pdf.b, 5)

    def test_beta_pdf_eval(self):
        """Test BetaPDF evaluation"""
        pdf = ma.BetaPDF(alpha=2, beta=2, a=1, b=5)
        val = pdf.pdf(3.0)
        self.assertGreater(val, 0)

    def test_max_unimodal_pdf_initialization(self):
        """Test MaxUnimodalPDF initialization"""
        pdf = ma.MaxUnimodalPDF()
        self.assertEqual(pdf.v_L, 1)
        self.assertEqual(pdf.v_H, 5)
        self.assertEqual(pdf.n_s, 5)

    def test_max_unimodal_pdf_at_extremes(self):
        """Test MaxUnimodalPDF at extreme values"""
        pdf = ma.MaxUnimodalPDF()
        # At lower bound, pmf should be all 0 except first
        pmf_low = pdf.pmf(1)
        self.assertEqual(pmf_low[0], 1)
        self.assertTrue(np.all(np.array(pmf_low[1:]) == 0))

        # At upper bound, pmf should be all 0 except last
        pmf_high = pdf.pmf(5)
        self.assertEqual(pmf_high[-1], 1)
        self.assertTrue(np.all(np.array(pmf_high[:-1]) == 0))

    def test_max_unimodal_pdf_middle_value(self):
        """Test MaxUnimodalPDF at middle values"""
        pdf = ma.MaxUnimodalPDF()
        pmf_mid = pdf.pmf(3.0)
        # Should be a valid probability distribution
        self.assertAlmostEqual(np.sum(pmf_mid), 1.0, places=10)

    def test_max_unimodal_mean_variance(self):
        """Test MaxUnimodalPDF mean and variance calculations"""
        pdf = ma.MaxUnimodalPDF()
        quality = 3.0
        mean = pdf.mean(quality)
        var = pdf.var(quality)
        
        self.assertAlmostEqual(mean, quality, places=10)
        self.assertGreater(var, 0)

    def test_min_variance_pdf_initialization(self):
        """Test MinVariancePDF initialization"""
        pdf = ma.MinVariancePDF()
        self.assertEqual(pdf.v_L, 1)
        self.assertEqual(pdf.v_H, 5)
        self.assertEqual(pdf.n_s, 5)

    def test_min_variance_pdf_integer_quality(self):
        """Test MinVariancePDF at integer quality values"""
        pdf = ma.MinVariancePDF()
        # For quality 3.0: floor=3, ceil=3, so pmf[3] gets all probability
        pmf = pdf.pmf(3.0)
        # This corresponds to rating value 4 (index 3 in 0-based, but ratings [1,2,3,4,5])
        expected = [0, 0, 0, 1, 0]
        self.assertEqual(pmf, expected)

    def test_min_variance_pdf_fractional_quality(self):
        """Test MinVariancePDF at fractional quality values"""
        pdf = ma.MinVariancePDF()
        pmf = pdf.pmf(3.5)
        # Should be probability distribution summing to 1
        self.assertAlmostEqual(np.sum(pmf), 1.0)

    def test_binovotes_initialization(self):
        """Test BinoVotes initialization"""
        quality_pdf = ma.MinVariancePDF()
        bv = ma.BinoVotes(v_L=1, v_H=5, n_s=5, n_v=10, quality_pdf=quality_pdf)
        
        self.assertEqual(bv.v_L, 1)
        self.assertEqual(bv.v_H, 5)
        self.assertEqual(bv.n_s, 5)
        self.assertEqual(bv.n_v, 10)
        self.assertEqual(bv.len, 40)  # n_v * (n_s - 1)


if __name__ == "__main__":
    unittest.main()

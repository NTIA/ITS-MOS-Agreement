import numpy as np

import mos_agreement as ma


def test_fourth_degree_least_squares_recovers_exact_curve():
    mos = np.linspace(1, 5, 17)
    expected_scale = 0.08
    expected_shift = 0.45
    variances = (mos - 1) * (5 - mos) * (
        expected_scale * (mos - 3) ** 2 + expected_shift
    )

    scale, shift = ma.fourth_degree_least_squares(mos=mos, vars=variances)

    assert np.isclose(scale, expected_scale)
    assert np.isclose(shift, expected_shift)


def test_least_squares_weights_match_requested_edge_fraction():
    in_edges = np.array([True, True, False, False])

    weights = ma.least_squares_weights(in_edges, target_edge_frac=0.75)

    assert np.isclose(weights[in_edges].sum() / weights.sum(), 0.75)
    assert np.isclose(weights[~in_edges].sum() / weights.sum(), 0.25)


def test_minimum_variance_constraint_detects_valid_and_invalid_fits():
    assert not ma.check_minimum_variance_violations(scale=0.1, shift=0.5)
    assert ma.check_minimum_variance_violations(scale=0, shift=0)

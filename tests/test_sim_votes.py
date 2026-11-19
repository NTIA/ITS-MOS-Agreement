import numpy as np

import mos_agreement as ma


def test_binovotes_at_quality_endpoints_are_deterministic():
    votes = ma.binovotes(quality=[1, 5], n_v=20, seed=7)

    np.testing.assert_array_equal(votes[:, 0], np.ones(20))
    np.testing.assert_array_equal(votes[:, 1], np.full(20, 5))


def test_sim_setup_without_seed_still_normalizes_scalar_quality():
    rng, quality = ma.sim_setup(3.5, seed=None)

    assert isinstance(rng, np.random.Generator)
    np.testing.assert_array_equal(quality, [3.5])


def test_adjacent_two_choice_votes_use_only_neighboring_ratings():
    quality = np.array([1.25, 2.5, 4.75])

    votes = ma.adjacent_two_choice(quality=quality, n_v=100, seed=17)

    for column, mean in enumerate(quality):
        assert set(np.unique(votes[:, column])).issubset(
            {np.floor(mean), np.ceil(mean)}
        )
        assert np.isclose(votes[:, column].mean(), mean, atol=0.2)


def test_mixed_behavior_votes_are_reproducible_with_seed():
    args = {
        "quality": [2.4, 3.6],
        "vars": [0.35, 0.45],
        "n_v": 25,
        "seed": 23,
    }

    first = ma.mixed_behavior_votes(**args)
    second = ma.mixed_behavior_votes(**args)

    np.testing.assert_array_equal(first, second)


def test_mixed_behavior_votes_supports_variance_above_binovotes():
    votes = ma.mixed_behavior_votes(
        quality=[3],
        vars=[1.5],
        n_v=25,
        seed=31,
    )

    assert votes.shape == (25, 1)
    assert np.all((votes >= 1) & (votes <= 5))

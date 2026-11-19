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


def test_standalone_atc_and_mvu_generators_repeat_with_seed():
    """Both public model generators should be repeatable for the same seed."""
    quality = np.array([1.5, 3.25, 4.5])

    atc_first = ma.adjacent_two_choice(quality=quality, n_v=20, seed=13)
    atc_second = ma.adjacent_two_choice(quality=quality, n_v=20, seed=13)
    mvu_first = ma.maximum_variance_unimodal(quality=quality, n_v=20, seed=13)
    mvu_second = ma.maximum_variance_unimodal(quality=quality, n_v=20, seed=13)

    np.testing.assert_array_equal(atc_first, atc_second)
    np.testing.assert_array_equal(mvu_first, mvu_second)


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
    """Integer quality inputs retain fractional mixing probabilities."""
    votes = ma.mixed_behavior_votes(
        quality=[3],
        vars=[1.5],
        n_v=20_000,
        seed=31,
    )

    assert votes.shape == (20_000, 1)
    assert np.all((votes >= 1) & (votes <= 5))
    assert np.isclose(votes.mean(), 3, atol=0.03)
    assert np.isclose(votes.var(), 1.5, atol=0.04)


def test_mixed_behavior_votes_preserve_target_means_and_variances():
    """The ATC and MVU branches should realize their requested moments."""
    quality = np.array([2.5, 3.0])
    target_variance = np.array([0.5, 1.5])
    votes = ma.mixed_behavior_votes(
        quality=quality,
        vars=target_variance,
        n_v=40_000,
        seed=37,
    )

    np.testing.assert_allclose(votes.mean(axis=0), quality, atol=0.025)
    np.testing.assert_allclose(votes.var(axis=0), target_variance, atol=0.025)


def test_maximum_variance_unimodal_sampling_preserves_moments():
    """Batched MVU sampling should preserve each item's mean and variance."""
    quality = np.array([1.5, 3.25, 4.0])
    expected_variance = ma.maximum_unimodal_vote_variance(quality)

    votes = ma.maximum_variance_unimodal(
        quality=quality,
        n_v=25_000,
        seed=41,
    )

    np.testing.assert_allclose(votes.mean(axis=0), quality, atol=0.03)
    np.testing.assert_allclose(votes.var(axis=0), expected_variance, atol=0.04)

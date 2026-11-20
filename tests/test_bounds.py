import numpy as np

import mos_agreement as ma


def test_mos_data_binovotes_average_vote_var_at_scale_endpoints():
    for mos_mean in (1, 5):
        vote_var = ma.mos_data_binovotes_average_vote_var(
            mos_mean=mos_mean, mos_var=0, n_v=10
        )
        assert vote_var == 0


def test_quality_distribution_bounds_zero_vote_error():
    rmse, corr = ma.quality_distribution_bounds(quality_var=2.5, vote_se2=0)

    assert rmse == 0
    assert corr == 1


def test_mos_data_bounds_averages_per_item_standard_errors():
    vote_var = np.array([0.2, 0.6, 1.0])
    n_v = np.array([2, 3, 5])
    average_se2 = np.mean(vote_var / n_v)
    quality_var = 1.4 - average_se2

    rmse, corr = ma.mos_data_bounds(mos_var=1.4, vote_var=vote_var, n_v=n_v)

    assert np.isclose(rmse, np.sqrt(average_se2))
    assert np.isclose(corr, np.sqrt(quality_var / 1.4))

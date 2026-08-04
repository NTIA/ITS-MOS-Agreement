import numpy as np


def mos_data_bounds(mos_var, vote_var, n_v, s_L=1, s_H=5, n_s=5):
    """
    mos_data_bounds

    Estimate performance bounds for RMSE and Correlation based on MOS data statistics.

    Data-driven bounds for datasets that include MOS and an associated variance value
    for each MOS value.

    Parameters
    ----------
    mos_var : np.float
        Variance estimate of MOS distribution.
    vote_var : np.array, np.float
        Vote variance for each MOS value in the dataset. If passed in as a float this
        should be the average vote variance across the dataset. Calculation is more
        accurate if this is an array.
    n_v : np.array, int, float
        Number of votes per file for each file in the dataset. If passed in as a float
        this should be the average number of votes per file in the dataset. Calculation
        is more accurate if this is an array.

    Returns
    -------
    rmse: np.float
        Estimate of RMSE bound between MOS and true quality.
    corr: np.float
        Estimate of Correlation bound between MOS and true quality.

    Raises
    ------
    ValueError
        _description_
    """
    # Get the standard error squared for each MOS value
    vote_se2 = vote_var / n_v
    average_vote_se2 = np.mean(vote_se2)
    quality_var = mos_var - average_vote_se2
    rmse, corr = quality_distribution_bounds(
        quality_var=quality_var, vote_se2=average_vote_se2
    )
    return rmse, corr


def mos_data_binovotes_bounds(mos_mean, mos_var, n_v, s_L=1, s_H=5, n_s=5):
    """
    mos_data_binovotes_bounds

    Estimate performance bounds for RMSE and Correlation based on MOS data statistics
    under BinoVotes voting model.

    Data-driven bounds for datasets that include only MOS values with no associated vote
    variance information. Instead this function assumes vote variance according to a
    BinoVotes voting model.

    Parameters
    ----------
    mos_mean : np.float
        Mean estimate of MOS distribution.
    mos_var : np.float
        Variance estimate of MOS distribution.
    n_v : int, float
        Average number of votes per file in the dataset.
    s_L : int, optional
        Lower value of rating scale, by default 1.
    s_H : int, optional
        Highest value of the rating scale, by default 5.
    n_s : int, optional
        Number of values in the rating scale, by default 5.

    Returns
    -------
    rmse: np.float
        Estimate of RMSE bound between MOS and true quality.
    corr: np.float
        Estimate of Correlation bound between MOS and true quality.

    Raises
    ------
    ValueError
        _description_
    """
    binovotes_average_vote_var = mos_data_binovotes_average_vote_var(
        mos_mean=mos_mean, mos_var=mos_var, n_v=n_v, s_L=s_L, s_H=s_H, n_s=n_s
    )
    rmse, corr = mos_data_bounds(
        mos_var=mos_var, vote_var=binovotes_average_vote_var, n_v=n_v
    )
    return rmse, corr


def mos_data_binovotes_average_vote_var(mos_mean, mos_var, n_v, s_L=1, s_H=5, n_s=5):
    """
    mos_data_binovotes_average_vote_var

    Estimate the average vote variance under BinoVotes voting model from MOS data.

    Parameters
    ----------
    mos_mean : np.float
        Mean estimate of MOS distribution.
    mos_var : np.float
        Variance estimate of MOS distribution.
    n_v : int, float
        Average number of votes per file in the dataset.
    s_L : int, optional
        Lower value of rating scale, by default 1.
    s_H : int, optional
        Highest value of the rating scale, by default 5.
    n_s : int, optional
        Number of values in the rating scale, by default 5.

    Returns
    -------
    average_vote_var: np.float
        Estimate of average vote variance across the dataset.

    Raises
    ------
    ValueError
        _description_
    """
    n_m = n_v * (n_s - 1)
    scale_factor = n_v / (n_m - 1)
    binovotes_average_vote_var = scale_factor * (
        (mos_mean - s_L) * (s_H - mos_mean) - mos_var
    )
    return binovotes_average_vote_var


def quality_distribution_bounds(
    quality_var,
    vote_se2,
):
    """
    quality_distribution_bounds

    Estimate performance bounds for RMSE and Correlation based on quality distribution
    statistics and an expected vote variance.

    Note that this function requires knowledge of the true quality distribution,
    which is not available in real subjective experiments. This function is primarily
    useful for simulations where the true quality distribution is known, or when
    estimates of the quality distribution can be made through the MOS distribution.

    Parameters
    ----------
    quality_var : np.float
        Variance value of quality distribution.
    vote_se2 : np.float
        Standard error squared of votes under a voting model across the entire voting
        scale. Is equal to `E[v_r(y)] / n_v`.

    Returns
    -------
    rmse: np.float
        RMSE bound between MOS and true quality.
    corr: np.float
        Correlation bound between MOS and true quality.

    Raises
    ------
    ValueError
        _description_
    """
    rmse = np.sqrt(vote_se2)
    corr = np.sqrt(quality_var / (quality_var + vote_se2))
    return rmse, corr


def quality_distribution_binovotes_bounds(
    quality_mean, quality_var, n_v, s_L=1, s_H=5, n_s=5
):
    """
    quality_distribution_binovotes_bounds

    Estimate performance bounds for RMSE and Correlation based on quality distribution
    statistics under BinoVotes voting model.

    Note that this function requires knowledge of the true quality distribution,
    which is not available in real subjective experiments. This function is primarily
    useful for simulations where the true quality distribution is known, or when
    estimates of the quality distribution can be made through the MOS distribution.

    Parameters
    ----------
    quality_mean : np.float
        Mean value of quality distribution.
    quality_var : np.float
        Variance value of quality distribution.
    n_v : int, float
        Average number of votes per file in the dataset.
    s_L : int, optional
        Lower value of rating scale, by default 1.
    s_H : int, optional
        Highest value of rating scale, by default 5.
    n_s : int, optional
        Number of values in the rating scale, by default 5.

    Returns
    -------
    rmse: np.float
        RMSE bound between MOS and true quality.
    corr: np.float
        Correlation bound between MOS and true quality.

    Raises
    ------
    ValueError
        _description_
    """
    numerator = (quality_mean - s_L) * (s_H - quality_mean) - quality_var
    denominator = n_v * (n_s - 1)
    mse = numerator / denominator
    rmse = np.sqrt(mse)
    corr = np.sqrt(quality_var / (quality_var + mse))
    return rmse, corr

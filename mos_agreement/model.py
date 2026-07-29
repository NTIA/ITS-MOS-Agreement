import os
import yaml

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.stats as stats

from scipy.special import binom

from .distributions import MaxUnimodalPDF


def mos_data_bounds(mos_var, average_vote_var, n_v, s_L=1, s_H=5, n_s=5):
    """
    mos_data_bounds

    Estimate performance bounds for RMSE and Correlation based on MOS data statistics.

    Data-driven bounds for datasets that include MOS and an associated variance value
    for each MOS value.

    Parameters
    ----------
    mos_var : np.float
        Variance estimate of MOS distribution.
    average_vote_var : np.float
        Estimate of the average vote variance across the dataset.
    n_v : int, float
        Average number of votes per file in the dataset.

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
    quality_var = mos_var - average_vote_var / n_v
    rmse, corr = quality_distribution_bounds(
        quality_var=quality_var, expected_vote_var=average_vote_var, n_v=n_v
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
        mos_var=mos_var, average_vote_var=binovotes_average_vote_var, n_v=n_v
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
    expected_vote_var,
    n_v,
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
    expecte_vote_var : np.float
        Expected value of voting variance under a voting model across the entire voting
        scale.
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
        RMSE bound between MOS and true quality.
    corr: np.float
        Correlation bound between MOS and true quality.

    Raises
    ------
    ValueError
        _description_
    """
    rmse = np.sqrt(expected_vote_var / n_v)
    corr = np.sqrt(quality_var / (quality_var + expected_vote_var / n_v))
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


# ---------------------
# BinoVotes Simulations
# ---------------------
def sim_setup(quality, seed):
    """
    sim_setup

    Setup for vote simulations.

    Parameters
    ----------
    quality : float, list, np.array
        Quality values which votes converge to.
    seed : int
        Seed for random number generation.
    Returns
    -------
    rng : np.random.Generator
        Random number generator initialized with the given seed.
    quality : np.array
        Quality values converted to a numpy array.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    if isinstance(quality, list):
        quality = np.array(quality)
    elif isinstance(quality, (int, float)):
        quality = np.array([quality])
    return rng, quality


def binovotes(quality, n_v, step=1, s_L=1, s_H=5, seed=None):
    """
    binovotes

    Generate votes according to BinoVotes model.

    Parameters
    ----------
    quality : float, list, np.array
        Quality values which votes converge to.
    n_v : int
        Number of votes per file.
    step : int, optional
        Step size of rating scale, by default 1.
    s_L : int, optional
        Lower value of rating scale, by default 1.
    s_H : int, optional
        Highest value of the rating scale, by default 5.
    seed : int, optional
        Seed for random number generation, by default None.

    Returns
    -------
    np.array
        A (n_v x len(quality)) array of votes generated according to BinoVotes model.
    """
    rng, quality = sim_setup(quality, seed)
    # Define the binomial n value based off of the given scale
    scale = np.arange(s_L, (s_H + step), step)
    n_bino = len(scale) - 1

    # Convert from quality scale to probability of successful trial scale
    p_bino = (quality - s_L) / (s_H - s_L)
    # BinoVotes
    votes = s_L + step * rng.binomial(n_bino, p_bino, (n_v, quality.size))
    return votes


def binomos(mos=True, *args, **kwargs):
    """
    binomos

    Convenient wrapper to generate MOS scores from BinoVotes.

    Parameters
    ----------
    mos : bool, optional
        Flag to return MOS scores rather than individual votes via averaging,
        by default True.

    Returns
    -------
    np.array
        Generated MOS scores or individual votes.
    """
    votes = binovotes(*args, **kwargs)
    if mos:
        votes = np.mean(votes, 0)
    return votes


def mixed_behavior_votes(quality, vars, n_v, step=1, s_L=1, s_H=5, seed=None):
    """
    mixed_behavior_votes

    Generate votes according to a mixed behavior model.

    Parameters
    ----------
    quality : float
        Quality votes converge to.
    vars : float
        Variance of the vote distribution.
    n_v : int
        Number of votes per file.
    step : int, optional
        Step size of rating scale, by default 1.
    s_L : int, optional
        Lower value of rating scale, by default 1.
    s_H : int, optional
        Highest value of the rating scale, by default 5.
    seed : _type_, optional
        Seed for random number generation, by default None.

    Returns
    -------
    np.array
        A (n_v x len(quality)) array of votes generated according to the mixed behavior
        model.
    """
    if not (s_L == 1 and s_H == 5 and step == 1):
        raise ValueError(
            "Mixed behavior model currently only supports standard 1-5 rating scale."
        )
    rng, quality = sim_setup(quality, seed)
    # Get alpha values for the given quality and variance
    alpha_min, alpha_max, use_max = get_alpha(quality, vars)

    # Initialize single alpha array
    alpha = np.zeros_like(quality)
    alpha[use_max] = alpha_max[use_max]
    alpha[~use_max] = alpha_min[~use_max]
    # Draw probabilities to determine behavior
    behavior_probs = rng.random(size=len(quality))

    # Determine when to use BinoVotes
    use_bino = behavior_probs < alpha

    # Initialize votes array (n_v x len(quality))
    votes = np.zeros((n_v, len(quality)), dtype=int)
    # Generate votes according to BinoVotes for those that use BinoVotes
    votes[:, use_bino] = binovotes(
        quality=quality[use_bino], n_v=n_v, s_L=s_L, s_H=s_H, step=step, seed=seed
    )
    # Determine when to use atc vs mvu for remaining votes
    use_alternate = ~use_bino
    use_atc = use_alternate & ~use_max
    use_mvu = use_alternate & use_max
    votes[:, use_atc] = adjacent_two_choice(
        quality=quality[use_atc], n_v=n_v, s_L=s_L, s_H=s_H, step=step, seed=seed
    )
    votes[:, use_mvu] = maximum_variance_unimodal(
        quality=quality[use_mvu], n_v=n_v, seed=seed
    )
    return votes


def effective_votes(n_bino, n_mos, n_s=5):
    """
    effective_votes

    Effective number of votes per file when we generate BinoMOS using n_bino votes with
    a MOS value that comes from n_mos votes per file.

    Parameters
    ----------
    n_bino : int
        Number of votes per file in binovotes draw
    n_mos : int
        Number of votes per file associated with MOS value used as "truth"
    n_s : int, optional
        Number of values in rating scale, by default 5
    """
    effective = 1 / (1 / n_bino + 1 / n_mos - 1 / (n_bino * n_mos * (n_s - 1)))
    return effective


def adjacent_two_choice(quality, n_v, step=1, s_L=1, s_H=5, seed=None):
    """
    adjacent_two_choice

    Generate votes according to Adjacent Two-Choice (ATC) voting model.

    Parameters
    ----------
    quality : float
        Quality votes converge to.
    n_v : int
        Number of votes per file.
    step : int, optional
        Step size of rating scale, by default 1.
    s_L : int, optional
        Lower value of rating scale, by default 1.
    s_H : int, optional
        Highest value of the rating scale, by default 5.
    seed : _type_, optional
        Seed for random number generation, by default None.

    Returns
    -------
    np.array
        A (n_v x len(quality)) array of votes generated according to the Adjacent
        Two-Choice (ATC) model.
    """
    rng, quality = sim_setup(quality, seed)

    # Convert from original quality scale to integers scale (0,1,...,n_s-1), quality
    # values can still be floats
    int_scale = (quality - s_L) / step
    # Lower neighbor for quality values
    lower = np.floor(int_scale)
    # Upper neighbor for quality values
    upper = np.ceil(int_scale)
    # Difference between quality on int scale and its upper (becomes probability of
    # adding 1 to lower)
    diff = upper - int_scale
    # Draw probabilities, will determine if we add vote lower or upper
    probs = rng.random(size=(n_v, int_scale.size))
    # Determine if vote is upper or lower, false for upper, true for lower, e.g., 1 or 0
    vote_upper = probs > diff
    # Check if quality is on the rating scale directly (in this case it is an integer)
    is_lower = int_scale == lower
    # Set all votes for those quality values to lower (or upper, since they are the
    # same)
    vote_upper[:, is_lower] = False
    # Get vote on the integer scale
    vote_int_scale = lower + vote_upper
    # Convert back to original quality scale
    votes = vote_int_scale * step + s_L

    return votes


def maximum_variance_unimodal(quality, n_v, seed=None):
    """
    maximum_variance_unimodal

    Generate votes according to Maximum Variance Unimodal (MVU) voting model.

    Currently does not support ratings scales outside of the standard, integer 1-5
    scale.

    Parameters
    ----------
    quality : float
        Quality votes converge to.
    n_v : int
        Number of votes per file.
    seed : _type_, optional
        Seed for random number generation, by default None.

    Returns
    -------
    np.array
        A (n_v x len(quality)) array of votes generated according to the Maximum
        Variance Unimodal (MVU) model.
    """
    rng, quality = sim_setup(quality, seed)
    # PDF generator
    pdf_gen = MaxUnimodalPDF()
    # Get PDF for each quality value
    pdfs = np.array([pdf_gen.pdf(q) for q in quality])
    # Define rating scale as [1, 2, 3, 4, 5]
    scale = np.arange(1, 6)
    # Generate votes from scale according to pdfs
    votes = np.array([rng.choice(scale, size=n_v, p=pdf) for pdf in pdfs]).transpose()
    return votes


def binovotes_variance(quality, s_L=1, s_H=5):
    """
    binovotes_variance

    Assumes only integer values in rating scale, e.g., an individual cannot vote 1.5.

    Parameters
    ----------
    quality : float
        quality value.
    s_L : int, optional
        Lower limit of rating scale, by default 1
    s_H : int, optional
        Upper limit of rating scale, by default 5

    Returns
    -------
    float
        Variance of BinoVotes distribution for given quality value.
    """
    c = 1 / (s_H - s_L)
    return c * (quality - s_L) * (s_H - quality)


def minimum_vote_variance(quality):
    """
    minimum_vote_variance

    Assumes only integer values in rating scale, e.g., an individual cannot vote 1.5.

    Parameters
    ----------
    quality : float
        quality value.

    Returns
    -------
    float
        Variance from adjacent two choice voting model for given quality value.
    """
    return (quality - np.floor(quality)) * (np.ceil(quality) - quality)


def maximum_unimodal_vote_variance(quality, s_L=1, s_H=5):
    """
    maximum_unimodal_vote_variance

    Parameters
    ----------
    quality : float
        quality value.
    s_L : int, optional
        Lower limit of rating scale, by default 1
    s_H : int, optional
        Upper limit of rating scale, by default 5

    Returns
    -------
    float
        Variance from maximum variance unimodal voting model for given quality value.
    """
    # If mos is not a float or int it is a list or np.array and we need to call
    # these individually
    pdf = MaxUnimodalPDF()
    if isinstance(quality, (float, int)):
        var = pdf.var(quality, s_L=s_L, s_H=s_H)
    else:
        var = np.array([pdf.var(q) for q in quality])
    return var


def get_alpha(quality, var_target):
    var_bv = binovotes_variance(quality=quality)
    var_min = minimum_vote_variance(quality=quality)
    var_max = maximum_unimodal_vote_variance(quality=quality)

    alpha_min = (var_target - var_min) / (var_bv - var_min)
    alpha_max = (var_target - var_max) / (var_bv - var_max)

    # This can happen on the extreme edges of scale. Technically could be fixed by
    # making `step` in `check_minimum_variance_violations` really small, but the
    # differences are negligible. The violations limited to extreme edges of the scale
    # (1, 1.001) and (4.999, 5) and violations are extremely small.
    alpha_min[alpha_min < 0] = 0
    # alpha_max[alpha_max < 0] = 0

    # At edges of the scale variance is 0 so the alphas can become nan
    alpha_min[np.isnan(alpha_min)] = 1
    alpha_max[np.isnan(alpha_max)] = 1

    use_max = var_target > var_bv

    return alpha_min, alpha_max, use_max


# ---------------------
# Convenience functions
# ---------------------
def bin_vars(means, vars, n_votes, step=0.5):
    """
    bin_vars

    Bin observed variance values.

    Parameters
    ----------
    means : np.array
        Observed MOS values, or mean values.
    vars : np.array
        Observed variances associated with each MOS value.
    n_votes : np.array
        Number of votes associated with each MOS value.
    step : float, optional
        Step size for binning, by default 0.5

    Returns
    -------
    pd.DataFrame
        DataFrame containing binned MOS values, associated variances, number of files,
        and number of votes.
    """
    # Define MOS binning array (treat edges of the scale separately)
    mos_vals = np.concatenate([[1, 1.001], np.arange(1 + step, 5, step), [4.999, 5]])
    delta = step / 2
    stats_list = []
    for mos in mos_vals:
        # Get the indices for this mos value
        if mos == 1 or mos == 5:
            # Special case for ends of the scale, no binning
            mean_ix = means == mos
        elif mos == 1.001:
            # Bin for (1, 1.25]
            mean_ix = (1 < means) & (means <= mos + delta)
        elif mos == 4.999:
            # Bin for [4.75, 5)
            mean_ix = (mos - delta <= means) & (means < 5)
        else:
            mean_ix = (mos - delta <= means) & (means < mos + delta)
        mos_vs = means[mean_ix]
        mos_v = np.mean(mos_vs)

        # All the variances observed at this MOS value
        mean_vars = vars[mean_ix]
        # Average variance observed at this MOS value
        var_mean = np.mean(mean_vars)

        n_votes_for_mos = n_votes[mean_ix]
        mean_n_votes = np.mean(n_votes_for_mos)

        # Number of files that contribute to these values
        n_points = np.sum(mean_ix)
        # Save out values
        mean_vals = {
            # "dataset": data_name,
            "mos": mos_v,
            "data var": var_mean,
            "n files": n_points,
            "n_v": mean_n_votes,
        }
        stats_list.append(mean_vals)

    var_df = pd.DataFrame(stats_list)
    return var_df

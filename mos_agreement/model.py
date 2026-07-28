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
    _type_
        _description_
    """
    rng, quality = sim_setup(quality, seed)

    # # Define the binomial n value based off of the given scale
    # scale = np.arange(s_L, (s_H + step), step)
    # n_bino = len(scale) - 1

    # # Convert from quality scale to probability of successful trial scale
    # p_bino = (quality - s_L) / (s_H - s_L)

    # Mixed behavior model: combine BinoVotes with Adjacent Two-Choice or Maximum
    # Variance Unimodal
    # bino_votes = rng.binomial(n_bino, p_bino, (n_v, quality.size))
    # gaussian_noise = rng.normal(0, np.sqrt(vars), (n_v, quality.size))
    # TODO verify we are in the valid region
    raise ValueError(
        "Mixed behavior model not implemented yet. Please use binovotes or binomos"
        " instead."
    )
    # return votes


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


# TODO function to get mixture parameter given quality and variance

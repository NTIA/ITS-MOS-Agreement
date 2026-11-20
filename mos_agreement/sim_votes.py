import numpy as np

from .distributions import MaxUnimodalPDF


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
    elif isinstance(quality, (int, float, np.number)):
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


def _sample_discrete_votes(rng, pmfs, n_v):
    """Draw ``n_v`` category indices for each row of per-item PMFs.

    The PMF array is shaped ``(n_items, n_categories)``; the result is shaped
    ``(n_v, n_items)``. Inverse-CDF sampling lets all items share one batched
    random draw instead of calling ``Generator.choice`` once per item.
    """
    pmfs = np.asarray(pmfs, dtype=float)
    if not np.all(np.isfinite(pmfs)):
        raise ValueError("probabilities contain NaN")
    if np.any(pmfs < 0) or not np.all(np.isclose(pmfs.sum(axis=1), 1, atol=1e-8)):
        raise ValueError("probabilities do not sum to 1")

    cumulative = np.cumsum(pmfs, axis=1)
    random_values = rng.random((n_v, pmfs.shape[0]))
    # Count CDF cutoffs at or below each draw to obtain its category index.
    indices = np.sum(random_values[:, :, None] >= cumulative[None, :, :], axis=2)
    # Guard against a final cumulative probability a few ulps below one.
    return np.minimum(indices, pmfs.shape[1] - 1)


def _sample_adjacent_two_choice(rng, quality, step=1, s_L=1):
    """Draw one ATC rating per quality entry, preserving the input array shape."""
    int_scale = (np.asarray(quality) - s_L) / step
    lower = np.floor(int_scale)
    upper_diff = np.ceil(int_scale) - int_scale
    # The upper neighbor is chosen with probability equal to the fractional part.
    is_upper = (rng.random(int_scale.shape) > upper_diff) & (int_scale != lower)
    return s_L + step * (lower + is_upper)


def _sample_maximum_variance_unimodal(rng, quality, n_v):
    """Return batched MVU ratings for each quality using the shared PMF sampler."""
    pdf_gen = MaxUnimodalPDF()
    indices = _sample_discrete_votes(rng, pdf_gen.pdf_many(quality), n_v)
    return np.arange(1, 6)[indices]


def mixed_behavior_votes(quality, vars, n_v, step=1, s_L=1, s_H=5, seed=None):
    """
    mixed_behavior_votes

    Generate votes according to a mixed behavior model.

    Parameters
    ----------
    quality : float, list, np.array
        Per-item quality values that the votes converge to.
    vars : float, list, np.array
        Target vote variance for each item.
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

    Notes
    -----
    Each vote uses BinoVotes with a quality- and target-variance-dependent
    probability. Remaining votes use ATC when the target variance is below the
    BinoVotes variance and MVU when it is above. Samples are batched by model.
    """
    if not (s_L == 1 and s_H == 5 and step == 1):
        raise ValueError(
            "Mixed behavior model currently only supports standard 1-5 rating scale."
        )
    rng, quality = sim_setup(quality, seed)
    # Get alpha values for the given quality and variance
    alpha_min, alpha_max, use_max = get_alpha(quality, vars)

    # Keep fractional mixture probabilities when quality inputs are integers.
    alpha = np.zeros_like(quality, dtype=float)
    alpha[use_max] = alpha_max[use_max]
    alpha[~use_max] = alpha_min[~use_max]
    behavior_probs = rng.random(size=(n_v, len(quality)))

    # Choose the model independently for each vote, with the alternate model
    # selected by whether that item's target variance is above the BinoVotes range.
    use_bino = behavior_probs < alpha
    use_alternate = ~use_bino
    use_atc = use_alternate & ~use_max[None, :]
    use_mvu = use_alternate & use_max[None, :]
    votes = np.zeros((n_v, len(quality)), dtype=int)

    rows, columns = np.nonzero(use_bino)
    if rows.size:
        n_bino = len(np.arange(s_L, s_H + step, step)) - 1
        p_bino = (quality[columns] - s_L) / (s_H - s_L)
        votes[rows, columns] = s_L + step * rng.binomial(n_bino, p_bino)

    rows, columns = np.nonzero(use_atc)
    if rows.size:
        votes[rows, columns] = _sample_adjacent_two_choice(
            rng, quality[columns], step=step, s_L=s_L
        )

    mvu_columns = np.flatnonzero(np.any(use_mvu, axis=0))
    if mvu_columns.size:
        # Generate one batched MVU matrix, then keep only its selected vote cells.
        samples = _sample_maximum_variance_unimodal(
            rng, quality[mvu_columns], n_v
        )
        rows, local_columns = np.nonzero(use_mvu[:, mvu_columns])
        votes[rows, mvu_columns[local_columns]] = samples[rows, local_columns]

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
    quality : float, list, np.array
        Per-item quality values that the votes converge to.
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
        A ``(n_v, len(quality))`` array of votes generated according to ATC.
    """
    rng, quality = sim_setup(quality, seed)
    # Broadcast all items across the vote axis and use the same kernel as mixed votes.
    vote_quality = np.broadcast_to(quality, (n_v, quality.size))
    return _sample_adjacent_two_choice(rng, vote_quality, step=step, s_L=s_L)


def maximum_variance_unimodal(quality, n_v, seed=None):
    """
    maximum_variance_unimodal

    Generate votes according to Maximum Variance Unimodal (MVU) voting model.

    Currently does not support ratings scales outside of the standard, integer 1-5
    scale.

    Parameters
    ----------
    quality : float, list, np.array
        Per-item quality values that the votes converge to.
    n_v : int
        Number of votes per file.
    seed : _type_, optional
        Seed for random number generation, by default None.

    Returns
    -------
    np.array
        A ``(n_v, len(quality))`` array of MVU ratings.
    """
    rng, quality = sim_setup(quality, seed)
    return _sample_maximum_variance_unimodal(rng, quality, n_v)


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
    quality : float, list, np.array
        Quality value or array of quality values.
    s_L : int, optional
        Lower limit of rating scale, by default 1
    s_H : int, optional
        Upper limit of rating scale, by default 5

    Returns
    -------
    float or np.array
        Variance from the maximum variance unimodal voting model, matching the
        scalar or array form of ``quality``.
    """
    pdf = MaxUnimodalPDF()
    if isinstance(quality, (float, int, np.number)):
        var = pdf.var(quality)
    else:
        var = pdf.var_many(quality)
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
    list of dict
        List containing dicts with binned MOS values, associated variances, number of
        files, and number of votes.
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

    var_df = stats_list
    return var_df

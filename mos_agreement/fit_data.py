import numpy as np


def fit_fourth_degree_poly(
    mos, vars, p=2, s_L=1, s_H=5, weights=None, tolerance=1e-6, ignore_violation=False
):
    """
    fit_fourth_degree_poly

    Fit a fourth-degree polynomial for vote variance as a function of MOS as
    (mos - 1) * (5 - mos) * (scale * (mos - 3)**2 + shift).
    This ensures that the variance function is 0 at the ends of the scale and that it
    is symmetric about the middle of the scale (3).

    This function will check if the fitted polynomial violates the minimum variance
    function. If it does, it will use a bisection method to find the smallest weight
    value that must be applied to the data near the edge of the scale that ensures the
    fitted polynomial does not violate the minimum variance function. This relies on
    the assumption that minimum variance violations are more likely to occur near the
    edges of the scale, so by applying more weight to real data there (where violations
    are not possible), we can ensure that the resulting polynomial does not violate the
    admissible variance region.

    If minimum variance violations occur with the input value for `weights`, the
    bisection will be used, and the `weights` input by the user will be overridden.
    The return value of `target_edge_frac` will be None if the original `weights` are
    respected.

    Parameters
    ----------
    mos : np.array
        MOS values
    vars : np.array
        Variance values associated with MOS.
    s_L : int, optional
        Lower bound of the MOS scale, by default 1
    s_H : int, optional
        Upper bound of the MOS scale, by default 5
    weights : np.array, optional
        Weights for the least squares fit, by default None
    tolerance : float, optional
        Convergence tolerance for the bisection method, by default 1e-6

    Returns
    -------
    scale : float
        Scale parameter of the fitted polynomial
    shift : float
        Shift parameter of the fitted polynomial
    target_edge_frac : float or None
        Value used to weight least squares. Value can be interpreted as the fraction of
        data we are forcing the edges to look like, e.g., a value of 0.5 says we are
        weighting least squares as if 50% of our data was on the edges.
    """
    # Initialize to equal weighting if needed.
    if weights is None:
        weights = np.ones_like(mos)
    elif len(weights) != len(mos):
        raise ValueError("weights must be the same length as mos")
    # Fit the polynomial
    scale, shift = fourth_degree_least_squares(
        mos, vars, p=p, s_L=s_L, s_H=s_H, weights=weights
    )
    # Check for violations
    violation = check_minimum_variance_violations(
        scale=scale,
        shift=shift,
        p=p,
    )
    # Return if everything works
    if not violation or ignore_violation:
        return scale, shift, None

    # If we get here, we have a violation. We will use bisection to find the smallest
    # value of target_edge_frac that ensures that the weighted least squares fit does
    # not produce variance below the minimum variance.
    convergence_criteria = False
    in_edges = (mos < 1.5) | (mos > 4.5)
    edge_frac = np.mean(in_edges)
    # We will bisect between lower and upper
    # We know that this violates minimum
    lower = edge_frac
    # This should never violate the minimum
    upper = 1
    while not convergence_criteria:
        # We want to find the smallest value of target_edge_frac that ensures that the
        # weighted least squares fit does not produce variance below the minimum
        # variance
        mid = (lower + upper) / 2
        # Get weights for least squares
        weight = least_squares_weights(in_edges=in_edges, target_edge_frac=mid)
        # Do weighted-least squares
        scale, shift = fourth_degree_least_squares(
            mos, vars, p=p, s_L=s_L, s_H=s_H, weights=weight
        )
        # Check for violations
        violation = check_minimum_variance_violations(
            scale=scale,
            shift=shift,
            p=p,
        )

        if violation:
            # Violation means we need to increase the weight on the edges, so we move
            # the lower bound up to mid
            lower = mid
        else:
            # No violation means we can decrease the weight on the edges
            upper = mid
        # Check our convergence criteria
        if np.abs(upper - lower) < tolerance:
            convergence_criteria = True
    # Set target_edge_frac as upper to be safe about avoiding violations
    target_edge_frac = upper
    return scale, shift, target_edge_frac


def least_squares_weights(in_edges, target_edge_frac):
    """
    least_squares_weights

    Get weights for weighted least squares based off of the data within the edges of
    the rating scale and the target fraction of data that will appear to be in the
    edges with our weighted-least-squares.

    Parameters
    ----------
    in_edges : np.array
        Boolean array indicating which data points are in the edges of the rating scale.
    target_edge_frac : float
        Target fraction of data that should appear to be in the edges with the weighted
        least squares.

    Returns
    -------
    np.array
        Weights for the least squares fit.
    """
    # Get fraction of data in the edges
    edge_frac = np.mean(in_edges)
    # Get the weight for the edges
    edge_weight = target_edge_frac / edge_frac

    # Get fraction and weight of data in the middle
    middle_frac = 1 - target_edge_frac
    middle_weight = middle_frac / (1 - edge_frac)

    # Make weight vector
    weight = np.zeros_like(in_edges, dtype=float)
    weight[in_edges] = edge_weight
    weight[~in_edges] = middle_weight
    return weight


def fourth_degree_least_squares(mos, vars, p=2, s_L=1, s_H=5, weights=None):
    """
    fourth_degree_least_squares

    Fit a fourth-degree polynomial of the form p(x) = (x-1)(5-x)*(w_0 + w_1*(x-3)^p) to
    the data such that p(mos) \approx vars.

    Parameters
    ----------
    mos : np.array
        MOS values
    vars : np.array
        Variance values associated with MOS.
    p : int, optional
        Exponent for the polynomial term (x-3)^p, by default 2
    s_L : int, optional
        Lower bound of the MOS scale, by default 1
    s_H : int, optional
        Upper bound of the MOS scale, by default 5
    weights : np.array, optional
        Weights for the least squares fit, by default None

    Returns
    -------
    scale : float
        Scale parameter of the fitted polynomial
    shift : float
        Shift parameter of the fitted polynomial
    """
    weights = np.diag(weights)
    midpoint = (s_L + s_H) / 2
    A = np.array(
        [
            (mos - s_L) * (s_H - mos) * np.abs(mos - midpoint) ** p,
            (mos - s_L) * (s_H - mos),
        ]
    ).transpose()
    A = np.matmul(weights, A)
    y = np.matmul(weights, vars)
    coefs, resid, _, _ = np.linalg.lstsq(A, y)

    # First coef value is scale
    scale = coefs[0]
    # Second coef value is scale * shift
    shift = coefs[1]
    return scale, shift


def check_minimum_variance_violations(scale, shift, p=2):
    """
    check_minimum_variance_violations

    Check if fitted variance function violates the minimum possible variance values
    across the quality scale.

    Parameters
    ----------
    mos : _type_
        _description_
    scale : _type_
        _description_
    shift : _type_
        _description_
    s_L : int, optional
        _description_, by default 1
    s_H : int, optional
        _description_, by default 5

    Returns
    -------
    _type_
        _description_
    """
    # solved (x - 1)(5 - x)(shift + scale * |x - 3|**p) < (x - 1)(2 - x)
    # with x -> 1+
    violation = 4 * shift + 2**p * 4 * scale < 1

    return violation

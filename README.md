# ITS-MOS-Agreement
This code corresponds to the paper J. Pieper and S. D. Voran, "Bounds on Agreement between Subjective and Objective Measurements," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2026.3712414.

Additional follow on work supporting mixed-behavior voting models has been added, corresponding the the paper J. Pieper and S. D. Voran "A Mixed-Behavior Vote Model for Multimedia Subjective Quality Votes, Means, and Variances."

## Abstract

Objective estimators of multimedia quality are typically judged by comparing estimates with subjective ``truth data,'' most often via Pearson correlation coefficient (PCC) or mean-squared error (MSE).
But subjective test results contain noise, so striving for a PCC of 1.0 or an MSE of 0.0 is neither realistic nor repeatable. 
Numerous efforts have been made to acknowledge and appropriately accommodate subjective test noise in objective-subjective comparisons, typically resulting in new analysis frameworks and figures-of-merit. 
We take a different approach.  By making only the most basic assumptions, we are able to derive bounds on the PCC and MSE that can be expected for a subjective test. 

Consistent with intuition, these bounds are functions of subjective vote variance.
When a subjective test includes vote variance information, the calculation of the bounds is straight-forward, and we say the resulting bounds are ``fully data-driven.''
We derive these data-driven PCC and MSE bounds for 18 subjective tests.
We also provide two options for calculating bounds in cases where vote variance information is not available.
One option is to use vote variance information from other subjective tests. 
The second option is to use a model for subjective votes, and we introduce a binomial-based model for this purpose.
We use both approaches to derive bounds and compare them with the data-driven bounds.
These results set expectations for the achievable PCC and MSE for any subjective test, even those where vote variance information is not available. 

## Installation

Create the conda environment and install the package with
```
conda env create -f environment.yaml
conda activate mos-agreement
pip install .
```

## Description 

This package provides utilities for estimating agreement bounds between subjective MOS (mean opinion score) measurements and objective quality scores. It includes voting models, distribution functions, and data fitting utilities.

### Core Modules

**Bounds** (`bounds.py`)
- **Purpose**: Utilities to estimate bounds on agreement (RMSE and Pearson correlation) between subjective MOS measurements and true quality.
- **Main functions**:
	- **`mos_data_bounds`**: Estimate RMSE and correlation bounds from MOS variance, average vote variance, and average number of votes per item.
	- **`mos_data_binovotes_bounds`**: Convenience wrapper that assumes vote variance from the BinoVotes model and returns RMSE/correlation bounds using MOS stats.
	- **`mos_data_binovotes_average_vote_var`**: Compute the average vote variance implied by the BinoVotes model given MOS mean/variance and vote count.
	- **`quality_distribution_bounds`**: Compute RMSE and correlation bounds when the true quality variance and expected vote variance are known (useful for simulations).
	- **`quality_distribution_binovotes_bounds`**: Same as above but using the BinoVotes model to compute the expected vote variance from a quality mean.

**Simulations** (`sim_votes.py`)
- **Purpose**: Generate synthetic votes according to various voting behavior models.
- **Main functions**:
	- **`binovotes`**: Simulate individual votes according to the BinoVotes model (binomial draws mapped to the rating scale). Returns a vote matrix of shape `(n_votes, n_items)`.
	- **`binomos`**: Wrapper around `binovotes` that optionally returns MOS values (mean across votes) instead of the full vote matrix.
	- **`mixed_behavior_votes`**: Generate votes according to a mixed behavior model that combines BinoVotes, Adjacent Two-Choice (ATC), and Maximum Variance Unimodal (MVU) models.
	- **`adjacent_two_choice`**: Simulate votes according to the ATC voting model (minimum variance model).
	- **`maximum_variance_unimodal`**: Simulate votes according to the MVU voting model (maximum variance model).
	- **`binovotes_variance`**, **`minimum_vote_variance`**, **`maximum_unimodal_vote_variance`**: Variance functions for each voting model.
	- **`effective_votes`**: Compute the effective number of votes per file when generating synthetic MOS values.

**Distributions** (`distributions.py`)
- **Purpose**: Probability distribution classes for quality and voting behavior, used primarily for visualization and analysis in the paper.
- **Classes**:
	- **`UniformPDF`**: Wrapper for uniform distribution.
	- **`TriangularPDF`**: Wrapper for triangular distribution.
	- **`BetaPDF`**: Wrapper for beta distribution.
	- **`MaxUnimodalPDF`**: Maximum variance unimodal discrete distribution.
	- **`MinVariancePDF`**: Minimum variance discrete distribution (Adjacent Two-Choice model).
	- **`BinoVotes`**: BinoVotes distribution class for theoretical analysis.

**Fitting** (`fit_data.py`)
- **Purpose**: Fit observed vote variance data to parametric variance models with constraints.
- **Main functions**:
	- **`fit_fourth_degree_poly`**: Fit a fourth-degree polynomial for vote variance as a function of MOS with automatic constraint enforcement to ensure the fit respects the minimum possible variance boundary. The fourth-degree polynomial is of the form $(x - 1)(5 - x)(w_0 + w_1(x-3)^2)$.
	- **`fourth_degree_least_squares`**: Perform weighted least-squares fitting of the variance polynomial.
	- **`check_minimum_variance_violations`**: Verify that a fitted variance function does not violate minimum variance constraints.
	- **`least_squares_weights`**: Compute weights for weighted least-squares fitting.

### Usage Examples

- **Estimate bounds from MOS statistics**:
	In this situation one has access to the MOS mean, MOS variance, the average observed vote variance, and the number of votes per file. Each MOS value has a variance associated with it; the average observed vote variance is the average value of those variances. Many datasets do not provide this information.

    ```python
    import mos_agreement as ma
    rmse, corr = ma.mos_data_bounds(mos_var=0.8, average_vote_var=1, n_v=10)
    ```

- **Estimate bounds from MOS statistics without vote variance information**:
	In this situation, one has access only to the mean MOS value and the MOS variance. The BinoVotes model is used to approximate the average vote variance.

    ```python
    rmse, corr = ma.mos_data_binovotes_bounds(mos_mean=3.2, mos_var=0.8, n_v=10)
    ```

- **Estimate bounds from quality distribution and BinoVotes voting model**:
	Here we compute bounds directly from a true quality distribution and assume the BinoVotes voting model.

    ```python
    rmse, corr = ma.quality_distribution_binovotes_bounds(quality_mean=3.2, quality_var=0.6, n_v=10)
    ```

- **Simulate votes using BinoVotes and get MOS**:
    ```python
    import numpy as np
    quality = np.array([3.0, 4.2])
    votes = ma.binovotes(quality, n_v=10)
    mos = np.mean(votes, axis=0)
    ```

- **Generate votes using mixed-behavior model**:
    ```python
    quality = np.array([3.5, 4.2])
	target_variance = np.array([0.6, 0.5])
    votes = ma.mixed_behavior_votes(quality=quality, vars=target_variance, n_v=8)
	mos = np.mean(votes, axis=0)
    ```

See each module's docstrings for full details and additional parameters.

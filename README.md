# ITS-MOS-Agreement
This code corresponds to the paper "Bounds on Agreement between Subjective and Objective Measurements" by Jaden Pieper and Stephen D. Voran.

## Abstract

Objective estimators of multimedia quality are typically judged by comparing estimates with subjective "truth data," most often via Pearson correlation coefficient (PCC) or mean-squared error (MSE).
But subjective test results contain noise, so striving for a PCC of 1.0 or an MSE of 0.0 is neither realistic nor repeatable. 
Numerous efforts have been made to acknowledge and appropriately accommodate subjective test noise in objective-subjective comparisons, typically resulting in new analysis frameworks and figures-of-merit. 
We take a different approach. 
By making only the most basic assumptions, we are able to derive bounds on PCC and MSE that can be expected for a subjective test. 
Consistent with intuition, these bounds are functions of subjective vote variance.

We also introduce a binomial-based model for subjective votes (BinoVotes) that naturally leads to a MOS model (BinoMOS) with multiple unique desirable properties. 
BinoMOS reproduces the discrete nature of MOS values and its dependence on the number of votes per file. 
When subjective test results do not include vote variance information, these models provide that information for use in the PCC and MSE bounds. 
We compare this modeling with data from 18 subjective tests. 
The modeling yields PCC and MSE bounds that agree very well with those found from the data directly. 
These results allow one to set expectations for the PCC and MSE that might be achieved for any subjective test, even those where vote variance information is not available. 

## Installation

Create the conda environment and install the package with
```
conda env create -f environment.yml
conda activate mos-agreement
pip install .
```
## Description 

**Model**
- **Location**: [mos_agreement/model.py](mos_agreement/model.py)
- **Purpose**: Utilities to estimate bounds on agreement (RMSE and Pearson correlation)
	between subjective MOS (mean opinion score) measurements and true quality, plus a
	simple binomial voting model (BinoVotes) and its MOS wrapper (BinoMOS).
- **Main functions:**
	- **`mos_data_bounds`**: Estimate RMSE and correlation bounds from MOS variance,
		average vote variance, and average number of votes per item.
	- **`mos_data_binovotes_bounds`**: Convenience wrapper that assumes vote variance
		from the BinoVotes model and returns RMSE/correlation bounds using MOS stats.
	- **`mos_data_binovotes_average_vote_var`**: Compute the average vote variance
		implied by the BinoVotes model given MOS mean/variance and vote count.
	- **`quality_distribution_bounds`**: Compute RMSE and correlation bounds when the
		true quality variance and expected vote variance are known (useful for 
        simulations).
	- **`quality_distribution_binovotes_bounds`**: Same as above but using the
		BinoVotes model to compute the expected vote variance from a quality mean.
	- **`binovotes`**: Simulate individual votes according to the BinoVotes model
		(binomial draws mapped to the rating scale). Returns a vote matrix of shape
		`(n_votes, n_items)`.
	- **`binomos`**: Wrapper around `binovotes` that optionally returns MOS values
		(mean across votes) instead of the full vote matrix.
- **Usage notes**: The functions are lightweight NumPy-based utilities; Example quick 
        calls:

	- Estimate bounds from MOS statistics:
        
        In this situation one has access to the MOS mean, MOS variance, the average
        observed vote variance, and the number of votes per file. Each MOS value has a
        variance associated with it; the average observed vote variance is the average
        value of those variances. Many datasets do not provide this information.

        ```python
		import mos_agreement as ma
        rmse, corr = ma.mos_data_bounds(mos_var=0.8, average_vote_var=1, n_v=10)
		rmse, corr = ma.mos_data_binovotes_bounds(mos_mean=3.2, mos_var=0.8, n_v=10)
		```
    
    - Estimate bounds from MOS statistics without vote variance information

        In this situation, one has access only to the mean MOS value and the MOS 
        variance. The BinoVotes model is used to approximate the average vote variance.

		```python
		rmse, corr = ma.mos_data_binovotes_bounds(mos_mean=3.2, mos_var=0.8, n_v=10)
		```
    
    - Estimate bounds from quality distribution and arbitrary voting model

        Here we compute bounds directly from a true quality distribution 
        and an expected vote variance.

        ```python
        rmse, corr = ma.quality_distribution_bounds(quality_var=0.6, expected_vote_var=0.9, n_v=10)
        ```
    
    - Estimate bounds from quality distribution and BinoVotes voting model

        Here we compute bounds directly from a true quality distribution and assume the 
        BinoVotes voting model.

        ```python
        rmse, corr = ma.quality_distribution_binovotes_bounds(quality_mean=3.2, quality_var=0.6, n_v=10)
        ```

	- Simulate votes and get MOS:

		```python
		quality = np.array([3.0, 4.2])
		votes = ma.binovotes(quality, n_v=10)
		mos = np.mean(votes, 0)
		```

See [mos_agreement/model.py](mos_agreement/model.py) for full docstrings and details.
import numpy as np
import pytest

import mos_agreement as ma


def test_continuous_distribution_support_and_moments():
    uniform = ma.UniformPDF(a=1, b=5)
    triangular = ma.TriangularPDF(a=1, b=5, m=3)
    beta = ma.BetaPDF(alpha=2, beta=2, a=1, b=5)

    assert uniform.pdf(0) == 0
    assert uniform.mean() == 3
    assert uniform.var() == 4 / 3
    assert triangular.pdf(0) == 0
    assert triangular.mean() == 3
    assert beta.mean() == 3
    assert np.isclose(beta.var(), 0.8)


def test_max_unimodal_pdf_preserves_mean_and_probability():
    pdf = ma.MaxUnimodalPDF()

    for quality in (1, 1.5, 2.75, 4.9, 5):
        probabilities = np.asarray(pdf.pmf(quality))

        assert np.all(probabilities >= 0)
        assert np.isclose(probabilities.sum(), 1)
        assert np.isclose(pdf.mean(quality), quality)


def test_max_unimodal_pdf_vectorized_methods_match_scalar_reference():
    """Compare batched PMFs and variances with the scalar reference over the scale."""
    pdf = ma.MaxUnimodalPDF()
    quality = np.concatenate(([0], np.arange(1, 5, 0.01), [5, 6, np.nan]))

    vectorized_pmf = pdf.pdf_many(quality)
    vectorized_var = pdf.var_many(quality)
    scalar_pmf = np.array([pdf.pdf(q) for q in quality])
    scalar_var = np.array([pdf.var(q) for q in quality])

    np.testing.assert_allclose(vectorized_pmf, scalar_pmf, equal_nan=True)
    np.testing.assert_allclose(vectorized_var, scalar_var, equal_nan=True)


def test_max_unimodal_pdf_outside_support_and_nan():
    pdf = ma.MaxUnimodalPDF()

    assert pdf.pmf(0) == [0, 0, 0, 0, 0]
    assert pdf.pmf(6) == [0, 0, 0, 0, 0]
    assert np.isnan(pdf.pmf(np.nan)).all()


def test_min_variance_pdf_handles_both_scale_endpoints():
    pdf = ma.MinVariancePDF()

    assert pdf.pmf(1) == [1, 0, 0, 0, 0]
    assert pdf.pmf(5) == [0, 0, 0, 0, 1]


def test_binovotes_rejects_values_outside_its_sample_space():
    pdf = ma.BinoVotes(
        v_L=1,
        v_H=5,
        n_s=5,
        n_v=2,
        quality_pdf=ma.UniformPDF(a=1, b=5),
    )

    with pytest.raises(ValueError, match="not in sample space"):
        pdf.pmf(1.05)


def test_binovotes_pmf_normalizes_over_its_sample_space():
    pdf = ma.BinoVotes(
        v_L=1,
        v_H=5,
        n_s=5,
        n_v=2,
        quality_pdf=ma.UniformPDF(a=1, b=5),
    )
    sample_space = np.linspace(1, 5, pdf.len + 1)

    probabilities = [pdf.pmf(value) for value in sample_space]

    assert np.all(np.asarray(probabilities) >= 0)
    assert np.isclose(sum(probabilities), 1)

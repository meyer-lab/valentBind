"""Tests that invalid inputs are rejected by the shared input validation."""

import pytest

from ..model import commonChecks, polyc, polyfc


def test_commonChecks_rejects_shape_mismatch() -> None:
    """Rtot size must match the number of columns (receptors) in Kav."""
    with pytest.raises(AssertionError):
        commonChecks(1.0, [1.0, 2.0], 1.0, [[1.0, 2.0, 3.0]], [1.0])


def test_commonChecks_rejects_non_2d_Kav() -> None:
    """Kav must be a 2D matrix of ligands x receptors."""
    with pytest.raises(AssertionError):
        commonChecks(1.0, [1.0], 1.0, [1.0, 2.0], [1.0])


def test_commonChecks_normalizes_Ctheta() -> None:
    """Ctheta is renormalized to sum to one regardless of input scale."""
    _, _, _, _, Ctheta = commonChecks(1.0, [1.0], 1.0, [[1.0]], [2.0, 2.0])
    assert pytest.approx(float(Ctheta.sum())) == 1.0
    assert pytest.approx(float(Ctheta[0])) == 0.5


def test_polyfc_rejects_LigC_Kav_mismatch() -> None:
    """LigC must have one entry per row (ligand) of Kav."""
    with pytest.raises(AssertionError):
        polyfc(1e-9, 1e-12, 4, [1e5], [1.0, 1.0], [[1e6]])


def test_polyc_rejects_non_2d_Cplx() -> None:
    """Cplx must be a 2D matrix of complexes x monomer ligands."""
    with pytest.raises(AssertionError):
        polyc(1e-9, 1e-12, [1e5], [1, 0], [1.0], [[1e6]])


def test_polyc_rejects_Cplx_Kav_mismatch() -> None:
    """The number of monomer ligand columns in Cplx must match Kav's rows."""
    with pytest.raises(AssertionError):
        polyc(1e-9, 1e-12, [1e5], [[1, 0, 0]], [1.0], [[1e6], [1e5]])


def test_polyc_rejects_Cplx_Ctheta_mismatch() -> None:
    """Cplx must have one row per entry of Ctheta."""
    with pytest.raises(AssertionError):
        polyc(1e-9, 1e-12, [1e5], [[1, 0], [0, 1]], [1.0], [[1e6], [1e5]])

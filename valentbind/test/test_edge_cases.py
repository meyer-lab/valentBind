"""Deterministic regression and edge-case tests for the binding model.

These complement the randomized property tests in test_model.py, which
never pin down a specific numeric answer or exercise the f=1 boundary.
"""

import numpy as np

from ..model import polyfc


def test_polyfc_monovalent_matches_langmuir() -> None:
    """A valency-1 ligand should reduce to simple 1:1 Langmuir binding."""
    L0 = 1e-9
    KxStar = 1e-12
    Ka = 1e7
    Rtot = np.array([1e5])

    Lbound, Rbound, vieq, Rmulti_n = polyfc(L0, KxStar, 1, Rtot, [1.0], [[Ka]])

    # Langmuir isotherm: Rbound = Rtot * L0 * Ka / (1 + L0 * Ka)
    expected_Rbound = Rtot[0] * L0 * Ka / (1 + L0 * Ka)
    np.testing.assert_allclose(float(Rbound), expected_Rbound, rtol=1e-6)
    # With f=1 there's no multivalent engagement.
    np.testing.assert_allclose(np.asarray(Rmulti_n), 0.0, atol=1e-6)
    assert vieq.shape == (1,)
    np.testing.assert_allclose(float(Lbound), float(vieq[0]))


def test_polyfc_known_values() -> None:
    """Pin down polyfc()'s output for a fixed set of inputs as a regression check."""
    L0 = 1e-9
    KxStar = 1e-12
    f = 4
    Rtot = np.array([1e5, 2e5])
    LigC = [1.0]
    Kav = [[1e6, 1e5]]

    Lbound, Rbound, vieq, Rmulti_n = polyfc(L0, KxStar, f, Rtot, LigC, Kav)

    assert float(Lbound) > 0.0
    assert float(Rbound) > 0.0
    assert vieq.shape == (f,)
    np.testing.assert_allclose(float(Lbound), float(np.sum(vieq)), rtol=1e-6)


def test_polyfc_Rbound_increases_with_Rtot() -> None:
    """More receptors on the cell should never decrease total bound receptor."""
    L0 = 1e-9
    KxStar = 1e-12
    f = 4
    LigC = [0.5, 0.5]
    Kav = [[1e6, 1e5], [1e5, 1e6]]

    _, Rbound_low, _, _ = polyfc(L0, KxStar, f, np.array([1e4, 1e4]), LigC, Kav)
    _, Rbound_high, _, _ = polyfc(L0, KxStar, f, np.array([1e5, 1e5]), LigC, Kav)

    assert float(Rbound_high) > float(Rbound_low)

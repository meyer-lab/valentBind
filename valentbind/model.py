"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
import jax.numpy as jnp
from jaxopt import ScipyRootFinding, Bisection, FixedPointIteration
from jax.config import config
from scipy.special import binom

config.update("jax_enable_x64", True)


def Req_func(Phisum: float, Rtot: np.ndarray, L0: float, KxStar: float, f, A: np.ndarray):
    """ Mass balance. Transformation to account for bounds. """
    Req = Rtot / (1.0 + L0 * f * A * (1 + Phisum) ** (f - 1))
    return Phisum - jnp.dot(A * KxStar, Req.T)


def Req_func2(Req, Rtot, L0: float, KxStar, Cplx, Ctheta, Kav):
    Psi = Req * Kav * KxStar
    Psirs = jnp.sum(Psi, axis=1).reshape(-1, 1) + 1
    Psinorm = (Psi / Psirs)

    Rbound = L0 / KxStar * jnp.sum(Ctheta.reshape(-1, 1) * jnp.dot(Cplx, Psinorm) * jnp.exp(jnp.dot(Cplx, jnp.log1p(Psirs - 1))), axis=0)
    return Req + Rbound - Rtot


def commonChecks(L0: float, Rtot: np.ndarray, KxStar: float, Kav: np.ndarray, Ctheta: np.ndarray):
    """ Check that the inputs are sane. """
    Kav = jnp.array(Kav, dtype=float)
    Rtot = jnp.array(Rtot, dtype=float)
    Ctheta = jnp.array(Ctheta, dtype=float)
    assert Rtot.ndim <= 1
    assert Rtot.size == Kav.shape[1]
    assert Kav.ndim == 2
    assert Ctheta.ndim <= 1
    Ctheta = Ctheta / jnp.sum(Ctheta)
    return L0, Rtot, KxStar, Kav, Ctheta


def polyfc(L0: float, KxStar: float, f, Rtot: np.ndarray, LigC: np.ndarray, Kav: np.ndarray):
    """
    The main function. Generate all info for heterogenenous binding case
    L0: concentration of ligand complexes.
    KxStar: detailed balance-corrected Kx.
    f: valency
    Rtot: numbers of each receptor appearing on the cell.
    LigC: the composition of the mixture used.
    Kav: a matrix of Ka values. row = ligands, col = receptors
    """
    # Data consistency check
    L0, Rtot, KxStar, Kav, LigC = commonChecks(L0, Rtot, KxStar, Kav, LigC)
    assert LigC.size == Kav.shape[0]

    A = jnp.dot(LigC.T, Kav)

    # Find Phisum by fixed point iteration
    lsq = FixedPointIteration(fixed_point_fun=Req_func, tol=1e-12)
    lsq = lsq.run(jnp.zeros(1), Rtot, L0, KxStar, f, A)
    if lsq.state.error > 1e-9:
        lsq = ScipyRootFinding(method="lm", optimality_fun=Req_func, tol=1e-12)
        lsq = lsq.run(jnp.zeros(1), Rtot, L0, KxStar, f, A)
        assert lsq.state.success, "Failure in rootfinding. " + str(lsq)

    Phisum = lsq.params[0]

    Lbound = L0 / KxStar * ((1 + Phisum) ** f - 1)
    Rbound = L0 / KxStar * f * Phisum * (1 + Phisum) ** (f - 1)
    vieq = L0 / KxStar * binom(f, np.arange(1, f + 1)) * jnp.power(Phisum, np.arange(1, f + 1))
    return Lbound, Rbound, vieq


def polyc(L0: float, KxStar: float, Rtot: np.ndarray, Cplx: np.ndarray, Ctheta: np.ndarray, Kav: np.ndarray):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Ctheta: the composition of complexes
    :param Kav: Ka for monomer ligand to receptors
    :return:
        Lbound: a list of Lbound of each complex
        Rbound: a list of Rbound of each kind of receptor
    """
    # Consistency check
    L0, Rtot, KxStar, Kav, Ctheta = commonChecks(L0, Rtot, KxStar, Kav, Ctheta)
    Cplx = np.array(Cplx)
    assert Cplx.ndim == 2
    assert Kav.shape[0] == Cplx.shape[1]
    assert Cplx.shape[0] == Ctheta.size

    # Get an estimate of Req from polyfc
    LigC = Ctheta @ Cplx # Convert to LigC
    f_fc = np.amax(np.sum(Cplx, axis=1)) # Find the maximum valency among complexes
    Rbnd_fc = polyfc(L0, KxStar, f_fc, Rtot, LigC, Kav)[1]
    Req_fc = Rtot - Rbnd_fc
    Req_fc = np.clip(Req_fc, 0.0, Rtot)
    assert np.all(Req_fc >= 0.0)

    # Solve Req
    lsq = ScipyRootFinding(method="lm", optimality_fun=Req_func2, tol=1e-10)
    lsq = lsq.run(Req_fc, Rtot, L0, KxStar, Cplx, Ctheta, Kav)
    assert lsq.state.success, "Failure in rootfinding. " + str(lsq)
    Req = lsq.params
    assert np.all(np.isfinite(Req))

    # Calculate the results
    Psi = np.ones((Kav.shape[0], Kav.shape[1] + 1))
    Psi[:, : Kav.shape[1]] *= Req.T * Kav * KxStar
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Lbound = L0 / KxStar * Ctheta * np.expm1(np.dot(Cplx, np.log(Psirs))).flatten()
    Rbound = L0 / KxStar * Ctheta.reshape(-1, 1) * np.dot(Cplx, Psinorm) * np.exp(np.dot(Cplx, np.log(Psirs)))
    with np.errstate(divide='ignore'):
        Lfbnd = L0 / KxStar * Ctheta * np.exp(np.dot(Cplx, np.log(Psirs - 1.0))).flatten()
    assert len(Lbound) == len(Ctheta)
    assert Rbound.shape[0] == len(Ctheta)
    assert Rbound.shape[1] == len(Rtot)
    return Lbound, Rbound, Lfbnd

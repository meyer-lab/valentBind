"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import ScipyRootFinding
from scipy.special import binom

jax.config.update("jax_enable_x64", True)
jax_array = np.ndarray | jnp.ndarray


def Req_polyfc(
    Phisum: float, Rtot: jax_array, L0: float, KxStar: float, f, A: jax_array
):
    """Mass balance. Transformation to account for bounds."""
    Req = Rtot / (1.0 + L0 * f * A * (1 + Phisum) ** (f - 1))
    return Phisum - jnp.dot(A * KxStar, Req.T)


def Req_polyc(
    Req: jax_array,
    Rtot: jax_array,
    L0_Kx_Ct: jax_array,
    KxStar: float,
    Cplx: jax_array,
    Kav: jax_array,
) -> jax_array:
    Psi = Req * Kav * KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[:, jnp.newaxis]

    Rbound = L0_Kx_Ct * jnp.exp(Cplx @ jnp.log(Psirs)) @ Cplx @ Psinorm
    return Req + Rbound - Rtot


def commonChecks(
    L0: float, Rtot: jax_array, KxStar: float, Kav: jax_array, Ctheta: jax_array
):
    """Check that the inputs are sane."""
    Kav = jnp.array(Kav, dtype=float)
    Rtot = jnp.array(Rtot, dtype=float)
    Ctheta = jnp.array(Ctheta, dtype=float)
    assert Rtot.ndim <= 1
    assert Rtot.size == Kav.shape[1]
    assert Kav.ndim == 2
    assert Ctheta.ndim <= 1
    Ctheta = Ctheta / jnp.sum(Ctheta)
    return L0, Rtot, KxStar, Kav, Ctheta


def polyfc(
    L0: float,
    KxStar: float,
    f: int | float,
    Rtot: jax_array,
    LigC: jax_array,
    Kav: jax_array,
):
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
    lsq = ScipyRootFinding(method="lm", optimality_fun=Req_polyfc, tol=1e-12)
    lsq = lsq.run(jnp.zeros(1), Rtot, L0, KxStar, f, A)
    assert lsq.state.success, "Failure in rootfinding. " + str(lsq)
    Phisum = lsq.params[0]

    Lbound = L0 / KxStar * ((1 + Phisum) ** f - 1)
    Rbound = L0 / KxStar * f * Phisum * (1 + Phisum) ** (f - 1)
    vieq = (
        L0
        / KxStar
        * binom(f, np.arange(1, f + 1))
        * jnp.power(Phisum, np.arange(1, f + 1))
    )

    Req_n = Rtot / (1.0 + L0 * f * A * (1 + Phisum) ** (f - 1))
    Phi_n = A * KxStar * Req_n
    assert jnp.isclose(Phisum, jnp.sum(Phi_n))
    Rmulti_n = L0 * f / KxStar * Phi_n * ((1 + Phisum) ** (f - 1) - 1)
    return Lbound, Rbound, vieq, Rmulti_n


def Req_solve(func, Rtot, *args):
    """Run least squares regression to calculate the Req vector."""
    lsq = ScipyRootFinding(method="lm", optimality_fun=func, tol=1e-12)
    lsq = lsq.run(jnp.zeros_like(Rtot), Rtot, *args)
    assert lsq.state.success, "Failure in rootfinding. " + str(lsq)
    return lsq.params


def polyc(
    L0: float,
    KxStar: float,
    Rtot: jax_array,
    Cplx: jax_array,
    Ctheta: jax_array,
    Kav: jax_array,
):
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
    Cplx = jnp.array(Cplx)
    assert Cplx.ndim == 2
    assert Kav.shape[0] == Cplx.shape[1]
    assert Cplx.shape[0] == Ctheta.size

    L0_Kx_Ct = L0 / KxStar * Ctheta

    # Solve Req
    Req = Req_solve(Req_polyc, Rtot, L0_Kx_Ct, KxStar, Cplx, Kav)

    # Calculate the results
    Psi = Req * Kav * KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[:, jnp.newaxis]

    Lbound = L0_Kx_Ct * jnp.expm1(jnp.dot(Cplx, jnp.log(Psirs))).flatten()
    Rbound = (
        L0_Kx_Ct[:, jnp.newaxis]
        * jnp.dot(Cplx, Psinorm)
        * jnp.exp(Cplx @ jnp.log(Psirs))[:, jnp.newaxis]
    )
    with np.errstate(divide="ignore"):
        Lfbnd = L0_Kx_Ct * jnp.exp(jnp.dot(Cplx, jnp.log(Psirs - 1.0))).flatten()

    assert len(Lbound) == len(Ctheta)
    assert Rbound.shape[0] == len(Ctheta)
    assert Rbound.shape[1] == len(Rtot)
    return Lbound, Rbound, Lfbnd

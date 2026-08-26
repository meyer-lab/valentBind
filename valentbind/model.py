"""
Implementation of a simple multivalent binding model.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as opt
from scipy.special import binom

jax.config.update("jax_enable_x64", True)


def Req_polyfc(
    Phisum: jax.Array,
    args: tuple[jax.Array, float, float, int | float, jax.Array],
) -> jax.Array:
    """
    Mass balance residual for the homogeneous-ligand (polyfc) binding model.

    This is the root-finding target passed to the solver in :func:`polyfc`;
    it is zero when ``Phisum`` is the free-receptor-weighted binding
    potential that is consistent with the mass balance for the total
    receptor and free ligand concentrations.

    :param Phisum: Current guess for the binding potential (a length-1
        array, since the model reduces to a single scalar unknown).
    :param args: Tuple of ``(Rtot, L0, KxStar, f, A)`` where ``Rtot`` is the
        total receptor abundance per receptor type, ``L0`` is the total
        ligand complex concentration, ``KxStar`` is the detailed-balance
        corrected cross-linking constant, ``f`` is the ligand valency, and
        ``A`` is the ligand-composition-weighted affinity vector.
    :return: The residual ``Phisum - sum(A * KxStar * Req)``, which the
        solver drives to zero.
    """
    Rtot, L0, KxStar, f, A = args
    Req = Rtot / (1.0 + L0 * f * A * (1 + Phisum) ** (f - 1))
    return Phisum - jnp.dot(A * KxStar, Req.T)


def Req_polyc(
    Req: jax.Array,
    args: tuple[jax.Array, float, float, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    """
    Mass balance residual for the heterogeneous-complex (polyc) binding model.

    This is the root-finding target passed to the solver in :func:`polyc`;
    it is zero when ``Req`` is the vector of free-receptor abundances
    consistent with the mass balance for every receptor type.

    :param Req: Current guess for the free receptor abundance per receptor
        type.
    :param args: Tuple of ``(Rtot, L0, KxStar, Cplx, Ctheta, Kav)`` where
        ``Rtot`` is the total receptor abundance per receptor type, ``L0``
        is the total ligand complex concentration, ``KxStar`` is the
        detailed-balance corrected cross-linking constant, ``Cplx`` is the
        monomer composition of each ligand complex, ``Ctheta`` is the
        relative abundance of each complex, and ``Kav`` is the monomer
        ligand/receptor affinity matrix.
    :return: The residual ``Req + Rbound - Rtot``, which the solver drives
        to zero.
    """
    Rtot, L0, KxStar, Cplx, Ctheta, Kav = args
    Psi = Req * Kav * KxStar
    Psirs = Psi.sum(axis=1).reshape(-1, 1) + 1
    Psinorm = Psi / Psirs

    Rbound = (
        L0
        / KxStar
        * jnp.sum(
            Ctheta.reshape(-1, 1)
            * jnp.dot(Cplx, Psinorm)
            * jnp.exp(jnp.dot(Cplx, jnp.log1p(Psirs - 1))),
            axis=0,
        )
    )
    return Req + Rbound - Rtot


def commonChecks(
    L0: float,
    Rtot: npt.ArrayLike,
    KxStar: float,
    Kav: npt.ArrayLike,
    Ctheta: npt.ArrayLike,
) -> tuple[float, jax.Array, float, jax.Array, jax.Array]:
    """
    Validate and normalize the inputs shared by :func:`polyfc` and :func:`polyc`.

    Converts ``Rtot``, ``Kav``, and ``Ctheta`` to ``jax`` arrays, checks
    that their shapes are mutually consistent, and normalizes ``Ctheta`` so
    it sums to one.

    :param L0: Concentration of ligand complexes.
    :param Rtot: Total abundance of each receptor type on the cell.
    :param KxStar: Detailed-balance corrected cross-linking constant.
    :param Kav: Matrix of monomer ligand/receptor affinities (rows are
        ligands, columns are receptors).
    :param Ctheta: Relative abundance of each ligand or complex; renormalized
        to sum to one.
    :raises AssertionError: If the shapes of ``Rtot``, ``Kav``, or
        ``Ctheta`` are inconsistent with one another.
    :return: The tuple ``(L0, Rtot, KxStar, Kav, Ctheta)`` with ``Rtot``,
        ``Kav``, and ``Ctheta`` converted to arrays and ``Ctheta``
        normalized.
    """
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
    Rtot: npt.ArrayLike,
    LigC: npt.ArrayLike,
    Kav: npt.ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Solve the multivalent binding model for a single homogeneous ligand complex.

    Computes bound ligand, bound receptor, and per-valency binding
    statistics for a population of identical ligand complexes of valency
    ``f``, each assembled from a fixed mixture of monomer ligands
    (``LigC``), binding a set of receptors (``Rtot``) with affinities
    ``Kav``.

    :param L0: Concentration of ligand complexes.
    :param KxStar: Detailed-balance corrected cross-linking constant.
    :param f: Valency of the ligand complex.
    :param Rtot: Total abundance of each receptor type on the cell.
    :param LigC: Relative composition of monomer ligands within the
        complex; renormalized to sum to one.
    :param Kav: Matrix of monomer ligand/receptor affinities (rows are
        ligands, columns are receptors).
    :return: A tuple ``(Lbound, Rbound, vieq, Rmulti_n)`` where ``Lbound``
        is the total concentration of bound ligand complex, ``Rbound`` is
        the total abundance of bound receptor (summed across receptor
        types), ``vieq`` is the concentration of complex bound by exactly
        ``i`` receptors for each valency ``i`` from 1 to ``f``, and
        ``Rmulti_n`` is the abundance of each receptor type engaged in
        multivalent (more than one ligand-receptor bond) binding.
    """
    # Data consistency check
    L0, Rtot, KxStar, Kav, LigC = commonChecks(L0, Rtot, KxStar, Kav, LigC)
    assert LigC.size == Kav.shape[0]

    A = jnp.dot(LigC.T, Kav)

    # Find Phisum by fixed point iteration
    solver = opt.LevenbergMarquardt(rtol=1e-9, atol=1e-9)
    result = opt.root_find(
        Req_polyfc, solver, y0=jnp.zeros(1), args=(Rtot, L0, KxStar, f, A), throw=True
    )
    Phisum = result.value[0]

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


def Req_solve(func: Callable[..., jax.Array], Rtot: jax.Array, *args) -> jax.Array:
    """
    Run Levenberg-Marquardt root finding to calculate the free receptor vector.

    :param func: Residual function to find the root of; called as
        ``func(Req, (Rtot, *args))``.
    :param Rtot: Total abundance of each receptor type on the cell; also
        used as the shape template for the initial guess (zeros).
    :param args: Additional positional arguments forwarded to ``func``
        after ``Rtot``.
    :return: The free receptor abundance vector ``Req`` that zeroes
        ``func``.
    """
    solver = opt.LevenbergMarquardt(rtol=1e-9, atol=1e-9)
    result = opt.root_find(
        func, solver, y0=jnp.zeros_like(Rtot), args=(Rtot, *args), throw=True
    )
    return result.value


def polyc(
    L0: float,
    KxStar: float,
    Rtot: npt.ArrayLike,
    Cplx: npt.ArrayLike,
    Ctheta: npt.ArrayLike,
    Kav: npt.ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Solve the multivalent binding model for a mixture of heterogeneous ligand complexes.

    Computes bound ligand, bound receptor, and free ligand statistics for
    a population of ligand complexes that can differ in their monomer
    composition (``Cplx``), binding a set of receptors (``Rtot``) with
    affinities ``Kav``.

    :param L0: Concentration of ligand complexes.
    :param KxStar: Detailed-balance corrected cross-linking constant.
    :param Rtot: Total abundance of each receptor type on the cell.
    :param Cplx: Monomer ligand composition of each complex; rows are
        complexes, columns are monomer ligand types.
    :param Ctheta: Relative abundance of each complex; renormalized to sum
        to one.
    :param Kav: Matrix of monomer ligand/receptor affinities (rows are
        ligands, columns are receptors).
    :raises AssertionError: If the shapes of ``Cplx``, ``Kav``, or
        ``Ctheta`` are inconsistent with one another.
    :return: A tuple ``(Lbound, Rbound, Lfbnd)`` where ``Lbound`` is the
        concentration of bound ligand complex for each complex type,
        ``Rbound`` is the abundance of bound receptor for each complex type
        and receptor type, and ``Lfbnd`` is the concentration of complex
        bound by exactly one receptor for each complex type.
    """
    # Consistency check
    L0, Rtot, KxStar, Kav, Ctheta = commonChecks(L0, Rtot, KxStar, Kav, Ctheta)
    Cplx = jnp.array(Cplx)
    assert Cplx.ndim == 2
    assert Kav.shape[0] == Cplx.shape[1]
    assert Cplx.shape[0] == Ctheta.size

    # Solve Req
    Req = Req_solve(Req_polyc, Rtot, L0, KxStar, Cplx, Ctheta, Kav)

    # Calculate the results
    Psi = Req.T * Kav * KxStar
    Psi = jnp.concatenate((Psi, jnp.ones((Kav.shape[0], 1))), axis=1)
    Psirs = jnp.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Lbound = L0 / KxStar * Ctheta * jnp.expm1(jnp.dot(Cplx, jnp.log(Psirs))).flatten()
    Rbound = (
        L0
        / KxStar
        * Ctheta.reshape(-1, 1)
        * jnp.dot(Cplx, Psinorm)
        * jnp.exp(jnp.dot(Cplx, jnp.log(Psirs)))
    )
    with np.errstate(divide="ignore"):
        Lfbnd = (
            L0
            / KxStar
            * Ctheta
            * jnp.exp(jnp.dot(Cplx, jnp.log(Psirs - 1.0))).flatten()
        )
    assert len(Lbound) == len(Ctheta)
    assert Rbound.shape[0] == len(Ctheta)
    assert Rbound.shape[1] == len(Rtot)
    return Lbound, Rbound, Lfbnd

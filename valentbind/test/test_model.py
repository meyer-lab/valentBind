from collections.abc import Iterator

import jax
import numpy as np
import numpy.typing as npt
from jax import jacrev
from scipy.special import binom

from ..model import polyc, polyfc


def genPerm(len: int, sum: int) -> Iterator[list[int]]:
    """
    Enumerate every way to write ``sum`` as an ordered sum of ``len``
    non-negative integers.

    :param len: Number of terms in each partition.
    :param sum: Target value the terms of each partition must add to.
    :return: An iterator over all partitions (each a list of length ``len``)
        of ``sum`` into ``len`` non-negative integers.
    """
    if len <= 1:
        yield [sum]
    else:
        for i in range(sum + 1):
            for sub in genPerm(len - 1, sum - i):
                yield sub + [i]


def multinomial(params: npt.NDArray) -> float:
    """
    Compute the multinomial coefficient for the given group sizes.

    :param params: Sizes of each group; their sum is the total number of
        items being partitioned.
    :return: The multinomial coefficient ``sum(params)! / prod(p! for p in params)``.
    """
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


def polyfc2(
    L0: float,
    KxStar: float,
    f: int,
    Rtot: npt.ArrayLike,
    LigC: npt.ArrayLike,
    Kav: npt.ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """This function should give the same result as polyfc() but less efficient.
    This function is used for testing only.
    Use polyfc() for random complexes calculation.
    """
    LigC = np.array(LigC)
    assert LigC.ndim == 1
    LigC = LigC / np.sum(LigC)

    Cplx = np.array(list(genPerm(LigC.size, f)))
    Ctheta = np.exp(np.dot(Cplx, np.log(LigC).reshape(-1, 1))).flatten()
    Ctheta *= np.array([multinomial(Cplx[i, :]) for i in range(Cplx.shape[0])])
    assert abs(sum(Ctheta) - 1.0) < 1e-12

    return polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)


def test_grad() -> None:
    """Test the gradient of Lbnd w.r.t. Rtot."""
    L0 = 1.0e-9
    KxStar = 1.0e-12
    f = 8
    nl = np.random.randint(1, 10)
    nr = np.random.randint(1, 10)
    Rtot = np.floor(
        100.0 + np.random.rand(nr) * (10.0 ** np.random.randint(4, 6, size=nr))
    )
    LigC = np.random.rand(nl) * (10.0 ** np.random.randint(1, 2, size=nl))
    Kav = np.random.rand(nl, nr) * (10.0 ** np.random.randint(3, 7, size=(nl, nr)))

    gfunc = jacrev(lambda x: polyfc(L0, KxStar, f, x, LigC, Kav)[0], 0)
    outt = gfunc(Rtot)
    assert np.all(outt > 0.0)
    assert outt.shape == Rtot.shape


def test_equivalence() -> None:
    """Test that polyfc() and the slower reference polyfc2() agree, and
    check polyfc() internal consistency."""
    L0 = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
    KxStar = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
    f = np.random.randint(1, 10)
    nl = np.random.randint(1, 10)
    nr = np.random.randint(1, 10)
    Rtot = np.floor(
        100.0 + np.random.rand(nr) * (10.0 ** np.random.randint(4, 6, size=nr))
    )
    LigC = np.random.rand(nl) * (10.0 ** np.random.randint(1, 2, size=nl))
    Kav = np.random.rand(nl, nr) * (10.0 ** np.random.randint(3, 7, size=(nl, nr)))

    res = polyfc(L0, KxStar, f, Rtot, LigC, Kav)
    res2 = polyfc2(L0, KxStar, f, Rtot, LigC, Kav)

    # test polyfc math
    np.testing.assert_allclose(res[0], np.sum(res[2]))  # Lbound = sum(vieq)
    np.testing.assert_allclose(
        res[1], np.sum(res[2] * np.arange(1, f + 1))
    )  # Rbound = sum(each vieq * i)
    np.testing.assert_allclose(
        np.sum(res[3]), res[1] - np.sum(res[2][0])
    )  # sum(Rmulti_n) = Rbound - vieq[f = 1]

    # test equivalence
    np.testing.assert_allclose(res[0], np.sum(res2[0]))
    np.testing.assert_allclose(res[1], np.sum(res2[1]))


def test_null_monomer() -> None:
    """Test that adding a zero-affinity monomer ligand to a complex doesn't
    change binding."""
    # [3 0 0] should be equivalent to [3 0 5] if the last ligand has affinity 0
    L0 = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
    KxStar = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
    Rtot = [1e5]
    Kav = [[2e7], [3e5], [0]]

    res11 = polyc(L0, KxStar, Rtot, [[3, 0, 0]], [1], Kav)
    res12 = polyc(L0, KxStar, Rtot, [[3, 0, 5]], [1], Kav)
    res21 = polyc(L0, KxStar, Rtot, [[0, 6, 0]], [1], Kav)
    res22 = polyc(L0, KxStar, Rtot, [[0, 6, 3]], [1], Kav)
    res31 = polyc(L0, KxStar, Rtot, [[2, 4, 0]], [1], Kav)
    res32 = polyc(L0, KxStar, Rtot, [[2, 4, 5]], [1], Kav)

    for i in range(2):
        assert res11[i] == res12[i]
        assert res21[i] == res22[i]
        assert res31[i] == res32[i]


def test_Lfbnd() -> None:
    """Test that Lfbnd (singly-bound complex) sums to Lbound and matches
    per-receptor Rbound totals."""
    L0 = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
    KxStar = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
    nl = 4
    nr = np.random.randint(1, 10)
    Rtot = np.floor(
        100.0 + np.random.rand(nr) * (10.0 ** np.random.randint(4, 6, size=nr))
    )
    Kav = np.random.rand(nl, nr) * (10.0 ** np.random.randint(3, 7, size=(nl, nr)))
    Cplx = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Ctheta = np.random.rand(4)
    Ctheta = Ctheta / sum(Ctheta)

    res = polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)
    np.testing.assert_allclose(np.sum(res[0]), np.sum(res[2]))
    for i in range(len(res[0])):
        np.testing.assert_allclose(res[0][i], np.sum(res[1], axis=1)[i])

# valentBind

`valentBind` is a Python implementation of a multivalent binding model:
it computes how much ligand and receptor end up bound at equilibrium
when multivalent ligand complexes (e.g., antibodies, cytokine
complexes, or other multi-headed binders) interact with one or more
receptor types on a cell surface, accounting for avidity effects from
multiple simultaneous bonds.

It is used across several projects in the
[Meyer Lab](https://github.com/meyer-lab) to model antibody Fc-receptor
and cytokine-receptor binding.

## Installation

```bash
pip install git+https://github.com/meyer-lab/valentBind.git
```

## Quick start

There are two entry points, depending on whether your ligand complexes
are all identical (`polyfc`) or drawn from a mixture of different
complex compositions (`polyc`).

### `polyfc`: a single, homogeneous ligand complex

```python
from valentbind import polyfc

L0 = 1e-9  # concentration of ligand complexes (M)
KxStar = 1e-12  # detailed-balance-corrected cross-linking constant
f = 4  # valency of the ligand complex
Rtot = [1e5]  # total abundance of each receptor type on the cell
LigC = [1.0]  # relative composition of monomer ligands in the complex
Kav = [[1e6]]  # monomer ligand/receptor affinity matrix (ligands x receptors)

Lbound, Rbound, vieq, Rmulti_n = polyfc(L0, KxStar, f, Rtot, LigC, Kav)
```

`Lbound` is the total concentration of ligand complex bound, `Rbound`
is total receptor engaged, `vieq` breaks that down by how many
receptors each bound complex engages, and `Rmulti_n` is the receptor
abundance engaged in more than one bond.

### `polyc`: a mixture of heterogeneous ligand complexes

```python
from valentbind import polyc

L0 = 1e-9
KxStar = 1e-12
Rtot = [1e5, 2e4]  # two receptor types
Cplx = [[2, 0], [1, 1]]  # two complex types, each made of two monomer ligands
Ctheta = [0.7, 0.3]  # relative abundance of each complex type
Kav = [[1e6, 1e5], [1e5, 1e7]]  # affinities: 2 ligands x 2 receptors

Lbound, Rbound, Lfbnd = polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)
```

See the [API reference](api.md) for the full parameter and return
value documentation, and the [`examples/`](https://github.com/meyer-lab/valentBind/tree/main/examples)
directory for complete plotting scripts.

## Development

```bash
git clone https://github.com/meyer-lab/valentBind.git
cd valentBind
uv sync
make test   # run the test suite
make ty     # type check
uv run ruff check .   # lint
```

# valentBind

[![Test](https://github.com/meyer-lab/valentBind/actions/workflows/pytest.yml/badge.svg)](https://github.com/meyer-lab/valentBind/actions/workflows/pytest.yml)
[![Code Quality](https://github.com/meyer-lab/valentBind/actions/workflows/code-quality.yml/badge.svg)](https://github.com/meyer-lab/valentBind/actions/workflows/code-quality.yml)
[![Docs](https://github.com/meyer-lab/valentBind/actions/workflows/docs.yml/badge.svg)](https://meyer-lab.github.io/valentBind/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`valentBind` is a Python implementation of a multivalent binding model:
it computes how much ligand and receptor end up bound at equilibrium
when multivalent ligand complexes (e.g., antibodies, cytokine
complexes, or other multi-headed binders) interact with one or more
receptor types on a cell surface, accounting for avidity effects from
multiple simultaneous bonds. It's used across several projects in the
[Meyer Lab](https://github.com/meyer-lab) to model antibody Fc-receptor
and cytokine-receptor binding.

**[Full documentation and API reference](https://meyer-lab.github.io/valentBind/)**

## Installation

```bash
pip install git+https://github.com/meyer-lab/valentBind.git
```

## Quick start

There are two entry points, depending on whether your ligand complexes
are all identical (`polyfc`) or drawn from a mixture of different
complex compositions (`polyc`).

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

See the [docs](https://meyer-lab.github.io/valentBind/) for the full
API reference (including `polyc`, for mixtures of heterogeneous
complexes) and the [`examples/`](examples/) directory for complete
plotting scripts.

## Development

```bash
git clone https://github.com/meyer-lab/valentBind.git
cd valentBind
uv sync
make test   # run the test suite
uv run ruff check .   # lint
```

# Quantum Number Conservation Rules

[![10.5281/zenodo.5526360](https://zenodo.org/badge/doi/10.5281/zenodo.5526360.svg)](https://doi.org/10.5281/zenodo.5526360)
[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)

[![PyPI package](https://badge.fury.io/py/qrules.svg)](https://pypi.org/project/qrules)
[![Conda package](https://anaconda.org/conda-forge/qrules/badges/version.svg)](https://anaconda.org/conda-forge/qrules)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/qrules)](https://pypi.org/project/qrules)

[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/qrules/stable?filepath=docs/usage)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/qrules/blob/stable)
[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/ComPWA/qrules)
[![GitPod](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ComPWA/qrules)

[![Documentation](https://readthedocs.org/projects/qrules/badge/?version=latest)](https://qrules.readthedocs.io)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ComPWA/qrules/main.svg)](https://results.pre-commit.ci/latest/github/ComPWA/qrules/main)
[![pytest](https://github.com/ComPWA/qrules/workflows/pytest/badge.svg)](https://github.com/ComPWA/qrules/actions?query=branch%3Amain+workflow%3Apytest)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy.readthedocs.io)
[![Test coverage](https://codecov.io/gh/ComPWA/qrules/branch/main/graph/badge.svg)](https://codecov.io/gh/ComPWA/qrules)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/deeee5b9e2bb4b3daa655942c71e17da)](https://www.codacy.com/gh/ComPWA/qrules)
[![Spelling checked](https://img.shields.io/badge/cspell-checked-brightgreen.svg)](https://github.com/streetsidesoftware/cspell/tree/master/packages/cspell)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort)

QRules is a Python package for **validating and generating particle reactions**
using quantum number conservation rules. The user only has to provide a certain
set of boundary conditions (initial and final state, allowed interaction types,
expected decay topologies, etc.). QRules will then span the space of allowed
quantum numbers over all allowed decay topologies and particle instances that
correspond with the sets of allowed quantum numbers it has found.

The resulting state transition objects are particularly useful for **amplitude
analysis / Partial Wave Analysis** as they contain all information (such as
expected masses, widths, and spin projections) that is needed to formulate an
amplitude model.

Visit [qrules.rtfd.io](https://qrules.readthedocs.io) for more information!

For an overview of **upcoming releases and planned functionality**, see
[here](https://github.com/ComPWA/qrules/milestones?direction=asc&sort=title&state=open).

## Available features

- **Input**: Particle database
  - Source of truth: PDG
  - Predefined particle list file
  - Option to overwrite and append with custom particle definitions
- **State transition graph**
  - Feynman graph like description of the reactions
  - Visualization of the decay topology
- **Conservation rules**
  - Open-closed design
  - Large set of predefined rules
    - Spin/Angular momentum conservation
    - Quark and Lepton flavor conservation (incl. isospin)
    - Baryon number conservation
    - EM-charge conservation
    - Parity, C-Parity, G-Parity conservation
    - Mass conservation
  - Predefined sets of conservation rules representing Strong, EM, Weak
    interactions

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)

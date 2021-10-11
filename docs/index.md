# Welcome to QRules!

```{title} Welcome

```

[![10.5281/zenodo.5526360](https://zenodo.org/badge/doi/10.5281/zenodo.5526360.svg)](https://doi.org/10.5281/zenodo.5526360)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/qrules)](https://pypi.org/project/qrules)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/qrules/blob/stable)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/qrules/stable?filepath=docs/usage)

:::{margin}

The original project was the
[PWA Expert System](https://expertsystem.readthedocs.io). QRules originates
from its
[`reaction`](https://expertsystem.readthedocs.io/en/stable/api/expertsystem.reaction.html)
module.

:::

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

The {doc}`/usage` pages illustrate several features of {mod}`qrules`. You can
run each of them as Jupyter notebooks with the {fa}`rocket` launch button in
the top-right corner. Enjoy!

```{rubric} Internal design

```

QRules consists of three major components:

1. **State transition graphs**

   A {class}`.StateTransitionGraph` is a
   [directed graph](https://en.wikipedia.org/wiki/Directed_graph) that consists
   of **nodes** and **edges**. In a directed graph, each edge must be connected
   to at least one node (in correspondence to
   [Feynman graphs](https://en.wikipedia.org/wiki/Feynman_diagram)). This way,
   a graph describes the transition from one state to another.

   - **Edges** correspond to states (particles with spin). In other words,
     edges are a collection of properties such as the quantum numbers that
     characterize a state that the particle is in.

   - **Nodes** represents interactions and contain all information for the
     transition of this specific step. Most importantly, a node contains a
     collection of conservation rules that have to be satisfied. An interaction
     node has $M$ ingoing lines and $N$ outgoing lines, where
     $M,N \in \mathbb{Z}$, $M > 0, N > 0$.

2. **Conservation rules**

   The central component are the {mod}`.conservation_rules`. They belong to
   individual nodes and receive properties about the node itself, as well as
   properties of the ingoing and outgoing edges of that node. Based on those
   properties the conservation rules determine whether edges pass or not.

3. **Solvers**

   The determination of the correct state properties in the graph is done by
   solvers. New properties are set for intermediate edges and interaction nodes
   and their validity is checked with the conservation rules.

:::{margin}

The main solver used by {mod}`qrules` is the
[`constraint`](https://labix.org/doc/constraint) package for the
[Constraint Satisfaction Problem](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem)
(CSP).

:::

```{rubric} QRules workflow

```

1. **Preparation**

   1.1. Build all possible topologies. A **topology** is represented by a
   {class}`.StateTransitionGraph`, in which the edges and nodes are empty (no
   particle information).

   1.2. Fill the topology graphs with the user provided information. Typically
   these are the graph's ingoing edges (initial state) and outgoing edges
   (final state).

2. **Solving**

   2.1. _Propagate_ quantum number information through the complete graph while
   respecting the specified conservation laws. Information like mass is not
   used in this first solving step.

   2.2. _Clone_ graphs while inserting concrete matching particles for the
   intermediate edges (mainly adds the mass variable).

   2.3. _Validate_ the complete graphs, so run all conservation law check that
   were postponed from the first step.

```{rubric} Table of Contents

```

```{toctree}
---
maxdepth: 2
---
install
usage
references
API <api/qrules>
Changelog <https://github.com/ComPWA/qrules/releases>
Upcoming features <https://github.com/ComPWA/qrules/milestones?direction=asc&sort=title&state=open>
Help developing <https://compwa-org.readthedocs.io/en/stable/develop.html>
```

- {ref}`Python API <modindex>`
- {ref}`General Index <genindex>`
- {ref}`Search <search>`

```{toctree}
---
caption: Related projects
hidden:
---
AmpForm <https://ampform.readthedocs.io>
TensorWaves <https://tensorwaves.readthedocs.io>
PWA Pages <https://pwa.readthedocs.io>
```

```{toctree}
---
caption: ComPWA Organization
hidden:
---
Website <https://compwa-org.readthedocs.io>
GitHub Repositories <https://github.com/ComPWA>
About <https://compwa-org.readthedocs.io/en/stable/about.html>
```

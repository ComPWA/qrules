# Welcome to QRules!

```{title} Welcome

```

[![PyPI package](https://badge.fury.io/py/qrules.svg)](https://pypi.org/project/qrules)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/qrules)](https://pypi.org/project/qrules)
[![Test coverage](https://codecov.io/gh/ComPWA/qrules/branch/main/graph/badge.svg?token=PPRMC5E6SX)](https://codecov.io/gh/ComPWA/qrules)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/deeee5b9e2bb4b3daa655942c71e17da)](https://www.codacy.com/gh/ComPWA/qrules)

````{margin}
```{tip}
For an overview of upcoming releases and planned functionality, see
[here](https://github.com/ComPWA/qrules/milestones?direction=asc&sort=title&state=open).
```
````

QRules is a system for validating and generating particle reactions, using
quantum number conservation rules. The user only has to provide a basic
information of the particle reaction, such as an initial state and a final
state. Helper functions provide easy ways to configure the system, but the user
still has full control. QRules then constructs several hypotheses for what
happens during the transition from initial to final state.

:::{dropdown} Original project: PWA Expert System

The original project was the {doc}`PWA Expert System <expertsystem:index>`.
QRules originates from its {mod}`~expertsystem.reaction` module.

:::

## Internal design

Internally, QRules consists of three major components.

### 1. State Transition Graphs

A {class}`.StateTransitionGraph` is a
[directed graph](https://en.wikipedia.org/wiki/Directed_graph) that consists of
**nodes** and **edges**. In a directed graph, each edge must be connected to at
least one node (in correspondence to Feynman graphs). This way, a graph
describes the transition from one state to another.

- The edges correspond to particles/states, in other words a collection of
  properties such as the quantum numbers that characterize the particle state.

- Each node represents an interaction and contains all information for the
  transition of this specific step. Most importantly, a node contains a
  collection of conservation rules that have to be satisfied. An interaction
  node has $M$ ingoing lines and $N$ outgoing lines, where
  $M,N \in \mathbb{Z}$, $M > 0, N > 0$.

### 2. Conservation Rules

The central component of the expert system are the
{mod}`conservation rules <.conservation_rules>`. They belong to individual
nodes and receive properties about the node itself, as well as properties of
the ingoing and outgoing edges of that node. Based on those properties the
conservation rules determine whether edges pass or not.

### 3. Solvers

The determination of the correct state properties in the graph is done by
solvers. New properties are set for intermediate edges and interaction nodes
and their validity is checked with the conservation rules.

## QRules workflow

1. Preparation

   1.1. Build all possible topologies. A **topology** is represented by a
   {ref}`graph <index:1. State Transition Graphs>`, in which the edges and
   nodes are empty (no particle information).

   1.2. Fill the topology graphs with the user provided information. Typically
   these are the graph's ingoing edges (initial state) and outgoing edges
   (final state).

2. Solving

   2.1. _Propagate_ quantum number information through the complete graph while
   respecting the specified conservation laws. Information like mass is not
   used in this first solving step.

   2.2. _Clone_ graphs while inserting concrete matching particles for the
   intermediate edges (mainly adds the mass variable).

   2.3. _Validate_ the complete graphs, so run all conservation law check that
   were postponed from the first step.

## Table of Contents

```{toctree}
---
maxdepth: 2
---
install
usage
references
API <api/qrules>
Changelog <https://github.com/ComPWA/qrules/releases>
Develop <https://pwa.readthedocs.io/develop.html>
```

- {ref}`Python API <modindex>`
- {ref}`General Index <genindex>`
- {ref}`Search <search>`

```{toctree}
---
caption: Related projects
hidden:
---
AmpForm <http://ampform.readthedocs.io>
TensorWaves <http://tensorwaves.readthedocs.io>
PWA Pages <http://pwa.readthedocs.io>
```

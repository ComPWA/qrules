# Installation

[![PyPI package](https://badge.fury.io/py/qrules.svg)](https://pypi.org/project/qrules)
[![Conda package](https://anaconda.org/conda-forge/qrules/badges/version.svg)](https://anaconda.org/conda-forge/qrules)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/qrules)](https://pypi.org/project/qrules)

## Quick installation

The fastest way of installing this package is through PyPI or Conda:

:::{tabbed} PyPI

```shell
python3 -m pip install qrules
```

:::

:::{tabbed} Conda

```shell
conda install -c conda-forge qrules
```

:::

This installs the [latest release](https://github.com/ComPWA/qrules/releases)
that you can find on the
[`stable`](https://github.com/ComPWA/qrules/tree/stable) branch.

Optionally, you can install the dependencies required for
{doc}`visualizing topologies </usage/visualize>` with the following
{ref}`optional dependency syntax <compwa-org:develop:Optional dependencies>`:

```shell
pip install qrules[viz]  # installs qrules with graphviz
```

The latest version on the [`main`](https://github.com/ComPWA/qrules/tree/main)
branch can be installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/qrules@main
```

## Editable installation

It is highly recommend to use the more dynamic
{ref}`'editable installation' <compwa-org:develop:Editable installation>`. This
allows you to:

- exactly
  {ref}`pin all dependencies <compwa-org:develop:Pinning dependency versions>`
  to a specific version, so that your work is **reproducible**.
- edit the source code of the framework and
  {doc}`help improving it <compwa-org:develop>`.

For this, you first need to get the source code with
[Git](https://git-scm.com):

```shell
git clone https://github.com/ComPWA/qrules.git
cd qrules
```

Next, you install the project in editable mode with either
[Conda](https://docs.conda.io) or [`pip`](https://pypi.org/project/pip). It's
recommended to use Conda, because this also pins the version of Python.

:::{tabbed} Conda

```shell
conda env create
```

This installs the project in a Conda environment following the definitions in
[`environment.yml`](https://github.com/ComPWA/qrules/blob/main/environment.yml).

:::

:::{tabbed} PyPI

1. **[Recommended]** Create a virtual environment with
   [`venv`](https://docs.python.org/3/library/venv.html) (see
   {ref}`here <compwa-org:develop:Virtual environment>`).

2. Install the project as an
   {ref}`'editable installation' <compwa-org:develop:Editable installation>`
   with {ref}`additional packages <compwa-org:develop:Optional dependencies>`
   for the developer and all dependencies pinned through
   [constraints files](https://pip.pypa.io/en/stable/user_guide/#constraints-files):

   ```shell
   python3 -m pip install -c .constraints/py3.x.txt -e .[dev]
   ```

:::

See {ref}`compwa-org:develop:Updating` for how to update the dependencies when
new commits come in.

That's all! Have a look at {doc}`/usage` to try out the package. You can also
have a look at {doc}`compwa-org:develop` for tips on how to work with this
'editable' developer setup!

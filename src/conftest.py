"""Pytest configuration for the doctests in :mod:`qrules`.

A ``conftest.py`` only applies to its own directory tree, so the doctests that pytest
collects from ``src`` (see ``testpaths`` and ``--doctest-modules``) are not affected by
`tests/conftest.py <../tests/conftest.py>`_.
"""

from qrules.settings import NumberOfThreads

# Ensure consistent test coverage and avoid nested multiprocessing when running pytest
# multithreaded (e.g. with pytest-xdist)
# https://github.com/ComPWA/qrules/issues/11
NumberOfThreads.set(n_cores=1)

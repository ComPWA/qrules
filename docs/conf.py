from __future__ import annotations

import os
import sys

from sphinx_api_relink.helpers import (
    get_branch_name,
    get_execution_mode,
    get_package_version,
    pin,
    set_intersphinx_version_remapping,
)

sys.path.insert(0, os.path.abspath("."))
from _extend_docstrings import extend_docstrings  # noqa: PLC2701

extend_docstrings()
set_intersphinx_version_remapping({
    "ipython": {
        "8.12.2": "8.12.1",
        "8.12.3": "8.12.1",
    },
})

BRANCH = get_branch_name()
ORGANIZATION = "ComPWA"
PACKAGE = "qrules"
REPO_NAME = "qrules"
REPO_TITLE = "Quantum number conservation rules"

BINDER_LINK = f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?filepath=docs/usage"

add_module_names = False
api_github_repo = f"{ORGANIZATION}/{REPO_NAME}"
api_target_substitutions: dict[str, str | tuple[str, str]] = {
    "EdgeType": "typing.TypeVar",
    "GraphEdgePropertyMap": ("obj", "qrules.argument_handling.GraphEdgePropertyMap"),
    "GraphElementProperties": ("obj", "qrules.solving.GraphElementProperties"),
    "GraphNodePropertyMap": ("obj", "qrules.argument_handling.GraphNodePropertyMap"),
    "GraphSettings": ("obj", "qrules.solving.GraphSettings"),
    "InitialFacts": ("obj", "qrules.combinatorics.InitialFacts"),
    "NewEdgeType": "typing.TypeVar",
    "NewNodeType": "typing.TypeVar",
    "NodeQuantumNumber": ("obj", "qrules.quantum_numbers.NodeQuantumNumber"),
    "NodeType": "typing.TypeVar",
    "ParticleWithSpin": ("obj", "qrules.particle.ParticleWithSpin"),
    "Path": "pathlib.Path",
    "qrules.topology.EdgeType": "typing.TypeVar",
    "qrules.topology.NodeType": "typing.TypeVar",
    "SpinFormalism": ("obj", "qrules.transition.SpinFormalism"),
    "StateDefinition": ("obj", "qrules.combinatorics.StateDefinition"),
    "typing.Literal[-1, 1]": "typing.Literal",
}
api_target_types: dict[str, str | tuple[str, str]] = {
    "qrules.combinatorics.InitialFacts": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.baryon_number": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.bottomness": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.c_parity": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.charge": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.charmness": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.electron_lepton_number": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.g_parity": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.isospin_magnitude": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.isospin_projection": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.mass": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.muon_lepton_number": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.parity": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.pid": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.spin_magnitude": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.spin_projection": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.strangeness": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.tau_lepton_number": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.topness": "obj",
    "qrules.quantum_numbers.EdgeQuantumNumbers.width": "obj",
    "qrules.quantum_numbers.NodeQuantumNumbers.l_magnitude": "obj",
    "qrules.quantum_numbers.NodeQuantumNumbers.l_projection": "obj",
    "qrules.quantum_numbers.NodeQuantumNumbers.parity_prefactor": "obj",
    "qrules.quantum_numbers.NodeQuantumNumbers.s_magnitude": "obj",
    "qrules.quantum_numbers.NodeQuantumNumbers.s_projection": "obj",
    "qrules.solving.GraphElementProperties": "obj",
    "qrules.solving.GraphSettings": "obj",
    "qrules.transition.StateTransition": "obj",
}
author = "Common Partial Wave Analysis"
autodoc_default_options = {
    "exclude-members": ", ".join([
        "items",
        "keys",
        "values",
    ]),
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": ", ".join([
        "__call__",
    ]),
}
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "GraphElementProperties": "qrules.solving.GraphElementProperties",
    "GraphSettings": "qrules.solving.GraphSettings",
    "InitialFacts": "qrules.combinatorics.InitialFacts",
    "StateTransition": "qrules.transition.StateTransition",
}
autodoc_typehints_format = "short"
autosectionlabel_prefix_document = True
bibtex_bibfiles = ["bibliography.bib"]
codeautolink_concat_default = True
comments_config = {
    "hypothesis": True,
    "utterances": {
        "repo": f"{ORGANIZATION}/{REPO_NAME}",
        "issue-term": "pathname",
        "label": "📝 Docs",
    },
}
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest
copyright = f"2020, {ORGANIZATION}"
default_role = "py:obj"
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr/template.md",
    "tests",
]
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_api_relink",
    "sphinx_codeautolink",
    "sphinx_comments",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_hep_pdgref",
    "sphinx_pybtex_etal_style",
    "sphinx_thebe",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
]
generate_apidoc_package_path = f"../src/{PACKAGE}"
graphviz_output_format = "svg"
html_copy_source = True  # needed for download notebook button
html_css_files = ["linebreaks-api.css"]
html_favicon = "_static/favicon.ico"
html_last_updated_fmt = "%-d %B %Y"
html_logo = (
    "https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg"
)
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "Common Partial Wave Analysis",
            "url": "https://compwa.github.io",
            "icon": "_static/favicon.ico",
            "type": "local",
        },
        {
            "name": "GitHub",
            "url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": f"https://pypi.org/project/{PACKAGE}",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Conda",
            "url": f"https://anaconda.org/conda-forge/{PACKAGE}",
            "icon": "https://avatars.githubusercontent.com/u/22454001?s=100",
            "type": "url",
        },
        {
            "name": "Launch on Binder",
            "url": f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?filepath=docs",
            "icon": "https://mybinder.readthedocs.io/en/latest/_static/favicon.png",
            "type": "url",
        },
        {
            "name": "Launch on Colaboratory",
            "url": f"https://colab.research.google.com/github/{ORGANIZATION}/{REPO_NAME}/blob/{BRANCH}",
            "icon": "https://avatars.githubusercontent.com/u/33467679?s=100",
            "type": "url",
        },
    ],
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "logo": {"text": REPO_TITLE},
    "path_to_docs": "docs",
    "repository_branch": BRANCH,
    "repository_url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
}
html_title = REPO_TITLE
intersphinx_mapping = {
    "ampform": ("https://ampform.readthedocs.io/en/stable", None),
    "attrs": (f"https://www.attrs.org/en/{pin('attrs')}", None),
    "compwa": ("https://compwa.github.io", None),
    "constraint": ("https://python-constraint.github.io/python-constraint", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable", None),
    "IPython": (f"https://ipython.readthedocs.io/en/{pin('IPython')}", None),
    "jsonschema": ("https://python-jsonschema.readthedocs.io/en/stable", None),
    "mypy": ("https://mypy.readthedocs.io/en/stable", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "python": ("https://docs.python.org/3", None),
}
linkcheck_anchors = False
linkcheck_ignore = [
    "https://doi.org/10.1002/andp.19955070504",  # 403 for onlinelibrary.wiley.com
]
project = REPO_TITLE
modindex_common_prefix = [f"{PACKAGE}."]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 2
myst_substitutions = {
    "branch": BRANCH,
    "run_interactive": f"""
```{{margin}}
Run this notebook [on Binder]({BINDER_LINK}) or
{{ref}}`locally on Jupyter Lab <compwa:develop:Jupyter Notebooks>` to interactively
modify the parameters.
```
""",
}
myst_update_mathjax = False
nb_execution_mode = get_execution_mode()
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_output_stderr = "remove"
nitpick_ignore_regex = [
    (r"py:(class|obj)", "json.encoder.JSONEncoder"),
    (r"py:(class|obj)", r"qrules\.topology\.EdgeType"),
    (r"py:(class|obj)", r"qrules\.topology\.KT"),
    (r"py:(class|obj)", r"qrules\.topology\.NewEdgeType"),
    (r"py:(class|obj)", r"qrules\.topology\.NewNodeType"),
    (r"py:(class|obj)", r"qrules\.topology\.NodeType"),
    (r"py:(class|obj)", r"qrules\.topology\.VT"),
]
nitpicky = True
primary_domain = "py"
project = "QRules"
pygments_style = "sphinx"
release = get_package_version(PACKAGE)
suppress_warnings = [
    "myst.domains",
    # skipping unknown output mime type: application/json
    # https://github.com/ComPWA/qrules/runs/8132605149?check_suite_focus=true#step:5:92
    "mystnb.unknown_mime_type",
]
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
version = get_package_version(PACKAGE)

# flake8: noqa
# pylint: disable=import-error,import-outside-toplevel,invalid-name,protected-access
# pyright: reportMissingImports=false
"""Extend docstrings of the API.

This small script is used by ``conf.py`` to dynamically modify docstrings.
"""

import inspect
import logging
import textwrap
from typing import Callable, Dict, Optional, Type, Union

import qrules

logging.getLogger().setLevel(logging.ERROR)


def extend_docstrings() -> None:
    script_name = __file__.rsplit("/", maxsplit=1)[-1]
    script_name = ".".join(script_name.split(".")[:-1])
    definitions = dict(globals())
    for name, definition in definitions.items():
        module = inspect.getmodule(definition)
        if module is None:
            continue
        if module.__name__ not in {"__main__", script_name}:
            continue
        if not inspect.isfunction(definition):
            continue
        if not name.startswith("extend_"):
            continue
        if name == "extend_docstrings":
            continue
        function_arguments = inspect.signature(definition).parameters
        if len(function_arguments):
            raise ValueError(f"Local function {name} should not have a signature")
        definition()


def extend_create_isobar_topologies() -> None:
    from qrules.topology import create_isobar_topologies

    topologies = qrules.topology.create_isobar_topologies(4)
    dot_renderings = map(
        lambda t: qrules.io.asdot(t, render_resonance_id=True),
        topologies,
    )
    images = [_graphviz_to_image(dot, indent=6) for dot in dot_renderings]
    _append_to_docstring(
        create_isobar_topologies,
        f"""

    .. panels::
      :body: text-center
      {images[0]}

      ---
      {images[1]}
    """,
    )


def extend_create_n_body_topology() -> None:
    from qrules.topology import create_n_body_topology

    topology = create_n_body_topology(
        number_of_initial_states=2,
        number_of_final_states=5,
    )
    dot = qrules.io.asdot(topology, render_initial_state_id=True)
    _append_to_docstring(
        create_n_body_topology,
        _graphviz_to_image(dot, indent=4),
    )


def extend_Topology() -> None:
    from qrules.topology import Topology, create_isobar_topologies

    topologies = create_isobar_topologies(number_of_final_states=3)
    dot = qrules.io.asdot(
        topologies[0],
        render_initial_state_id=True,
        render_resonance_id=True,
    )
    _append_to_docstring(
        Topology,
        _graphviz_to_image(dot, indent=4),
    )


def _append_to_docstring(class_type: Union[Callable, Type], appended_text: str) -> None:
    assert class_type.__doc__ is not None
    class_type.__doc__ += appended_text


_GRAPHVIZ_COUNTER = 0
_IMAGE_DIR = "_images"


def _graphviz_to_image(  # pylint: disable=too-many-arguments
    dot: str,
    options: Optional[Dict[str, str]] = None,
    format: str = "svg",
    indent: int = 0,
    caption: str = "",
    label: str = "",
) -> str:
    import graphviz  # type: ignore[import]

    if options is None:
        options = {}
    global _GRAPHVIZ_COUNTER  # pylint: disable=global-statement
    output_file = f"graphviz_{_GRAPHVIZ_COUNTER}"
    _GRAPHVIZ_COUNTER += 1
    graphviz.Source(dot).render(f"{_IMAGE_DIR}/{output_file}", format=format)
    restructuredtext = "\n"
    if label:
        restructuredtext += f".. _{label}:\n"
    restructuredtext += f".. figure:: /{_IMAGE_DIR}/{output_file}.{format}\n"
    for option, value in options.items():
        restructuredtext += f"  :{option}: {value}\n"
    if caption:
        restructuredtext += f"\n  {caption}\n"
    return textwrap.indent(restructuredtext, indent * " ")

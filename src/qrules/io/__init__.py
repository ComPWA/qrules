"""Serialization module for the `qrules`.

The `.io` module provides tools to export or import objects from `qrules` to and from
disk, so that they can be used by external packages, or just to store (cache) the state
of the system.
"""

from __future__ import annotations

import base64
import io as stdlib_io
import json
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import requests
import yaml

from qrules.io import _dict, _dot, _mermaid
from qrules.particle import Particle, ParticleCollection
from qrules.topology import Topology

if TYPE_CHECKING:
    from _typeshed import StrPath
    from PIL.Image import Image


def asdict(instance: object) -> dict:
    if isinstance(instance, ParticleCollection):
        return _dict.from_particle_collection(instance)
    if attrs.has(type(instance)):
        return _dict.from_attrs_decorated(instance)
    msg = f"No conversion to dict available for class {type(instance).__name__}"
    raise NotImplementedError(msg)


def fromdict(definition: dict) -> object:
    keys = set(definition.keys())
    if keys >= __REQUIRED_PARTICLE_FIELDS:
        return _dict.build_particle(definition)
    if keys == {"particles"}:
        return _dict.build_particle_collection(definition)
    if keys == {"transitions", "formalism"}:
        return _dict.build_reaction_info(definition)
    if keys == {"topology", "states", "interactions"}:
        return _dict.build_transition(definition)
    if keys == __REQUIRED_TOPOLOGY_FIELDS:
        return _dict.build_topology(definition)
    msg = f"Could not determine type from keys {keys}"
    raise NotImplementedError(msg)


__REQUIRED_PARTICLE_FIELDS = {
    field.name for field in attrs.fields(Particle) if field.default == attrs.NOTHING
}
__REQUIRED_TOPOLOGY_FIELDS = {
    field.name for field in attrs.fields(Topology) if field.init
}


def asdot(
    instance: object,
    *,
    render_node: bool | None = None,
    render_final_state_id: bool = True,
    render_resonance_id: bool = False,
    render_initial_state_id: bool = False,
    strip_spin: bool = False,
    collapse_graphs: bool = False,
    edge_style: dict[str, Any] | None = None,
    node_style: dict[str, Any] | None = None,
    **figure_style: Any,
) -> str:
    """Convert a `object` to a DOT language `str`.

    Only works for objects that can be represented as a graph, particularly a
    `.MutableTransition` or a `list` of `.MutableTransition` instances.

    Args:
        instance: the input `object` that is to be rendered as DOT (graphviz) language.

        strip_spin: Normally, each `.MutableTransition` has a `.Particle` with a spin
            projection on its edges. This option hides the projections, leaving only
            `.Particle` names on edges.

        collapse_graphs: Group all transitions by equivalent kinematic topology
            and combine all allowed particles on each edge.

        render_node: Whether or not to render node ID (in the case of a `.Topology`)
            and/or node properties (in the case of a `.MutableTransition`). Meaning of
            the labels:

            - :math:`P`: parity prefactor
            - :math:`s`: tuple of **coupled spin** magnitude and its
              projection
            - :math:`l`: tuple of **angular momentum** and its projection

            See `.InteractionProperties` for more info.

        render_final_state_id: Add edge IDs for the final state edges.
        render_resonance_id: Add edge IDs for the intermediate state edges.
        render_initial_state_id: Add edge IDs for the initial state edges.
        edge_style: Styling of a Graphviz edge.
        node_style: Styling of a Graphviz node.
        figure_style: Styling of the whole figure.

    .. seealso::

        See `Graphviz attributes <https://graphviz.org/doc/info/attrs.html>`_ for the
        available styling arguments.

    .. seealso:: :doc:`/usage/visualize`
    """
    print_dot = _dot.GraphPrinter(
        render_node=render_node,
        render_final_state_id=render_final_state_id,
        render_resonance_id=render_resonance_id,
        render_initial_state_id=render_initial_state_id,
        strip_spin=strip_spin,
        collapse_graphs=collapse_graphs,
        figure_style=figure_style,
        edge_style=edge_style,
        node_style=node_style,
    )
    return print_dot(instance)


def asmermaid_source(
    instance: object,
    *,
    render_node: bool | None = None,
    render_final_state_id: bool = True,
    render_resonance_id: bool = False,
    render_initial_state_id: bool = False,
    strip_spin: bool = False,
    collapse_graphs: bool = False,
    figure_style: dict[str, Any] | None = None,
    edge_style: dict[str, Any] | None = None,
    node_style: dict[str, Any] | None = None,
) -> str:
    """Convert a `object` to a Mermaid flowchart source `str`.

    This mirrors the public interface of :func:`asdot` for the Mermaid renderer.
    """
    print_mermaid = _mermaid.MermaidPrinter(
        render_node=render_node,
        render_final_state_id=render_final_state_id,
        render_resonance_id=render_resonance_id,
        render_initial_state_id=render_initial_state_id,
        strip_spin=strip_spin,
        collapse_graphs=collapse_graphs,
        figure_style=figure_style,
        edge_style=edge_style,
        node_style=node_style,
    )
    return print_mermaid(instance)


def asmermaid(
    instance: object,
    *,
    render_node: bool | None = None,
    render_final_state_id: bool = True,
    render_resonance_id: bool = False,
    render_initial_state_id: bool = False,
    strip_spin: bool = False,
    collapse_graphs: bool = False,
    figure_style: dict[str, Any] | None = None,
    edge_style: dict[str, Any] | None = None,
    node_style: dict[str, Any] | None = None,
) -> str:
    """Convert a `object` to a Mermaid Markdown `str`.

    The returned string contains a fenced Mermaid block that can be rendered by
    JupyterLab, Sphinx, or other Markdown consumers with Mermaid support.
    """
    source = asmermaid_source(
        instance,
        render_node=render_node,
        render_final_state_id=render_final_state_id,
        render_resonance_id=render_resonance_id,
        render_initial_state_id=render_initial_state_id,
        strip_spin=strip_spin,
        collapse_graphs=collapse_graphs,
        figure_style=figure_style,
        edge_style=edge_style,
        node_style=node_style,
    )
    return f"```mermaid\n{source}\n```"


def show_mermaid_markdown(
    instance: object,
    *,
    render_node: bool | None = None,
    render_final_state_id: bool = True,
    render_resonance_id: bool = False,
    render_initial_state_id: bool = False,
    strip_spin: bool = False,
    collapse_graphs: bool = False,
    figure_style: dict[str, Any] | None = None,
    edge_style: dict[str, Any] | None = None,
    node_style: dict[str, Any] | None = None,
) -> Any:
    """Display Mermaid Markdown in a notebook.

    This is a small convenience wrapper around :func:`asmermaid` for IPython and
    Jupyter notebooks.
    """
    from IPython.display import Markdown, display

    markdown = Markdown(
        asmermaid(
            instance,
            render_node=render_node,
            render_final_state_id=render_final_state_id,
            render_resonance_id=render_resonance_id,
            render_initial_state_id=render_initial_state_id,
            strip_spin=strip_spin,
            collapse_graphs=collapse_graphs,
            figure_style=figure_style,
            edge_style=edge_style,
            node_style=node_style,
        )
    )
    display(markdown)
    # return markdown


def render_mermaid_image(source: str) -> Image:
    """Render a Mermaid flowchart source to a PIL image.

    The rendering is delegated to the public Mermaid Ink service. Use
    :func:`show_mermaid_image` for a convenience display helper.
    """
    graph_bytes = source.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")
    try:
        response = requests.get(
            f"https://mermaid.ink/img/{base64_string}",
            timeout=20,
            headers={"User-Agent": "qrules-mermaid-renderer/1.0"},
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code == 400:
            msg = (
                "Mermaid image rendering failed with HTTP 400. "
                "The source may be too large or contain unsupported syntax."
            )
            raise RuntimeError(msg) from exc
        msg = f"Mermaid image rendering failed with HTTP {status_code}."
        raise RuntimeError(msg) from exc
    except requests.RequestException as exc:
        msg = "Mermaid image rendering failed due to a network error."
        raise RuntimeError(msg) from exc

    image_bytes = response.content

    from PIL import Image

    return Image.open(stdlib_io.BytesIO(image_bytes))


def show_mermaid_image(
    source: str,
    *,
    figsize: tuple[float, float] = (8, 4),
    fallback_to_source: bool = True,
    max_source_chars: int = 4000,
) -> bool:
    """Render Mermaid source and display it with Matplotlib.

    Returns:
        `True` if the image could be rendered and shown, `False` if fallback output
        was used.
    """
    import matplotlib.pyplot as plt

    try:
        image = render_mermaid_image(source)
    except RuntimeError as exc:
        if not fallback_to_source:
            raise
        print("Image rendering failed. Falling back to Mermaid source output.")
        print(f"Reason: {exc}")
        if len(source) <= max_source_chars:
            print(source)
        else:
            print(source[:max_source_chars])
            print("... source output truncated ...")
        return False

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    return True


def load(filename: str | Path) -> object:
    with open(filename) as stream:
        file_extension = _get_file_extension(filename)
        if file_extension == "json":
            definition = json.load(stream)
            return fromdict(definition)
        if file_extension in {"yaml", "yml"}:
            definition = yaml.load(stream, Loader=yaml.SafeLoader)
            return fromdict(definition)
    msg = f'No loader defined for file type "{file_extension}"'
    raise NotImplementedError(msg)


class _IncreasedIndent(yaml.Dumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:  # ruff: ignore[unused-method-argument]
        return super().increase_indent(flow, indentless=False)

    def write_line_break(self, data: str | None = None) -> None:
        """See https://stackoverflow.com/a/44284819."""
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


def write(instance: object, filename: StrPath) -> None:
    with open(filename, "w") as stream:
        file_extension = _get_file_extension(filename)
        if file_extension == "json":
            json.dump(asdict(instance), stream, indent=2, cls=JSONSetEncoder)
            return
        if file_extension in {"yaml", "yml"}:
            yaml.dump(
                asdict(instance),
                stream,
                sort_keys=False,
                Dumper=_IncreasedIndent,
                default_flow_style=False,
            )
            return
        if file_extension == "gv":
            if isinstance(instance, str):  # direct output of asdot
                output_str = instance
            else:
                output_str = asdot(instance)
            with open(filename, "w") as stream:
                stream.write(output_str)
            return
        if file_extension == "mmd":
            if isinstance(instance, str):  # direct output of asmermaid_source
                output_str = instance
            else:
                output_str = asmermaid_source(instance)
            output_str = _normalize_mermaid_file_content(output_str, file_extension)
            stream.write(output_str)
            return
        if file_extension == "md":
            if isinstance(instance, str):  # direct output of asmermaid
                output_str = instance
            else:
                output_str = asmermaid(instance)
            output_str = _normalize_mermaid_file_content(output_str, file_extension)
            stream.write(output_str)
            return
    msg = f'No writer defined for file type "{file_extension}"'
    raise NotImplementedError(msg)


_MERMAID_FENCE_PATTERN = re.compile(
    r"^\s*```mermaid[ \t]*\n(?P<source>.*?)\n```[ \t]*$",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_mermaid_file_content(content: str, file_extension: str) -> str:
    match = _MERMAID_FENCE_PATTERN.match(content)
    if file_extension == "mmd":
        if match is not None:
            warnings.warn(
                "Markdown fence removed from .mmd Mermaid file content.",
                UserWarning,
                stacklevel=3,
            )
            return f"{match.group('source')}\n"
        return content

    if match is None:
        warnings.warn(
            "Markdown fence added to .md Mermaid file content.",
            UserWarning,
            stacklevel=3,
        )
        return f"```mermaid\n{content.rstrip()}\n```\n"
    return content


def _get_file_extension(filename: StrPath) -> str:
    path = Path(filename)
    extension = path.suffix.lower()
    if not extension:
        msg = f'No file extension in file name "{filename}"'
        raise ValueError(msg)
    return extension[1:]


class JSONSetEncoder(json.JSONEncoder):
    """`~json.JSONEncoder` that supports `set` and `frozenset`.

    >>> import json
    >>> instance = {"val1": {1, 2, 3}, "val2": frozenset({2, 3, 4, 5})}
    >>> json.dumps(instance, cls=JSONSetEncoder)
    '{"val1": [1, 2, 3], "val2": [2, 3, 4, 5]}'
    """

    # https://stackoverflow.com/a/8230505
    def default(self, o: Any) -> Any:
        if isinstance(o, (frozenset, set)):
            return list(o)
        return super().default(o)

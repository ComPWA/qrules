from __future__ import annotations

import logging
import re
from collections import abc
from fractions import Fraction
from functools import singledispatch
from inspect import isfunction
from typing import TYPE_CHECKING, Any, Protocol

import attrs
from attrs import frozen

from qrules.particle import Particle, Spin, _render_fraction
from qrules.quantum_numbers import EdgeQuantumNumbers, InteractionProperties
from qrules.solving import (
    EdgeSettings,
    GraphEdgePropertyMap,
    NodeSettings,
    QNProblemSet,
)
from qrules.topology import (
    FrozenDict,
    FrozenTransition,
    MutableTransition,
    Topology,
    Transition,
)
from qrules.transition import ProblemSet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from typing_extensions import TypeIs

    from qrules.argument_handling import Rule

_LOGGER = logging.getLogger(__name__)

RenderedGraph = ProblemSet | QNProblemSet | Topology | Transition
RenderPair = tuple[Topology, RenderedGraph]
RenderInput = RenderedGraph | RenderPair

_TEXT_TO_LATEX_ESCAPES = str.maketrans({
    "\\": R"\textbackslash{}",
    "{": R"\{",
    "}": R"\}",
    "$": R"\$",
    "&": R"\&",
    "#": R"\#",
    "_": R"\_",
    "%": R"\%",
    "~": R"\textasciitilde{}",
    "^": R"\textasciicircum{}",
})
R"""Characters that are special within the argument of a LaTeX ``\text``.

Names that `qrules` does not control, such as particle names without a ``latex`` field
or quantum number keys, run through this table before ``._LatexFormatter.text``
interpolates them. The replacements are the `standard LaTeX escapes
<https://latexref.xyz/Printing-special-characters.html>`_, which are `all supported by
KaTeX <https://katex.org/docs/support_table>`_ as well, the renderer that ``._mermaid``
hands the result to.
"""


def is_render_pair(value: object, /) -> TypeIs[RenderPair]:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], Topology)
        and isinstance(value[1], (ProblemSet, QNProblemSet, Topology, Transition))
    )


def unpack_render_input(obj: RenderInput) -> RenderPair:
    """Split a render input into the topology and the graph that is rendered onto it.

    A `.RenderPair` carries a topology that may differ from the one embedded in the
    graph, so it is returned as-is. Any other graph renders onto its own topology.
    """
    if is_render_pair(obj):
        return obj
    if isinstance(obj, (ProblemSet, QNProblemSet, Transition)):
        return obj.topology, obj
    if isinstance(obj, Topology):
        return obj, obj
    msg = f"Cannot render a {type(obj).__name__} as a transition graph"
    raise NotImplementedError(msg)


def select_transitions(
    graphs: Iterable[Any],
    /,
    *,
    collapse: bool,
    render_node: bool | None,
    strip_spin: bool,
) -> list[Any]:
    """Reduce a collection of transitions to the graphs that are worth rendering.

    The ``collapse`` and ``strip_spin`` flags are the printer attributes
    :code:`collapse_graphs` and :code:`strip_spin`. Projections can only be stripped
    from the interaction nodes if those nodes are not rendered.
    """
    if collapse:
        return collapse_graphs(graphs)
    if strip_spin:
        if render_node:
            return sorted({strip_projections(g) for g in graphs})
        return get_particle_graphs(graphs)
    return list(graphs)


def create_edge_label(
    graph: ProblemSet | QNProblemSet | Topology | Transition,
    edge_id: int,
    render_edge_id: bool,
    *,
    render_label: Callable[[Any], str] | None = None,
) -> str:
    if render_label is None:
        render_label = as_string
    if isinstance(graph, Topology):
        if render_edge_id:
            return str(edge_id)
        return ""
    if isinstance(graph, (ProblemSet, QNProblemSet)):
        edge_setting = graph.solving_settings.states.get(edge_id)
        initial_fact = graph.initial_facts.states.get(edge_id)
        edge_property: EdgeSettings | GraphEdgePropertyMap | Particle | None = None
        if edge_setting:
            edge_property = edge_setting
        if initial_fact:
            edge_property = initial_fact
        return __render_edge_with_id(
            edge_id, edge_property, render_edge_id, render_label
        )
    edge_prop = graph.states.get(edge_id)
    return __render_edge_with_id(edge_id, edge_prop, render_edge_id, render_label)


def __render_edge_with_id(
    edge_id: int,
    edge_prop: Any,
    render_edge_id: bool,
    render_label: Callable[[Any], str],
) -> str:
    if edge_prop is None or not edge_prop:
        return str(edge_id)
    edge_label = render_label(edge_prop)
    if not render_edge_id:
        return edge_label
    if "\n" in edge_label:
        return f"{edge_id}:\n{edge_label}"
    return f"{edge_id}: {edge_label}"


@singledispatch
def as_string(obj: Any) -> str:
    """Render an edge or node property on a `.Transition` as a `str`.

    This function is decorated with :func:`functools.singledispatch`, which means that
    you can easily register other converter functions. An example:

    >>> from qrules.io._labels import as_string
    >>> as_string(10)
    '10'
    >>> _ = as_string.register(int, lambda _: "new int rendering")
    >>> as_string(10)
    'new int rendering'
    """
    if obj is not None:
        _LOGGER.warning(f"No label renderer implemented type {type(obj).__name__}")
    return str(obj)


@singledispatch
def as_latex(obj: Any) -> str:
    """Render an edge or node property as LaTeX source.

    Implementations can be registered for specific types through
    :func:`functools.singledispatch`. This allows graph renderers to request LaTeX
    labels independently of the plain-text labels produced by `.as_string`. If no
    specialized implementation exists, the object is converted to a `str` and a warning
    is emitted.
    """
    if obj is not None:
        _LOGGER.warning(
            f"No LaTeX label renderer implemented type {type(obj).__name__}"
        )
    return str(obj)


class _LabelFormatter(Protocol):
    def render(self, obj: Any) -> str: ...
    def text(self, value: str) -> str: ...
    def fraction(self, value: Fraction, *, plusminus: bool = False) -> str: ...
    def lines(self, values: list[str]) -> str: ...
    def domain(self, values: list[str]) -> str: ...
    def assignment(self, key: str, value: str, *, compact: bool = False) -> str: ...
    def membership(self, key: str, domain: str) -> str: ...
    def particle(self, name: str, latex: str | None) -> str: ...
    def spin(self, magnitude: str, projection: str) -> str: ...
    def superscript(self, base: str, exponent: str) -> str: ...


class _PlainFormatter:
    @staticmethod
    def render(obj: Any) -> str:
        return as_string(obj)

    @staticmethod
    def text(value: str) -> str:
        return str(value)

    @staticmethod
    def fraction(value: Fraction, *, plusminus: bool = False) -> str:
        return _render_fraction(value, plusminus=plusminus)

    @staticmethod
    def lines(values: list[str]) -> str:
        return "\n".join(values)

    @staticmethod
    def domain(values: list[str]) -> str:
        return f"[{', '.join(values)}]"

    @staticmethod
    def assignment(key: str, value: str, *, compact: bool = False) -> str:
        separator = "=" if compact else " = "
        return f"{key}{separator}{value}"

    @staticmethod
    def membership(key: str, domain: str) -> str:
        return f"{key} ∊ {domain}"

    @staticmethod
    def particle(name: str, latex: str | None) -> str:
        del latex
        return name

    @staticmethod
    def spin(magnitude: str, projection: str) -> str:
        return f"|{magnitude},{projection}⟩"

    @staticmethod
    def superscript(base: str, exponent: str) -> str:
        return base + exponent.translate(_SUPERSCRIPT_SIGNS)


class _LatexFormatter:
    @staticmethod
    def render(obj: Any) -> str:
        return as_latex(obj)

    @staticmethod
    def text(value: str) -> str:
        return Rf"\text{{{_escape_text_for_latex(value)}}}"

    @staticmethod
    def fraction(value: Fraction, *, plusminus: bool = False) -> str:
        return _render_latex_fraction(value, plusminus=plusminus)

    @staticmethod
    def lines(values: list[str]) -> str:
        return _render_latex_lines(values)

    @staticmethod
    def domain(values: list[str]) -> str:
        return R"\left[" + ", ".join(values) + R"\right]"

    @staticmethod
    def assignment(key: str, value: str, *, compact: bool = False) -> str:
        del compact
        return f"{key} = {value}"

    @staticmethod
    def membership(key: str, domain: str) -> str:
        return Rf"{key} \in {domain}"

    @staticmethod
    def particle(name: str, latex: str | None) -> str:
        if latex:
            return latex
        return _LatexFormatter.text(name)

    @staticmethod
    def spin(magnitude: str, projection: str) -> str:
        return Rf"\left|{magnitude},{projection}\right\rangle"

    @staticmethod
    def superscript(base: str, exponent: str) -> str:
        if not exponent:
            return base
        return f"{base}^{{{exponent}}}"


_SUPERSCRIPT_SIGNS = str.maketrans({"+": "⁺", "-": "⁻"})
_PLAIN_FORMATTER = _PlainFormatter()
_LATEX_FORMATTER = _LatexFormatter()
_PARTICLE_COLUMN_MAX_ROWS = 6


@as_latex.register(int)
@as_latex.register(float)
@as_latex.register(str)
@as_string.register(int)
@as_string.register(float)
@as_string.register(str)
def _(obj: Any) -> str:
    return str(obj)


@as_latex.register(Fraction)
def _(value: Fraction) -> str:
    return _render_latex_fraction(value)


@as_latex.register(type(None))
def _(_: None) -> str:
    return R"\mathrm{None}"


@as_string.register(dict)
def _(obj: dict) -> str:
    return __render_mapping(obj, _PLAIN_FORMATTER)


@as_latex.register(dict)
def _(obj: dict) -> str:
    return __render_mapping(obj, _LATEX_FORMATTER)


def __render_mapping(obj: dict, formatter: _LabelFormatter) -> str:
    lines: list[str] = []
    for key, value in obj.items():
        if isinstance(key, type) or callable(key):
            key_repr = key.__name__
        else:
            key_repr = str(key)
        if not value and not key_repr.endswith(("magnitude", "projection")):
            continue
        value_repr = __render_key_and_value(key_repr, value, formatter)
        lines.append(formatter.assignment(formatter.text(key_repr), value_repr))
    return formatter.lines(lines)


def __render_key_and_value(
    key: str,
    value: Any,
    formatter: _LabelFormatter = _PLAIN_FORMATTER,
) -> str:
    if isinstance(value, (Fraction, int)):
        fraction = Fraction(value)
        no_pm = key.endswith("magnitude") or key == "pid"
        return formatter.fraction(fraction, plusminus=not no_pm)
    return formatter.render(value)


def _render_latex_fraction(value: Fraction, *, plusminus: bool = False) -> str:
    sign = ""
    if value < 0:
        sign = R"\text{-}"
        value = abs(value)
    elif plusminus and value > 0:
        sign = R"\text{+}"
    if value.denominator == 1:
        return f"{sign}{value.numerator}"
    return Rf"{sign}\frac{{{value.numerator}}}{{{value.denominator}}}"


def _escape_text_for_latex(text: str) -> str:
    return str(text).translate(_TEXT_TO_LATEX_ESCAPES)


def _render_latex_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    content = R" \\ ".join(lines)
    return Rf"\begin{{gathered}} {content} \end{{gathered}}"


@as_string.register(InteractionProperties)
def _(obj: InteractionProperties) -> str:
    return __render_interaction(obj, _PLAIN_FORMATTER)


@as_latex.register(InteractionProperties)
def _(obj: InteractionProperties) -> str:
    return __render_interaction(obj, _LATEX_FORMATTER)


def __render_interaction(obj: InteractionProperties, formatter: _LabelFormatter) -> str:
    lines: list[str] = []
    if obj.l_magnitude is not None:
        if obj.l_projection is None:
            l_label = formatter.fraction(Fraction(obj.l_magnitude))
        else:
            l_label = formatter.render(Spin(obj.l_magnitude, obj.l_projection))
        lines.append(formatter.assignment("L", l_label, compact=True))
    if obj.s_magnitude is not None:
        if obj.s_projection is None:
            s_label = formatter.fraction(Fraction(obj.s_magnitude))
        else:
            s_label = formatter.render(Spin(obj.s_magnitude, obj.s_projection))
        lines.append(formatter.assignment("S", s_label, compact=True))
    if obj.parity_prefactor is not None:
        label = formatter.fraction(Fraction(obj.parity_prefactor), plusminus=True)
        lines.append(formatter.assignment("P", label, compact=True))
    return formatter.lines(lines)


@as_string.register(EdgeSettings)
@as_string.register(NodeSettings)
def _(settings: EdgeSettings | NodeSettings) -> str:
    return __render_settings(settings, _PLAIN_FORMATTER)


@as_latex.register(EdgeSettings)
@as_latex.register(NodeSettings)
def _(settings: EdgeSettings | NodeSettings) -> str:
    return __render_settings(settings, _LATEX_FORMATTER)


def __render_settings(
    settings: EdgeSettings | NodeSettings, formatter: _LabelFormatter
) -> str:
    lines: list[str] = []
    if settings.conservation_rules:
        lines.append(formatter.text("RULES"))
        rule_descriptions = (
            f"{__render_rule(rule)} - {priority}"
            for rule, priority in settings.conservation_rules.items()
        )
        sorted_names = sorted(rule_descriptions, key=__extract_priority, reverse=True)
        lines.extend(formatter.text(name) for name in sorted_names)
    if settings.qn_domains:
        lines.append(formatter.text("DOMAINS"))
        domains = sorted(
            formatter.membership(
                formatter.text(qn.__name__),
                __render_domain(domain, key=qn.__name__, formatter=formatter),
            )
            for qn, domain in settings.qn_domains.items()
        )
        lines.extend(domains)
    return formatter.lines(lines)


def __render_rule(rule: Rule) -> str:
    return __get_type(rule).__name__


def __get_type(rule: Rule, /) -> type[Rule]:
    if isfunction(rule):
        return rule  # ty: ignore[invalid-return-type]
    return type(rule)


def __extract_priority(description: str) -> int | float:
    """Get the priority from a rule description, as rendered by `.as_string`.

    Rules without a priority (``"NA"``) rank below any numeric priority.

    >>> descriptions = ["a - 9", "b - 10", "c - NA", "d - -1"]
    >>> sorted(descriptions, key=__extract_priority, reverse=True)
    ['b - 10', 'a - 9', 'd - -1', 'c - NA']
    """
    matches = re.match(r".* - (-?[0-9]+|NA)$", description)
    if matches is None:
        msg = f"{description} does not contain a priority number"
        raise ValueError(msg)
    priority = matches[1]
    if priority == "NA":
        return float("-inf")
    return int(priority)


def __render_domain(
    domain: list[Any],
    key: str,
    formatter: _LabelFormatter = _PLAIN_FORMATTER,
) -> str:
    """Render a domain as a `str`.

    >>> half = Fraction(0.5)
    >>> __render_domain([-half, +half], key="spin_projection")
    '[-1/2, +1/2]'
    >>> __render_domain([0, 1], key="l_magnitude")
    '[0, 1]'
    >>> __render_domain([None, +1, -1], key="parity")
    '[-1, +1, None]'
    """
    domain = sorted(domain, key=lambda x: +9999 if x is None else x)
    domain_str = [__render_key_and_value(key, x, formatter) for x in domain]
    return formatter.domain(domain_str)


@as_string.register(Particle)
def _(particle: Particle) -> str:
    return __render_particle(particle, _PLAIN_FORMATTER)


@as_latex.register(Particle)
def _(particle: Particle) -> str:
    return __render_particle(particle, _LATEX_FORMATTER)


def __render_particle(particle: Particle, formatter: _LabelFormatter) -> str:
    return formatter.particle(particle.name, particle.latex)


@as_string.register(Spin)
def _(spin: Spin) -> str:
    return __render_spin(spin, _PLAIN_FORMATTER)


@as_latex.register(Spin)
def _(spin: Spin) -> str:
    return __render_spin(spin, _LATEX_FORMATTER)


def __render_spin(spin: Spin, formatter: _LabelFormatter) -> str:
    spin_magnitude = formatter.fraction(spin.magnitude)
    spin_projection = formatter.fraction(spin.projection, plusminus=True)
    return formatter.spin(spin_magnitude, spin_projection)


@frozen
class QuantumNumberSignature:
    """PDG-style :math:`I^G(J^{PC})` summary of a quantum-number property map.

    `collapse_graphs` converts states that are quantum-number property maps to this
    compact form, so that a collapsed edge lists signatures instead of complete maps.
    The :math:`C`-parity is only taken over for self-conjugate states (zero charge,
    baryon number, and strangeness) and the :math:`G`-parity only for nonstrange
    non-baryonic states, since the quantum numbers are undefined otherwise.

    >>> from qrules.quantum_numbers import EdgeQuantumNumbers as EQN
    >>> signature = QuantumNumberSignature.from_property_map({
    ...     EQN.spin_magnitude: 1,
    ...     EQN.parity: -1,
    ...     EQN.c_parity: -1,
    ...     EQN.isospin_magnitude: 1,
    ...     EQN.g_parity: +1,
    ... })
    >>> as_string(signature)
    '1⁺(1⁻⁻)'
    >>> as_latex(signature)
    '1^{+}(1^{--})'
    >>> as_string(QuantumNumberSignature.from_property_map({EQN.spin_magnitude: 0.5}))
    '1/2'
    >>> baryon = QuantumNumberSignature.from_property_map({
    ...     EQN.spin_magnitude: 1.5,
    ...     EQN.parity: +1,
    ...     EQN.c_parity: -1,
    ...     EQN.isospin_magnitude: 1.5,
    ...     EQN.g_parity: +1,
    ...     EQN.baryon_number: 1,
    ... })
    >>> as_string(baryon)
    '3/2(3/2⁺)'
    """

    spin_magnitude: Fraction | None = None
    parity: int | None = None
    c_parity: int | None = None
    isospin_magnitude: Fraction | None = None
    g_parity: int | None = None

    @classmethod
    def from_property_map(cls, qn_map: Mapping[Any, Any]) -> QuantumNumberSignature:
        baryon_number = qn_map.get(EdgeQuantumNumbers.baryon_number) or 0
        charge = qn_map.get(EdgeQuantumNumbers.charge) or 0
        strangeness = qn_map.get(EdgeQuantumNumbers.strangeness) or 0
        c_parity = None
        if baryon_number == 0 and charge == 0 and strangeness == 0:
            c_parity = _to_optional_int(qn_map.get(EdgeQuantumNumbers.c_parity))
        g_parity = None
        if baryon_number == 0 and strangeness == 0:
            g_parity = _to_optional_int(qn_map.get(EdgeQuantumNumbers.g_parity))
        return cls(
            spin_magnitude=_to_optional_fraction(
                qn_map.get(EdgeQuantumNumbers.spin_magnitude)
            ),
            parity=_to_optional_int(qn_map.get(EdgeQuantumNumbers.parity)),
            c_parity=c_parity,
            isospin_magnitude=_to_optional_fraction(
                qn_map.get(EdgeQuantumNumbers.isospin_magnitude)
            ),
            g_parity=g_parity,
        )


def _to_optional_fraction(value: Any) -> Fraction | None:
    if value is None:
        return None
    return Fraction(value)


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@as_string.register(QuantumNumberSignature)
def _(signature: QuantumNumberSignature) -> str:
    return __render_signature(signature, _PLAIN_FORMATTER)


@as_latex.register(QuantumNumberSignature)
def _(signature: QuantumNumberSignature) -> str:
    return __render_signature(signature, _LATEX_FORMATTER)


def __render_signature(
    signature: QuantumNumberSignature, formatter: _LabelFormatter
) -> str:
    if signature.spin_magnitude is None:
        spin = formatter.text("?")
    else:
        spin = formatter.fraction(signature.spin_magnitude)
    jpc = formatter.superscript(
        spin, __render_parity_signs(signature.parity, signature.c_parity)
    )
    if signature.isospin_magnitude is None:
        return jpc
    ig = formatter.superscript(
        formatter.fraction(signature.isospin_magnitude),
        __render_parity_signs(signature.g_parity),
    )
    return f"{ig}({jpc})"


def __render_parity_signs(*parities: int | None) -> str:
    return "".join("+" if p > 0 else "-" for p in parities if p is not None)


@as_string.register(tuple)
def _(obj: tuple) -> str:
    return __render_tuple(obj, _PLAIN_FORMATTER)


@as_latex.register(tuple)
def _(obj: tuple) -> str:
    return __render_tuple(obj, _LATEX_FORMATTER)


def __render_tuple(obj: tuple, formatter: _LabelFormatter) -> str:
    if len(obj) == 2 and all(isinstance(o, (Fraction, float, int)) for o in obj):
        return __render_spin(Spin(*obj), formatter)
    rendered_items = [formatter.render(item) for item in obj]
    if (
        formatter is _LATEX_FORMATTER
        and len(obj) > _PARTICLE_COLUMN_MAX_ROWS
        and all(isinstance(item, Particle) for item in obj)
    ):
        return _render_latex_columns(rendered_items)
    return formatter.lines(rendered_items)


def _render_latex_columns(items: list[str]) -> str:
    column_count = (len(items) + _PARTICLE_COLUMN_MAX_ROWS - 1) // (
        _PARTICLE_COLUMN_MAX_ROWS
    )
    row_count = (len(items) + column_count - 1) // column_count
    columns = [items[i * row_count : (i + 1) * row_count] for i in range(column_count)]
    rows = [
        " & ".join(column[i] if i < len(column) else "" for column in columns)
        for i in range(row_count)
    ]
    content = R" \\ ".join(rows)
    alignment = "l" * column_count
    return Rf"\begin{{array}}{{{alignment}}} {content} \end{{array}}"


def get_particle_graphs(
    graphs: Iterable[Transition[Particle, InteractionProperties]],
) -> list[FrozenTransition[Particle, None]]:
    """Strip `list` of `.Transition` s of their interaction properties.

    Extract a `list` of unique `.Transition` instances with only `.Particle` instances
    on the edges and no interaction properties.

    .. seealso:: :doc:`/usage/visualize`
    """
    inventory = set()
    for transition in graphs:
        stripped_transition = strip_projections(transition)
        topology = stripped_transition.topology
        particle_transition: FrozenTransition[Particle, None] = FrozenTransition(
            stripped_transition.topology,
            states=stripped_transition.states,
            interactions=dict.fromkeys(topology.nodes),
        )
        inventory.add(particle_transition)
    return sorted(
        inventory,
        key=lambda g: [g.states[i].mass for i in g.topology.intermediate_edge_ids],
    )


def strip_projections(
    graph: Transition[Any, InteractionProperties],
) -> FrozenTransition[Particle, InteractionProperties]:
    transition = FrozenTransition(graph.topology, graph.states, graph.interactions)
    return transition.convert(
        state_converter=__to_particle,
        interaction_converter=lambda i: attrs.evolve(
            i, l_projection=None, s_projection=None
        ),
    )


def __to_particle(state: Any) -> Particle:
    if isinstance(state, Particle):
        return state
    msg = f"Cannot extract a particle from type {type(state).__name__}"
    raise NotImplementedError(msg)


def collapse_graphs(
    graphs: Iterable[Transition[Any, Any]],
) -> list[FrozenTransition[tuple, None]]:
    graphs = list(graphs)
    transition_groups: dict[Topology, MutableTransition[set[Particle], None]] = {
        g.topology: MutableTransition(
            g.topology,
            states={i: set() for i in g.topology.edges},
            interactions=dict.fromkeys(g.topology.nodes),
        )
        for g in graphs
    }
    for transition in graphs:
        topology = transition.topology
        group = transition_groups[topology]
        for state_id, state in transition.states.items():
            group.states[state_id].add(_strip_properties(state))
    collected_graphs: list[FrozenTransition[tuple[Particle, ...], None]] = []
    for topology in sorted(transition_groups):
        group = transition_groups[topology]
        collected_graphs.append(
            FrozenTransition(
                topology,
                states={
                    i: tuple(sorted(_summarize_property_maps(states), key=_sorting_key))
                    for i, states in group.states.items()
                },
                interactions=group.interactions,
            )
        )
    return collected_graphs


def _strip_properties(state: Any) -> Any:
    if isinstance(state, abc.Mapping):
        return FrozenDict(state)
    return state


def _summarize_property_maps(states: Iterable[Any]) -> set[Any]:
    """Replace collapsed quantum-number property maps by unique signatures.

    Property maps in which some quantum numbers are merely unassigned (`None`) are
    dropped when a more determined map with the same assigned values is present, so
    that the collapsed edge label stays minimal.
    """
    property_maps = [state for state in states if isinstance(state, abc.Mapping)]
    summarized: set[Any] = {
        state for state in states if not isinstance(state, abc.Mapping)
    }
    assigned_values = [
        {key: value for key, value in qn_map.items() if value is not None}
        for qn_map in property_maps
    ]
    for qn_map, assigned in zip(property_maps, assigned_values, strict=True):
        is_subsumed = any(assigned.items() < other.items() for other in assigned_values)
        if not is_subsumed:
            summarized.add(QuantumNumberSignature.from_property_map(qn_map))
    return summarized


def _sorting_key(obj: Any) -> Any:
    if isinstance(obj, QuantumNumberSignature):
        return as_string(obj)
    if isinstance(obj, str):
        return obj.lower()
    return obj

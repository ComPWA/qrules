from __future__ import annotations

import logging
import re
from fractions import Fraction
from functools import singledispatch
from inspect import isfunction
from typing import TYPE_CHECKING, Any, Protocol

import attrs

from qrules.particle import Particle, ParticleWithSpin, Spin, _render_fraction
from qrules.quantum_numbers import InteractionProperties
from qrules.solving import (
    EdgeSettings,
    GraphEdgePropertyMap,
    NodeSettings,
    QNProblemSet,
)
from qrules.topology import FrozenTransition, MutableTransition, Topology, Transition
from qrules.transition import ProblemSet, State

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from typing_extensions import TypeIs

    from qrules.argument_handling import Rule, RuleKey

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
        edge_property: EdgeSettings | GraphEdgePropertyMap | ParticleWithSpin | None = (
            None
        )
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
    def state(self, particle: str, projection: str) -> str: ...


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
    def state(particle: str, projection: str) -> str:
        return f"{particle}[{projection}]"


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
    def state(particle: str, projection: str) -> str:
        return Rf"{particle}\left[{projection}\right]"


_PLAIN_FORMATTER = _PlainFormatter()
_LATEX_FORMATTER = _LatexFormatter()
_PARTICLE_COLUMN_THRESHOLD = 6


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
    if settings.rule_priorities:
        lines.append(formatter.text("RULES"))
        rule_descriptions = (
            f"{__render_rule(rule)} - {__get_priority(rule, settings.rule_priorities)}"
            for rule in settings.conservation_rules
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


def __get_priority(rule: Rule, rule_priorities: dict[RuleKey, int]) -> int | str:
    rule_type = __get_type(rule)
    return rule_priorities.get(rule_type, "NA")


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


@as_string.register(State)
def _(state: State) -> str:
    return __render_state(state, _PLAIN_FORMATTER)


@as_latex.register(State)
def _(state: State) -> str:
    return __render_state(state, _LATEX_FORMATTER)


def __render_state(state: State, formatter: _LabelFormatter) -> str:
    particle = formatter.render(state.particle)
    spin_projection = formatter.fraction(state.spin_projection, plusminus=True)
    return formatter.state(particle, spin_projection)


@as_string.register(tuple)
def _(obj: tuple) -> str:
    return __render_tuple(obj, _PLAIN_FORMATTER)


@as_latex.register(tuple)
def _(obj: tuple) -> str:
    return __render_tuple(obj, _LATEX_FORMATTER)


def __render_tuple(obj: tuple, formatter: _LabelFormatter) -> str:
    if len(obj) == 2:
        if isinstance(obj[0], Particle) and isinstance(obj[1], (Fraction, float, int)):
            return __render_state(State(*obj), formatter)
        if all(isinstance(o, (Fraction, float, int)) for o in obj):
            return __render_spin(Spin(*obj), formatter)
    rendered_items = [formatter.render(item) for item in obj]
    if (
        formatter is _LATEX_FORMATTER
        and len(obj) > _PARTICLE_COLUMN_THRESHOLD
        and all(isinstance(item, Particle) for item in obj)
    ):
        return _render_latex_columns(rendered_items)
    return formatter.lines(rendered_items)


def _render_latex_columns(items: list[str]) -> str:
    row_count = (len(items) + 1) // 2
    first_column = items[:row_count]
    second_column = items[row_count:]
    rows = [
        f"{first} & {second_column[i] if i < len(second_column) else ''}"
        for i, first in enumerate(first_column)
    ]
    content = R" \\ ".join(rows)
    return Rf"\begin{{array}}{{ll}} {content} \end{{array}}"


def get_particle_graphs(
    graphs: Iterable[Transition[ParticleWithSpin, InteractionProperties]],
) -> list[FrozenTransition[Particle, None]]:
    """Strip `list` of `.Transition` s of the spin projections.

    Extract a `list` of `.Transition` instances with only `.Particle` instances on the
    edges.

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
    if isinstance(state, State):
        return state.particle
    if isinstance(state, tuple) and len(state) == 2:
        return state[0]
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
                    i: tuple(sorted(particles, key=_sorting_key))
                    for i, particles in group.states.items()
                },
                interactions=group.interactions,
            )
        )
    return collected_graphs


def _strip_properties(state: Any) -> Any:
    if isinstance(state, State):
        return state.particle
    return state


def _sorting_key(obj: Any) -> Any:
    if isinstance(obj, State):
        return obj.particle.name
    if isinstance(obj, str):
        return obj.lower()
    return obj

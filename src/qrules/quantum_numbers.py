"""Definitions used internally for type hints and signatures.

`qrules` is strictly typed (enforced through :doc:`mypy <mypy:index>`). This module
bundles structures and definitions that don't serve as data containers but only as type
hints. `.EdgeQuantumNumbers` and `.NodeQuantumNumbers` are the main structures and serve
as a bridge between the :mod:`.particle` and the :mod:`.conservation_rules` module.
"""

from __future__ import annotations

from fractions import Fraction
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Literal, NewType, Union

from attrs import field, frozen

from qrules._implementers import implement_pretty_repr

if TYPE_CHECKING:
    from collections.abc import Generator


def _to_parity(value: int) -> Literal[-1, 1]:
    if not isinstance(value, int):
        msg = f"Parity must be an integer, not {type(value)}"
        raise TypeError(msg)
    if value == -1:
        return -1
    if value == +1:
        return 1
    msg = f"Parity can only be +1 or -1, not {value}"
    raise ValueError(msg)


@total_ordering
@frozen(eq=False, hash=True, order=False, repr=False)
class Parity:  # noqa: PLW1641
    value: Literal[-1, 1] = field(converter=_to_parity)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parity):
            return self.value == other.value
        return self.value == other

    def __gt__(self, other: Any) -> bool:
        if other is None:
            return True
        return self.value > int(other)

    def __int__(self) -> Literal[-1, 1]:
        return self.value

    def __neg__(self) -> Parity:
        return Parity(-self.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({_float_as_signed_str(self.value)})"


def _float_as_signed_str(value: float, render_plus: bool = False) -> str:
    if value > 0 or render_plus:
        return f"+{value}"
    return str(value)


@frozen(init=False)
class EdgeQuantumNumbers:
    """Definition of quantum numbers for edges.

    This class defines the types that are used in the :mod:`.conservation_rules`, for
    instance in `.additive_quantum_number_rule`. You can also create data classes (see
    :func:`attrs.define`) with data members that are typed as the data members of
    `.EdgeQuantumNumbers` (see for example `.HelicityParityEdgeInput`) and use them in
    conservation rules that satisfy the appropriate rule protocol (see
    `.ConservationRule`, `.EdgeQNConservationRule`).
    """

    pid = NewType("pid", int)
    mass = NewType("mass", float)
    width = NewType("width", float)
    spin_magnitude = NewType("spin_magnitude", Fraction)
    spin_projection = NewType("spin_projection", Fraction)
    charge = NewType("charge", int)
    isospin_magnitude = NewType("isospin_magnitude", Fraction)
    isospin_projection = NewType("isospin_projection", Fraction)
    strangeness = NewType("strangeness", int)
    charmness = NewType("charmness", int)
    bottomness = NewType("bottomness", int)
    topness = NewType("topness", int)
    baryon_number = NewType("baryon_number", int)
    electron_lepton_number = NewType("electron_lepton_number", int)
    muon_lepton_number = NewType("muon_lepton_number", int)
    tau_lepton_number = NewType("tau_lepton_number", int)
    parity = NewType("parity", Parity)
    c_parity = NewType("c_parity", Parity)
    g_parity = NewType("g_parity", Parity)


for edge_qn_name, edge_qn_type in EdgeQuantumNumbers.__dict__.items():
    if not edge_qn_name.startswith("__"):
        edge_qn_type.__qualname__ = f"EdgeQuantumNumbers.{edge_qn_name}"
        edge_qn_type.__module__ = __name__


EdgeQuantumNumber = Union[
    EdgeQuantumNumbers.pid,
    EdgeQuantumNumbers.mass,
    EdgeQuantumNumbers.width,
    EdgeQuantumNumbers.spin_magnitude,
    EdgeQuantumNumbers.spin_projection,
    EdgeQuantumNumbers.charge,
    EdgeQuantumNumbers.isospin_magnitude,
    EdgeQuantumNumbers.isospin_projection,
    EdgeQuantumNumbers.strangeness,
    EdgeQuantumNumbers.charmness,
    EdgeQuantumNumbers.bottomness,
    EdgeQuantumNumbers.topness,
    EdgeQuantumNumbers.baryon_number,
    EdgeQuantumNumbers.electron_lepton_number,
    EdgeQuantumNumbers.muon_lepton_number,
    EdgeQuantumNumbers.tau_lepton_number,
    EdgeQuantumNumbers.parity,
    EdgeQuantumNumbers.c_parity,
    EdgeQuantumNumbers.g_parity,
]
"""Type hint for quantum numbers of edges"""

EdgeQuantumNumberTypes = Union[
    type[EdgeQuantumNumbers.pid],
    type[EdgeQuantumNumbers.mass],
    type[EdgeQuantumNumbers.width],
    type[EdgeQuantumNumbers.spin_magnitude],
    type[EdgeQuantumNumbers.spin_projection],
    type[EdgeQuantumNumbers.charge],
    type[EdgeQuantumNumbers.isospin_magnitude],
    type[EdgeQuantumNumbers.isospin_projection],
    type[EdgeQuantumNumbers.strangeness],
    type[EdgeQuantumNumbers.charmness],
    type[EdgeQuantumNumbers.bottomness],
    type[EdgeQuantumNumbers.topness],
    type[EdgeQuantumNumbers.baryon_number],
    type[EdgeQuantumNumbers.electron_lepton_number],
    type[EdgeQuantumNumbers.muon_lepton_number],
    type[EdgeQuantumNumbers.tau_lepton_number],
    type[EdgeQuantumNumbers.parity],
    type[EdgeQuantumNumbers.c_parity],
    type[EdgeQuantumNumbers.g_parity],
]
"""Type-Union for accessing the keys of the dicts in `.EdgeSettings`"""


@frozen(init=False)
class NodeQuantumNumbers:
    """Definition of quantum numbers for interaction nodes."""

    l_magnitude = NewType("l_magnitude", Fraction)
    l_projection = NewType("l_projection", Fraction)
    s_magnitude = NewType("s_magnitude", Fraction)
    s_projection = NewType("s_projection", Fraction)
    parity_prefactor = NewType("parity_prefactor", float)


for node_qn_name, node_qn_type in NodeQuantumNumbers.__dict__.items():
    if not node_qn_name.startswith("__"):
        node_qn_type.__qualname__ = f"NodeQuantumNumbers.{node_qn_name}"
        node_qn_type.__module__ = __name__


# for static typing
NodeQuantumNumber = Union[
    NodeQuantumNumbers.l_magnitude,
    NodeQuantumNumbers.l_projection,
    NodeQuantumNumbers.s_magnitude,
    NodeQuantumNumbers.s_projection,
    NodeQuantumNumbers.parity_prefactor,
]
"""Type hint for quantum numbers of interaction nodes."""

# for accessing the keys of the dicts in NodeSettings
NodeQuantumNumberTypes = Union[
    type[NodeQuantumNumbers.l_magnitude],
    type[NodeQuantumNumbers.l_projection],
    type[NodeQuantumNumbers.s_magnitude],
    type[NodeQuantumNumbers.s_projection],
    type[NodeQuantumNumbers.parity_prefactor],
]
"""Type-Union for accessing the keys of the dicts in `.NodeSettings`"""


def _to_optional_float(optional_float: float | None) -> float | None:
    if optional_float is None:
        return None
    return float(optional_float)


def _to_optional_fraction(optional_fraction: Fraction | None) -> Fraction | None:
    if optional_fraction is None:
        return None
    return Fraction(optional_fraction)


def _to_optional_int(optional_int: int | None) -> int | None:
    if optional_int is None:
        return None
    return int(optional_int)


@implement_pretty_repr
@frozen(order=True)
class InteractionProperties:
    """Immutable data structure containing interaction properties.

    Interactions are represented by a node on a `.MutableTransition`. This class
    represents the properties that are carried collectively by the edges that this node
    connects.

    Interaction properties are in particular important in the canonical basis of the
    helicity formalism. There, the *coupled spin* and angular momentum of each
    interaction are used for the Clebsch-Gordan coefficients for each term in a
    sequential amplitude.

    .. note:: As opposed to `NodeQuantumNumbers`, the `InteractionProperties` class
        serves as an interface to the user.
    """

    l_magnitude: int | None = field(  # L cannot be half integer
        default=None, converter=_to_optional_int
    )
    l_projection: int | None = field(default=None, converter=_to_optional_int)
    s_magnitude: Fraction | None = field(default=None, converter=_to_optional_fraction)
    s_projection: Fraction | None = field(default=None, converter=_to_optional_fraction)
    parity_prefactor: float | None = field(default=None, converter=_to_optional_float)


def arange(
    x_1: Fraction, x_2: Fraction, delta: Fraction = Fraction(1)
) -> Generator[Fraction, None, None]:
    current = Fraction(x_1)
    delta = Fraction(delta)
    while current < x_2:
        yield current
        current += delta

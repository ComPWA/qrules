"""Definitions used internally for type hints and signatures.

`qrules` is strictly typed. This module bundles structures and definitions that don't
serve as data containers but only as type hints. `.EdgeQuantumNumbers` and
`.NodeQuantumNumbers` are the main structures and serve as a bridge between the
:mod:`.particle` and the :mod:`.conservation_rules` module.
"""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from functools import total_ordering
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NewType, Protocol, TypeVar

from attrs import field, frozen

from qrules._implementers import implement_pretty_repr

if TYPE_CHECKING:
    from collections.abc import Generator


class _Unknown(Enum):
    """Sentinel type for a `Parity` that exists but whose value is not known.

    An `~enum.Enum` with a single member is the typed sentinel pattern recommended by
    :PEP:`484#support-for-singleton-types-in-unions`. Unlike a bare `object`, its
    identity survives `copy.deepcopy` and `pickle`, it has a readable `repr`, and type
    checkers can narrow on it. Once Python 3.15 is the minimal supported version, it can
    be replaced by the builtin ``sentinel()`` from :PEP:`661`.
    """

    UNKNOWN = "?"

    def __repr__(self) -> str:
        return "UNKNOWN"


_UNKNOWN = _Unknown.UNKNOWN


def _to_parity_value(
    value: int | Parity | _Unknown,
) -> Literal[-1, 1, _Unknown.UNKNOWN]:
    if isinstance(value, Parity):
        return value.value
    if value is _UNKNOWN:
        return _UNKNOWN
    if not isinstance(value, int):
        msg = f"Parity must be an integer, not {type(value)}"
        raise TypeError(msg)
    if value == -1:
        return -1
    if value == +1:
        return 1
    msg = f"Parity can only be +1, -1, or unknown, not {value}"
    raise ValueError(msg)


@total_ordering
@frozen(eq=False, hash=True, order=False, repr=False)
class Parity:  # ruff: ignore[eq-without-hash]
    """Intrinsic parity that is either :math:`+1`, :math:`-1`, or unknown.

    Use `Parity.UNKNOWN` for a parity that exists, but whose value is not (yet) known.
    This is different from `None`, which expresses that a parity (like C-parity or
    G-parity) is not defined for a particle at all.

    >>> Parity(+1)
    Parity(+1)
    >>> -Parity(-1)
    Parity(+1)
    >>> Parity.UNKNOWN
    Parity.UNKNOWN
    >>> -Parity.UNKNOWN
    Parity.UNKNOWN
    """

    value: Literal[-1, 1, _Unknown.UNKNOWN] = field(converter=_to_parity_value)
    UNKNOWN: ClassVar[Parity]
    """A parity that exists, but whose value is unknown."""

    @property
    def is_unknown(self) -> bool:
        return self.value is _UNKNOWN

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parity):
            return self.value == other.value
        return self.value == other

    def __gt__(self, other: Any) -> bool:
        # Total order for sorting: None < UNKNOWN < -1 < +1
        if other is None:
            return True
        if isinstance(other, Parity):
            return self.__sort_key() > other.__sort_key()
        return int(self) > int(other)

    def __sort_key(self) -> int:
        if self.value is _UNKNOWN:
            return -2
        return self.value

    def __int__(self) -> Literal[-1, 1]:
        if self.value is _UNKNOWN:
            msg = "Cannot convert an unknown parity to int"
            raise ValueError(msg)
        return self.value

    def __neg__(self) -> Parity:
        if self.value is _UNKNOWN:
            return self
        return Parity(-self.value)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if self.value is _UNKNOWN:
            return f"{class_name}.UNKNOWN"
        return f"{class_name}({_float_as_signed_str(self.value)})"


Parity.UNKNOWN = Parity(_UNKNOWN)


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


EdgeQuantumNumber = (
    EdgeQuantumNumbers.pid
    | EdgeQuantumNumbers.mass
    | EdgeQuantumNumbers.width
    | EdgeQuantumNumbers.spin_magnitude
    | EdgeQuantumNumbers.spin_projection
    | EdgeQuantumNumbers.charge
    | EdgeQuantumNumbers.isospin_magnitude
    | EdgeQuantumNumbers.isospin_projection
    | EdgeQuantumNumbers.strangeness
    | EdgeQuantumNumbers.charmness
    | EdgeQuantumNumbers.bottomness
    | EdgeQuantumNumbers.topness
    | EdgeQuantumNumbers.baryon_number
    | EdgeQuantumNumbers.electron_lepton_number
    | EdgeQuantumNumbers.muon_lepton_number
    | EdgeQuantumNumbers.tau_lepton_number
    | EdgeQuantumNumbers.parity
    | EdgeQuantumNumbers.c_parity
    | EdgeQuantumNumbers.g_parity
)
"""Type hint for quantum numbers of edges"""

_QuantumNumber_co = TypeVar("_QuantumNumber_co", covariant=True)


class QuantumNumberType(Protocol[_QuantumNumber_co]):
    """Runtime factory object created by `typing.NewType`."""

    __name__: str

    def __call__(self, value: Any, /) -> _QuantumNumber_co: ...


EdgeQuantumNumberTypes = QuantumNumberType[EdgeQuantumNumber]
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
NodeQuantumNumber = (
    NodeQuantumNumbers.l_magnitude
    | NodeQuantumNumbers.l_projection
    | NodeQuantumNumbers.s_magnitude
    | NodeQuantumNumbers.s_projection
    | NodeQuantumNumbers.parity_prefactor
)
"""Type hint for quantum numbers of interaction nodes."""

# for accessing the keys of the dicts in NodeSettings
NodeQuantumNumberTypes = QuantumNumberType[NodeQuantumNumber]
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
    interaction are used for the Clebsch–Gordan coefficients for each term in a
    sequential amplitude.

    .. note::

        As opposed to `NodeQuantumNumbers`, the `InteractionProperties` class serves as
        an interface to the user.
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

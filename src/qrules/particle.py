"""A collection of particle info containers.

The :mod:`.particle` module is the starting point of `qrules`. Its main interface is the
`ParticleCollection`, which is a collection of immutable `Particle` instances that are
uniquely defined by their properties. As such, it can be used stand-alone as a database
of quantum numbers (see :doc:`/usage/particle`).

The `.transition` module uses the properties of `Particle` instances when it computes
which `.MutableTransition` s are allowed between an initial state and final state.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import abc
from difflib import get_close_matches
from fractions import Fraction
from functools import total_ordering
from typing import TYPE_CHECKING, Any

import attrs
from attrs import field, frozen
from attrs.converters import optional
from attrs.validators import instance_of

from qrules._attrs import to_fraction, to_parity
from qrules.conservation_rules import GellMannNishijimaInput, gellmann_nishijima
from qrules.quantum_numbers import Parity, _float_as_signed_str

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from attrs import Attribute
    from IPython.lib.pretty import RepresentationPrinter

_LOGGER = logging.getLogger(__name__)


def _validate_fraction_for_spin(
    instance: Spin,
    attribute: Attribute,  # ruff: ignore[unused-function-argument]
    value: Fraction,  # ruff: ignore[unused-function-argument]
) -> Any:
    if instance.magnitude % Fraction(1, 2) != Fraction(0):
        msg = f"Spin magnitude {instance.magnitude} has to be a multitude of 0.5"
        raise ValueError(msg)
    if abs(instance.projection) > instance.magnitude:
        if instance.magnitude < Fraction(0):
            msg = f"Spin magnitude has to be positive, but is {instance.magnitude}"
            raise ValueError(msg)
        msg = (
            "Absolute value of spin projection cannot be larger than its"
            f" magnitude:\n abs({instance.projection}) > {instance.magnitude}"
        )
        raise ValueError(msg)
    if (instance.projection - instance.magnitude).denominator != 1:
        msg = (
            f"{type(instance).__name__}{(instance.magnitude, instance.projection)}: (projection -"
            " magnitude) should be integer"
        )
        raise ValueError(msg)


@total_ordering
@frozen(eq=False, hash=True, order=False)
class Spin:  # ruff: ignore[eq-without-hash]
    """Safe, immutable data container for spin **with projection**."""

    magnitude: Fraction = field(
        converter=to_fraction,
        validator=_validate_fraction_for_spin,
    )
    projection: Fraction = field(
        converter=to_fraction,
        validator=_validate_fraction_for_spin,
    )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Spin):
            return (
                self.magnitude == other.magnitude
                and self.projection == other.projection
            )
        return self.magnitude == other

    def __float__(self) -> float:
        return float(self.magnitude)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Spin):
            return attrs.astuple(self) > attrs.astuple(other)
        return self.magnitude > other

    def __neg__(self) -> Spin:
        return Spin(self.magnitude, -self.projection)

    def __repr__(self) -> str:
        return f"{type(self).__name__}{(self.magnitude, self.projection)}"

    def _repr_pretty_(self, p: RepresentationPrinter, _: bool) -> None:
        class_name = type(self).__name__
        magnitude = _render_fraction(self.magnitude)
        projection = _render_fraction(self.projection, plusminus=True)
        p.text(f"{class_name}({magnitude}, {projection})")


def _render_fraction(fraction: Fraction, plusminus: bool = False) -> str:
    if plusminus and fraction.numerator > 0:
        return f"+{fraction}"
    return str(fraction)


def _to_spin(value: Spin | tuple[Fraction, Fraction] | tuple[float, float], /) -> Spin:
    if isinstance(value, tuple):
        magnitude, projection = value
        return Spin(magnitude, projection)
    return value


@total_ordering
@frozen(kw_only=True, order=False, repr=True)
class Particle:
    """Immutable container of data defining a physical particle.

    A `Particle` is defined by the minimum set of the quantum numbers that every
    possible instances of that particle have in common (the "static" quantum numbers of
    the particle). A "non-static" quantum number is the spin projection. Hence
    `Particle` instances do **not** contain spin projection information.

    `Particle` instances are uniquely defined by their quantum numbers and properties
    like `~Particle.mass`. The `~Particle.name` and `~Particle.pid` are therefore just
    labels that are not taken into account when checking if two `Particle` instances are
    equal.

    .. note:: As opposed to classes such as `.EdgeQuantumNumbers` and
        `.NodeQuantumNumbers`, the `Particle` class serves as an interface to
        the user (see :doc:`/usage/particle`).
    """

    # Labels
    name: str = field(eq=False)
    pid: int = field(eq=False)
    latex: str | None = field(eq=False, default=None)
    # Unique properties
    spin: Fraction = field(converter=Fraction)
    mass: float = field(converter=float)
    width: float = field(converter=float, default=0.0)
    charge: int = field(default=0)
    isospin: Spin | None = field(converter=optional(_to_spin), default=None)
    strangeness: int = field(default=0, validator=instance_of(int))
    charmness: int = field(default=0, validator=instance_of(int))
    bottomness: int = field(default=0, validator=instance_of(int))
    topness: int = field(default=0, validator=instance_of(int))
    baryon_number: int = field(default=0, validator=instance_of(int))
    electron_lepton_number: int = field(default=0, validator=instance_of(int))
    muon_lepton_number: int = field(default=0, validator=instance_of(int))
    tau_lepton_number: int = field(default=0, validator=instance_of(int))
    parity: Parity | None = field(converter=optional(to_parity), default=None)
    c_parity: Parity | None = field(converter=optional(to_parity), default=None)
    g_parity: Parity | None = field(converter=optional(to_parity), default=None)

    def __attrs_post_init__(self) -> None:
        if self.isospin is not None and not gellmann_nishijima(
            GellMannNishijimaInput(
                charge=self.charge,
                isospin_projection=self.isospin.projection if self.isospin else None,
                strangeness=self.strangeness,
                charmness=self.charmness,
                bottomness=self.bottomness,
                topness=self.topness,
                baryon_number=self.baryon_number,
                electron_lepton_number=self.electron_lepton_number,
                muon_lepton_number=self.muon_lepton_number,
                tau_lepton_number=self.tau_lepton_number,
            )
        ):
            msg = (
                f"Cannot construct particle {self.name}, because its quantum numbers"
                " don't agree with the Gell-Mann-Nishijima formula:\n "
                f" Q[{self.charge}] !="
                f" Iz[{self.isospin.projection if self.isospin else 0}] + 1/2"
                f" (B[{self.baryon_number}] +  S[{self.strangeness}] + "
                f" C[{self.charmness}] + B'[{self.bottomness}] + T[{self.topness}])"
            )
            raise ValueError(msg)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Particle):

            def sorting_key(particle: Particle) -> tuple:
                name_root = _get_name_root(particle.name)
                return (
                    name_root[0].lower(),
                    name_root,
                    particle.mass,
                    particle.charge,
                )

            return sorting_key(self) > sorting_key(other)
        msg = f"Cannot compare {type(self).__name__} with {type(other).__name__}"
        raise NotImplementedError(msg)

    def __neg__(self) -> Particle:
        return create_antiparticle(self)

    def is_lepton(self) -> bool:
        return (
            self.electron_lepton_number != 0
            or self.muon_lepton_number != 0
            or self.tau_lepton_number != 0
        )

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}("):
                for attribute in attrs.fields(type(self)):
                    value = getattr(self, attribute.name)
                    if value != attribute.default:
                        p.breakable()
                        p.text(f"{attribute.name}=")
                        if isinstance(value, Parity):
                            p.text(_float_as_signed_str(int(value), render_plus=True))
                        else:
                            p.pretty(value)
                        p.text(",")
            p.breakable()
            p.text(")")


def _get_name_root(name: str) -> str:
    """Strip a string (particularly the `.Particle.name`) of specifications."""
    name_root = re.sub(r"\([^)]*\)", "", name)
    name_root = name_root.split("_", maxsplit=1)[0]
    name_root = re.sub(r"[\^\*\+\-~\d']", "", name_root)
    return name_root.removesuffix("bar")


ParticleWithSpin = tuple[Particle, Fraction]
"""A particle and its spin projection."""


class ParticleCollection(abc.MutableSet):  # ruff: ignore[eq-without-hash]
    """Searchable collection of immutable `.Particle` instances."""

    def __init__(self, particles: Iterable[Particle] | None = None) -> None:
        self.__particles: dict[str, Particle] = {}
        self.__pid_to_name: dict[int, str] = {}
        if particles is not None:
            self.update(particles)

    def __contains__(self, instance: object) -> bool:
        if isinstance(instance, str):
            return instance in self.__particles
        if isinstance(instance, Particle):
            return instance in self.__particles.values()
        if isinstance(instance, int):
            return instance in self.__pid_to_name
        msg = f"Cannot search for type {type(instance).__name__}"
        raise NotImplementedError(msg)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, abc.Iterable):
            return set(self) == set(other)
        msg = f"Cannot compare {type(self).__name__} with  {type(self).__name__}"
        raise NotImplementedError(msg)

    def __getitem__(self, particle_name: str) -> Particle:
        if particle_name in self.__particles:
            return self.__particles[particle_name]
        error_message = f"No particle with name '{particle_name}' in the database"
        candidates = [
            p.name
            for p in sorted(self, key=lambda p: p.mass)
            if p.name.startswith(particle_name)
        ]
        if not candidates:
            candidates = get_close_matches(particle_name, self.names, n=5)
        if len(candidates) == 1:
            error_message += f". Did you mean '{candidates[0]}'?"
        elif len(candidates) > 1:
            error_message += f". Did you mean one of these? {candidates}"
        raise KeyError(error_message)

    def __iter__(self) -> Iterator[Particle]:
        return self.__particles.values().__iter__()

    def __len__(self) -> int:
        return len(self.__particles)

    def __iadd__(self, other: Particle | ParticleCollection) -> Self:
        if isinstance(other, Particle):
            self.add(other)
        elif isinstance(other, ParticleCollection):
            self.update(other)
        else:
            msg = f"Cannot add {type(other).__name__}"
            raise NotImplementedError(msg)
        return self

    def __repr__(self) -> str:
        output = f"{type(self).__name__}({{"
        for particle in self:
            output += f"\n    {particle},"
        output += "})"
        return output

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}({{"):
                for particle in self:
                    p.breakable()
                    p.pretty(particle)
                    p.text(",")
            p.breakable()
            p.text("})")

    def add(self, value: Particle) -> None:
        if value in self.__particles.values():
            equivalent_particles = {p for p in self if p == value}
            equivalent_particle = next(iter(equivalent_particles))
            msg = (
                f'Added particle "{value.name}" is equivalent to existing particle'
                f' "{equivalent_particle.name}"'
            )
            raise ValueError(
                (msg),
            )
        if value.name in self.__particles:
            _LOGGER.warning(f'Overwriting particle with name "{value.name}"')
        if value.pid in self.__pid_to_name:
            _LOGGER.warning(
                f"Particle with PID {value.pid} already exists:"
                f' "{self.find(value.pid).name}"'
            )
        self.__particles[value.name] = value
        self.__pid_to_name[value.pid] = value.name

    def discard(self, value: Particle | str) -> None:
        particle_name = ""
        if isinstance(value, Particle):
            particle_name = value.name
        elif isinstance(value, str):
            particle_name = value
        else:
            msg = f"Cannot discard something of type {type(value).__name__}"
            raise NotImplementedError(msg)
        del self.__pid_to_name[self[particle_name].pid]
        del self.__particles[particle_name]

    def find(self, search_term: int | str) -> Particle:
        """Search for a particle by either name (`str`) or PID (`int`)."""
        if isinstance(search_term, str):
            particle_name = search_term
            return self[particle_name]
        if isinstance(search_term, int):
            if search_term not in self.__pid_to_name:
                msg = f"No particle with PID {search_term}"
                raise KeyError(msg)
            particle_name = self.__pid_to_name[search_term]
            return self[particle_name]
        msg = f"Cannot search for a search term of type {type(search_term)}"
        raise NotImplementedError(msg)

    def filter(self, function: Callable[[Particle], bool]) -> ParticleCollection:
        """Search by `Particle` properties using a :code:`lambda` function.

        For example:

        >>> from qrules.particle import load_pdg
        >>> pdg = load_pdg()
        >>> subset = pdg.filter(
        ...     lambda p: (
        ...         p.mass > 1.8
        ...         and p.mass < 2.15
        ...         and p.spin == 2
        ...         and p.strangeness == 1
        ...     )
        ... )
        >>> sorted(subset.names)
        ['K(2)(1820)+', 'K(2)(1820)0', 'K(2)*(1980)+', 'K(2)*(1980)0']
        """
        return ParticleCollection({particle for particle in self if function(particle)})

    def update(self, other: Iterable[Particle]) -> None:
        if not isinstance(other, abc.Iterable):
            msg = (
                f"Cannot update {type(self).__name__} from non-iterable class"
                f" {type(self).__name__}"
            )
            raise TypeError(msg)
        for particle in other:
            self.add(particle)

    @property
    def names(self) -> list[str]:
        return [p.name for p in sorted(self)]


def create_particle(  # ruff: ignore[too-many-positional-arguments]
    template_particle: Particle,
    name: str | None = None,
    latex: str | None = None,
    pid: int | None = None,
    mass: float | None = None,
    width: float | None = None,
    charge: int | None = None,
    spin: float | None = None,
    isospin: Spin | None = None,
    strangeness: int | None = None,
    charmness: int | None = None,
    bottomness: int | None = None,
    topness: int | None = None,
    baryon_number: int | None = None,
    electron_lepton_number: int | None = None,
    muon_lepton_number: int | None = None,
    tau_lepton_number: int | None = None,
    parity: int | None = None,
    c_parity: int | None = None,
    g_parity: int | None = None,
) -> Particle:
    return Particle(
        name=name or template_particle.name,
        pid=pid or template_particle.pid,
        latex=latex or template_particle.latex,
        mass=mass if mass is not None else template_particle.mass,
        width=width or template_particle.width,
        spin=spin or template_particle.spin,
        charge=charge or template_particle.charge,
        strangeness=strangeness or template_particle.strangeness,
        charmness=charmness or template_particle.charmness,
        bottomness=bottomness or template_particle.bottomness,
        topness=topness or template_particle.topness,
        baryon_number=(baryon_number or template_particle.baryon_number),
        electron_lepton_number=(
            electron_lepton_number or template_particle.electron_lepton_number
        ),
        muon_lepton_number=(muon_lepton_number or template_particle.muon_lepton_number),
        tau_lepton_number=(tau_lepton_number or template_particle.tau_lepton_number),
        isospin=template_particle.isospin if isospin is None else isospin,
        parity=template_particle.parity if parity is None else Parity(parity),
        c_parity=template_particle.c_parity if c_parity is None else Parity(c_parity),
        g_parity=template_particle.g_parity if g_parity is None else Parity(g_parity),
    )


def create_antiparticle(
    template_particle: Particle,
    new_name: str | None = None,
    new_latex: str | None = None,
) -> Particle:
    isospin: Spin | None = None
    if template_particle.isospin:
        isospin = -template_particle.isospin
    parity: Parity | None = None
    if template_particle.parity is not None:
        if template_particle.spin.denominator == 1:
            parity = template_particle.parity
        else:
            parity = -template_particle.parity
    return Particle(
        name=new_name or "anti-" + template_particle.name,
        pid=-template_particle.pid,
        latex=new_latex or Rf"\overline{{{template_particle.latex}}}",
        mass=template_particle.mass,
        width=template_particle.width,
        charge=-template_particle.charge,
        spin=template_particle.spin,
        isospin=isospin,
        strangeness=-template_particle.strangeness,
        charmness=-template_particle.charmness,
        bottomness=-template_particle.bottomness,
        topness=-template_particle.topness,
        baryon_number=-template_particle.baryon_number,
        electron_lepton_number=-template_particle.electron_lepton_number,
        muon_lepton_number=-template_particle.muon_lepton_number,
        tau_lepton_number=-template_particle.tau_lepton_number,
        parity=parity,
        c_parity=template_particle.c_parity,
        g_parity=template_particle.g_parity,
    )


def load_pdg() -> ParticleCollection:
    """Create a `.ParticleCollection` with all entries from the PDG.

    PDG info is imported from the official `PDG Python API
    <https://pdgapi.lbl.gov/doc/>`_.
    """
    from qrules._pdg import load_pdg as load_official_pdg  # ruff: ignore[import-outside-top-level]

    return load_official_pdg()

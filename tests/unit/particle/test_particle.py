from __future__ import annotations

import logging
from copy import deepcopy
from fractions import Fraction
from typing import Any, cast

import pytest
from attrs.exceptions import FrozenInstanceError
from IPython.lib.pretty import pretty

from qrules.particle import (
    Particle,
    ParticleCollection,
    Spin,
    create_antiparticle,
    create_particle,
    load_pdg,
)

# For eval tests
from qrules.quantum_numbers import Parity  # ruff: ignore[unused-import]


def gen_namespace_with_fraction():
    namespace = globals()
    namespace["Fraction"] = Fraction
    return namespace


def describe_load_pdg():
    def it_uses_scikit_hep_source_by_default():
        default_particles = load_pdg()
        scikit_hep_particles = load_pdg(source="particle")

        assert default_particles == scikit_hep_particles
        assert default_particles.find(-2212).name == "p~"

    def it_rejects_unknown_source():
        with pytest.raises(ValueError, match="Unknown particle source"):
            load_pdg(source=cast("Any", "unknown"))


def describe_Particle():
    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def it_repr(particle_database: ParticleCollection, repr_method):
        local_namespace = locals()
        local_namespace["Fraction"] = Fraction
        for instance in particle_database:
            from_repr = eval(repr_method(instance), None, gen_namespace_with_fraction())
            assert from_repr == instance

    def it_exceptions():
        test_state = Particle(
            name="MyParticle",
            pid=123,
            mass=1.2,
            width=0.1,
            spin=1,
            charge=0,
            isospin=(Fraction(1), Fraction(0)),
        )
        with pytest.raises(FrozenInstanceError):
            test_state.charge = 1  # ty: ignore[invalid-assignment]
        with pytest.raises(
            ValueError,
            match=r"Fails Gell-Mann–Nishijima",  # ruff: ignore[ambiguous-unicode-character-string]
        ):
            Particle(
                name="Fails Gell-Mann–Nishijima formula",  # ruff: ignore[ambiguous-unicode-character-string]
                pid=666,
                mass=0.0,
                spin=1,
                charge=0,
                parity=-1,
                c_parity=-1,
                g_parity=-1,
                isospin=(0, 0),
                charmness=1,
            )

    def it_eq():
        particle = Particle(
            name="MyParticle",
            pid=123,
            mass=1.2,
            spin=1,
            charge=0,
            isospin=(Fraction(1), Fraction(0)),
        )
        assert particle != Particle(
            name="MyParticle", pid=123, mass=1.5, width=0.2, spin=1
        )
        same_particle = deepcopy(particle)
        assert particle is not same_particle
        assert particle == same_particle
        assert hash(particle) == hash(same_particle)
        different_labels = Particle(
            name="Different name, same QNs",
            pid=753,
            mass=1.2,
            spin=1,
            charge=0,
            isospin=(Fraction(1), Fraction(0)),
        )
        assert particle == different_labels
        assert hash(particle) == hash(different_labels)
        assert particle.name != different_labels.name
        assert particle.pid != different_labels.pid

    @pytest.mark.parametrize(
        ("name1", "name2"),
        [
            # by name
            ("pi0", "a(0)(980)-"),
            # by mass
            ("pi+", "pi-"),
            ("pi-", "pi0"),
            ("pi+", "pi0"),
            ("K0", "K+"),
            # by charge
            ("a(0)(980)+", "a(0)(980)-"),
            ("a(0)(980)+", "a(0)(980)0"),
            ("a(0)(980)0", "a(0)(980)-"),
        ],
    )
    def it_gt(name1, name2, particle_database: ParticleCollection):
        pdg = particle_database
        assert pdg[name1] > pdg[name2]


def describe_ParticleCollection():
    def it_init(particle_database: ParticleCollection):
        new_pdg = ParticleCollection(particle_database)
        assert new_pdg is not particle_database
        assert new_pdg == particle_database
        with pytest.raises(TypeError):
            ParticleCollection(1)  # ty: ignore[invalid-argument-type]

    def it_equality(particle_database: ParticleCollection):
        assert list(particle_database) == particle_database
        with pytest.raises(NotImplementedError):
            assert particle_database == 0

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def it_repr(particle_database: ParticleCollection, repr_method):
        instance = particle_database
        local_namespace = locals()
        local_namespace["Fraction"] = Fraction
        from_repr = eval(repr_method(instance), None, gen_namespace_with_fraction())
        assert from_repr == instance

    def it_add(particle_database: ParticleCollection):
        subset_copy = particle_database.filter(lambda p: p.name.startswith("omega"))
        subset_copy += particle_database.filter(lambda p: p.name.startswith("pi"))
        n_subset = len(subset_copy)

        new_particle = create_particle(
            particle_database.find(443),
            pid=666,
            name="EpEm",
            mass=1.0,
            width=0.0,
        )
        subset_copy.add(new_particle)
        assert len(subset_copy) == n_subset + 1
        assert subset_copy["EpEm"] is new_particle

    def it_add_warnings(particle_database: ParticleCollection, caplog):
        pions = particle_database.filter(lambda p: p.name.startswith("pi"))
        pi_plus = pions["pi+"]
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            pions.add(create_particle(pi_plus, name="new pi+", mass=0.0))
        assert f"{pi_plus.pid}" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            pions.add(create_particle(pi_plus, width=1.0))
        assert "pi+" in caplog.text

    @pytest.mark.parametrize("name", ["gamma", "pi0", "K+"])
    def it_contains(name: str, particle_database: ParticleCollection):
        assert name in particle_database
        particle = particle_database[name]
        assert particle in particle_database
        assert particle.pid in particle_database

    def it_discard(particle_database: ParticleCollection):
        pions = particle_database.filter(lambda p: p.name.startswith("pi"))
        n_pions = len(pions)
        pim = pions["pi-"]
        pip = pions["pi+"]

        pions.discard(pions["pi+"])
        assert len(pions) == n_pions - 1
        assert "pi+" not in pions
        assert pip.name == "pi+"  # still exists

        pions.remove("pi-")
        assert len(pions) == n_pions - 2
        assert pim not in pions
        assert pim.name == "pi-"  # still exists

        with pytest.raises(NotImplementedError):
            pions.discard(111)  # ty: ignore[invalid-argument-type]

    def it_exceptions(particle_database: ParticleCollection):
        gamma = particle_database["gamma"]
        with pytest.raises(
            ValueError,
            match=(
                'Added particle "gamma_new" is equivalent to existing particle "gamma"'
            ),
        ):
            particle_database += create_particle(gamma, name="gamma_new")
        with pytest.raises(NotImplementedError):
            particle_database.find(3.12)  # ty: ignore[invalid-argument-type]
        with pytest.raises(NotImplementedError):
            particle_database += 3.12  # ty: ignore[unsupported-operator]
        with pytest.raises(NotImplementedError):
            assert 3.12 in particle_database
        with pytest.raises(AssertionError):
            assert gamma == "gamma"


def describe_Spin():
    def it_init_and_eq():
        isospin = Spin(1.5, -0.5)
        assert isospin == 1.5
        assert float(isospin) == 1.5
        assert isospin.magnitude == 1.5
        assert isospin.projection == -0.5
        isospin = Spin(1, -0.0)
        assert isinstance(isospin.magnitude, Fraction)
        assert isinstance(isospin.projection, Fraction)
        assert isospin.magnitude == 1.0
        assert isospin.projection == 0.0

    def it_hash():
        spin1 = Spin(0.0, 0.0)
        spin2 = Spin(1.5, -0.5)
        assert {spin2, spin1, deepcopy(spin1), deepcopy(spin2)} == {
            spin1,
            spin2,
        }

    @pytest.mark.parametrize(
        ("spin1", "spin2"),
        [
            (Spin(1, 0), Spin(0, 0)),
            (Spin(1, 1), Spin(1, 0)),
            (Spin(1, +1), Spin(1, -1)),
        ],
    )
    def it_gt(spin1: Spin, spin2: Spin):
        assert spin1 > spin2

    def it_neg():
        isospin = Spin(1.5, -0.5)
        flipped_spin = -isospin
        assert flipped_spin.magnitude == isospin.magnitude
        assert flipped_spin.projection == -isospin.projection

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    @pytest.mark.parametrize(
        "instance", [Spin(2.5, -0.5), Spin(1, 0), Spin(3, -1), Spin(0, 0)]
    )
    def it_repr(instance: Spin, repr_method):
        from_repr = eval(repr_method(instance), None, gen_namespace_with_fraction())
        assert from_repr == instance

    @pytest.mark.parametrize(
        ("magnitude", "projection"),
        [(0.3, 0.3), (1.0, 0.5), (0.5, 0.0), (-0.5, 0.5)],
    )
    def it_exceptions(magnitude, projection):
        regex_pattern = "|".join([  # ruff: ignore[static-join-to-f-string]
            r"Spin magnitude \d+/\d+ has to be a multitude of \d\.[05]",
            r"\(projection - magnitude\) should be integer",
            r"Spin magnitude has to be positive",
            r"Absolute value of spin projection cannot be larger than the magnitude",
        ])
        with pytest.raises(ValueError, match=regex_pattern):
            print(Spin(magnitude, projection))


def describe_create_antiparticle():
    @pytest.mark.parametrize(
        ("particle_name", "anti_particle_name"),
        [("D+", "D-"), ("mu+", "mu-"), ("W+", "W-")],
    )
    def it_creates_a_named_antiparticle(
        particle_database: ParticleCollection,
        particle_name,
        anti_particle_name,
    ):
        template_particle = particle_database[particle_name]
        anti_particle = create_antiparticle(
            template_particle, new_name=anti_particle_name
        )
        comparison_particle = particle_database[anti_particle_name]

        assert anti_particle == comparison_particle


def describe_create_particle():
    @pytest.mark.parametrize(
        "particle_name",
        ["p", "phi(1020)", "W-", "gamma"],
    )
    def it_create_particle(particle_database: ParticleCollection, particle_name: str):
        template_particle = particle_database[particle_name]
        new_particle = create_particle(
            template_particle,
            name="testparticle",
            pid=89,
            mass=1.5,
            width=0.5,
        )
        assert new_particle.name == "testparticle"
        assert new_particle.pid == 89
        assert new_particle.charge == template_particle.charge
        assert new_particle.spin == template_particle.spin
        assert new_particle.mass == 1.5
        assert new_particle.width == 0.5
        assert new_particle.baryon_number == template_particle.baryon_number
        assert new_particle.strangeness == template_particle.strangeness

    def it_isospin():
        template_particle = Particle(
            name="some particle",
            pid=0,
            spin=0,
            mass=3.12,
        )
        new_isospin = Spin(0, 0)
        new_particle = create_particle(
            template_particle,
            isospin=new_isospin,
        )
        assert template_particle.isospin != new_isospin
        assert new_particle.isospin == new_isospin

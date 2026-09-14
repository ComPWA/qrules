from importlib.metadata import version

import pytest

from qrules.particle import ParticleCollection, _get_name_root, create_antiparticle


def _get_omega_mesons() -> list[str]:
    scikit_hep_particle_version = ".".join(version("particle").split(".")[:2])
    if scikit_hep_particle_version in {"0.21", "0.22"}:
        return ["omega(782)", "omega(3)(1670)", "omega(1650)"]
    return ["omega(782)", "omega(1420)", "omega(3)(1670)", "omega(1650)"]


def describe_load_pdg():
    @pytest.mark.parametrize(
        ("name", "is_lepton"),
        [
            ("J/psi(1S)", False),
            ("p", False),
            ("e+", True),
            ("e-", True),
            ("nu(e)", True),
            ("nu(tau)~", True),
            ("tau+", True),
        ],
    )
    def it_loads_lepton_numbers(
        name: str,
        is_lepton: bool,
        particle_database: ParticleCollection,
    ):
        assert particle_database[name].is_lepton() == is_lepton

    def it_loads_consistent_pions(particle_database: ParticleCollection):
        pi_plus = particle_database.find(211)
        pi_minus = particle_database.find(-211)
        assert pi_plus == -pi_minus

    def it_loads_expected_f0_mesons(
        particle_database: ParticleCollection,
        skh_particle_version: str,
    ):
        f0_mesons = sorted(
            particle.name
            for particle in sorted(
                particle_database.filter(lambda p: p.name.startswith("f(0)"))
            )
        )
        expected = {
            "f(0)(500)",
            "f(0)(980)",
            "f(0)(1370)",
            "f(0)(1500)",
            "f(0)(1710)",
        }
        if skh_particle_version > "0.22":
            expected.add("f(0)(2020)")
        assert f0_mesons == sorted(expected)

    def it_loads_filterable_particle_properties(
        particle_database: ParticleCollection,
        skh_particle_version: str,
    ):
        search_result = particle_database.filter(lambda p: "f(0)" in p.name)
        if skh_particle_version < "0.23":
            assert len(search_result) == 5
        else:
            assert len(search_result) == 6
        f0_1500_from_subset = search_result["f(0)(1500)"]
        if skh_particle_version < "0.23":
            assert f0_1500_from_subset.mass == 1.506
        else:
            assert f0_1500_from_subset.mass == 1.522
        assert f0_1500_from_subset is particle_database["f(0)(1500)"]
        assert f0_1500_from_subset is not particle_database["f(0)(980)"]

        search_result = particle_database.filter(lambda p: p.pid == 22)
        gamma_from_subset = search_result["gamma"]
        assert len(search_result) == 1
        assert gamma_from_subset.pid == 22
        assert gamma_from_subset is particle_database["gamma"]
        filtered_result = particle_database.filter(
            lambda p: (
                p.mass > 1.8 and p.mass < 2.0 and p.spin == 2 and p.strangeness == 1
            )
        )
        sorted_result = sorted(filtered_result.names)
        expected = {
            "K(2)(1820)+",
            "K(2)(1820)0",
        }
        if skh_particle_version > "0.15":
            expected.update({"K(2)*(1980)+", "K(2)*(1980)0"})
        assert sorted_result == sorted(expected)

    def it_loads_particle_properties(particle_database: ParticleCollection):
        f2_1950 = particle_database.find(9050225)
        assert f2_1950.name == "f(2)(1950)"
        assert f2_1950.mass == 1.936
        phi = particle_database.find("phi(1020)")
        assert phi.pid == 333
        assert pytest.approx(phi.width) == 0.004249

    @pytest.mark.parametrize(
        ("search_term", "expected"),
        [
            (666, None),
            ("non-existing", None),
            # cspell:disable
            ("gamm", "gamma"),
            ("gama", ["gamma", "Sigma0", "Sigma-", "Sigma+", "Lambda"]),
            ("omega", _get_omega_mesons()),
            ("p~~", "p~"),
            ("~", ["p~", "n~"]),
            ("lambda", ["Lambda", "Lambda~", "Lambda(c)+", "Lambda(b)0"]),
            # cspell:enable
        ],
    )
    def it_provides_expected_lookup_suggestions(
        particle_database: ParticleCollection,
        search_term,
        expected: list[str] | str | None,
    ):
        with pytest.raises(LookupError) as exception:
            particle_database.find(search_term)
        if expected is not None:
            message = str(exception.value.args[0])
            list_str = message.strip("?")
            *_, list_str = list_str.split("Did you mean ")
            *_, list_str = list_str.split("one of these? ")
            assert eval(list_str) == expected

    def it_loads_consistent_antiparticles(
        particle_database: ParticleCollection,
        skh_particle_version: str,
    ):
        anti_particles = particle_database.filter(lambda p: "~" in p.name)
        if skh_particle_version < "0.14":
            assert len(anti_particles) == 165
        elif skh_particle_version < "0.16":
            assert len(anti_particles) == 172
        elif skh_particle_version < "0.25":
            assert len(anti_particles) == 175
        else:
            assert len(anti_particles) == 176
        for anti_particle in anti_particles:
            particle_name = anti_particle.name.replace("~", "")
            if "+" in particle_name:
                particle_name = particle_name.replace("+", "-")
            elif "-" in particle_name:
                particle_name = particle_name.replace("-", "+")
            created_particle = create_antiparticle(anti_particle, particle_name)
            assert created_particle == particle_database[particle_name]

    def it_loads_particles_consistently_by_pid(
        particle_database: ParticleCollection,
        skh_particle_version: str,
    ):
        n_particles_with_neg_pid = 0
        for particle in particle_database:
            anti_particles_by_pid = particle_database.filter(
                lambda p: p.pid == -particle.pid  # ruff: ignore[function-uses-loop-variable]
            )
            if len(anti_particles_by_pid) != 1:
                continue
            n_particles_with_neg_pid += 1
            anti_particle = next(iter(anti_particles_by_pid))
            assert particle == -anti_particle
        if skh_particle_version < "0.14":
            assert n_particles_with_neg_pid == 428
        elif skh_particle_version < "0.16":
            assert n_particles_with_neg_pid == 442
        elif skh_particle_version < "0.25":
            assert n_particles_with_neg_pid == 454
        else:
            assert n_particles_with_neg_pid == 456

    def it_loads_expected_name_roots(particle_database: ParticleCollection):
        name_roots = {_get_name_root(p.name) for p in particle_database}
        assert name_roots == {
            "a",
            "B",
            "b",
            "chi",
            "D",
            "Delta",
            "H",
            "e",
            "eta",
            "f",
            "g",
            "gamma",
            "h",
            "J/psi",
            "K",
            "Lambda",
            "mu",
            "N",
            "n",
            "nu",
            "Omega",
            "omega",
            "p",
            "phi",
            "pi",
            "psi",
            "rho",
            "Sigma",
            "tau",
            "Upsilon",
            "W",
            "Xi",
            "Y",
            "Z",
        }

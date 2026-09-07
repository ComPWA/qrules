import pytest

from qrules._pdg_latex import create_latex_name


@pytest.mark.parametrize(
    ("name", "isospin", "self_conjugate", "latex"),
    [
        ("pi+", "1", False, R"\pi^{+}"),
        ("pi0", "1", True, R"\pi^{0}"),
        ("pbar", "1/2", False, R"\overline{p}"),
        ("nubar_mu", None, False, R"\overline{\nu}_{\mu}"),
        ("J/psi(1S)", "0", True, R"J/\psi(1S)"),
        ("f_0(980)0", "0", True, R"f_{0}(980)"),
        ("Lambda_b()0", "0", False, R"\Lambda_{b}^{0}"),
        ("D_0^*(2300)+", "1/2", False, R"D_{0}^{*}(2300)^{+}"),
        ("D_s^*()+", "0", False, R"D_{s}^{*+}"),
        ("Xibar_c^'()0", "1/2", False, R"\overline{\Xi}_{c}^{\prime0}"),
        ("eta^'(958)0", "0", True, R"\eta^{\prime}(958)"),
    ],
)
def test_create_latex_name(
    name: str,
    isospin: str | None,
    self_conjugate: bool,
    latex: str,
):
    assert (
        create_latex_name(
            name,
            isospin=isospin,
            self_conjugate=self_conjugate,
        )
        == latex
    )


def test_rejects_unsupported_name_syntax():
    assert (
        create_latex_name(
            "not a PDG name",
            isospin=None,
            self_conjugate=False,
        )
        is None
    )

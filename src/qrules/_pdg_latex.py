"""Convert official PDG ASCII particle names to LaTeX."""

from __future__ import annotations

import re

_PARTICLE_NAME_PATTERN = re.compile(
    r"^(?P<base>[A-Za-z]+(?:/[A-Za-z]+)?)"
    r"(?P<scripts>(?:[_^](?:[A-Za-z0-9]+|\*|'))*)"
    r"(?P<qualifier>\([^()]*\))?"
    r"(?P<charge>\+\+|--|\+|-|0)?$"
)
_SCRIPT_PATTERN = re.compile(r"(?P<kind>[_^])(?P<value>[A-Za-z0-9]+|\*|')")

_LATEX_SYMBOLS = {
    "Delta": R"\Delta",
    "Lambda": R"\Lambda",
    "Omega": R"\Omega",
    "Sigma": R"\Sigma",
    "Upsilon": R"\Upsilon",
    "Xi": R"\Xi",
    "chi": R"\chi",
    "eta": R"\eta",
    "gamma": R"\gamma",
    "mu": R"\mu",
    "nu": R"\nu",
    "omega": R"\omega",
    "phi": R"\phi",
    "pi": R"\pi",
    "psi": R"\psi",
    "rho": R"\rho",
    "tau": R"\tau",
}


def create_latex_name(
    name: str,
    *,
    isospin: str | None,
    self_conjugate: bool,
) -> str | None:
    """Convert a canonical PDG name, returning `None` for unsupported syntax."""
    match = _PARTICLE_NAME_PATTERN.fullmatch(name)
    if match is None:
        return None

    latex = _render_base(match.group("base"))

    subscripts: list[str] = []
    superscripts: list[str] = []
    for script in _SCRIPT_PATTERN.finditer(match.group("scripts")):
        value = script.group("value")
        if script.group("kind") == "_":
            subscripts.append(_LATEX_SYMBOLS.get(value, value))
        else:
            superscripts.append(R"\prime" if value == "'" else value)
    latex += "".join(Rf"_{{{value}}}" for value in subscripts)

    charge = match.group("charge")
    if charge == "0" and self_conjugate and isospin == "0":
        charge = None
    qualifier = match.group("qualifier")
    if qualifier in {None, "()"} and charge is not None and superscripts:
        superscripts.append(charge)
        charge = None
    if superscripts:
        latex += Rf"^{{{''.join(superscripts)}}}"
    if qualifier != "()" and qualifier is not None:
        latex += qualifier
    if charge is not None:
        latex += Rf"^{{{charge}}}"
    return latex


def _render_base(base: str) -> str:
    is_antiparticle = base.endswith("bar")
    if is_antiparticle:
        base = base.removesuffix("bar")
    latex = "/".join(_LATEX_SYMBOLS.get(part, part) for part in base.split("/"))
    if is_antiparticle:
        latex = Rf"\overline{{{latex}}}"
    return latex

"""`{name}` substitution in scenario text.

`{name}` is replaced by the variable's value, `{{` gives a literal `{` and
`}}` a literal `}`. Braces around anything that is not a variable name, such
as a regex quantifier `{2,3}`, are left as written. Values are inserted once
and never substituted again.
"""

import re
from collections.abc import Callable, Mapping

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PART = re.compile(r"\{\{|\}\}|\{(" + NAME.pattern + r")\}")


class UnknownVariableError(ValueError):
    def __init__(self, names: tuple[str, ...], defined: Mapping[str, str]):
        self.names = names
        listed = ", ".join("{" + n + "}" for n in names)
        known = f"defined: {', '.join(sorted(defined))}" if defined else "no variables defined"
        super().__init__(f"unknown variable {listed} ({known})")


def substitute(
    text: str,
    variables: Mapping[str, str],
    escape: Callable[[str], str] | None = None,
) -> str:
    """`text` with its variables filled in, each value passed through
    `escape` first when given. Raises UnknownVariableError naming every
    variable that is not in `variables`."""
    missing: list[str] = []

    def fill(match: re.Match[str]) -> str:
        part = match.group(0)
        if part == "{{":
            return "{"
        if part == "}}":
            return "}"
        name = match.group(1)
        if name not in variables:
            if name not in missing:
                missing.append(name)
            return part
        value = variables[name]
        return escape(value) if escape else value

    result = _PART.sub(fill, text)
    if missing:
        raise UnknownVariableError(tuple(missing), variables)
    return result

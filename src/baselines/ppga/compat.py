"""Small compatibility helpers for the ported PPGA baseline."""

from __future__ import annotations


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return AttrDict(super().copy())


def strtobool(value: str) -> bool:
    value = str(value).lower()
    if value in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if value in {"n", "no", "f", "false", "off", "0"}:
        return False
    raise ValueError(f"invalid truth value {value!r}")

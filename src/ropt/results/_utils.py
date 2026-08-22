from typing import Any, Literal, overload

from numpy.typing import NDArray

DomainType = Literal["optimizer", "user"]
"""Selects the domain that result values are expressed in.

Values are `"optimizer"` for the domain the optimizer works in, and `"user"` for
the domain the configuration is written in. The two coincide unless transforms
are configured.
"""


@overload
def _immutable_copy(data: NDArray[Any]) -> NDArray[Any]: ...


@overload
def _immutable_copy(data: NDArray[Any] | None) -> NDArray[Any] | None: ...


def _immutable_copy(data: NDArray[Any] | None) -> NDArray[Any] | None:
    if data is not None:
        data = data.copy()
        data.setflags(write=False)
    return data

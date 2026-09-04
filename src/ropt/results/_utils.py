from typing import Any, overload

from numpy.typing import NDArray


@overload
def _immutable_copy(data: NDArray[Any]) -> NDArray[Any]: ...


@overload
def _immutable_copy(data: NDArray[Any] | None) -> NDArray[Any] | None: ...


def _immutable_copy(data: NDArray[Any] | None) -> NDArray[Any] | None:
    if data is not None:
        data = data.copy()
        data.setflags(write=False)
    return data

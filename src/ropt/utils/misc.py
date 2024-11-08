"""Miscellaneous utilities."""

from itertools import zip_longest
from typing import Any, Tuple, Union


def format_tuple(values: Union[str, Tuple[Any, ...]], delimiters: str = ":") -> str:
    """Format a tuple of strings into a single string.

    This function converts a tuple into a single string by joining the tuple
    items with specified delimiters. The `delimiters` argument can contain
    multiple characters, each of which will be used in turn to separate the
    items in `name`. If fewer delimiters are provided than needed, the final
    delimiter is reused for any remaining items. If `delimiters` is not
    specified, `:` is used by default.

    Args:
        values:     The tuple of strings to format.
        delimiters: The delimiters to use, applied sequentially.

    Returns:
        The formatted string.
    """
    if not delimiters:
        return "".join(str(item) for item in values)
    if isinstance(values, tuple):
        truncated_delimiters = delimiters[: len(values) - 1]
        return str(values[0]) + "".join(
            str(item)
            for item_tuple in zip_longest(
                truncated_delimiters, values[1:], fillvalue=delimiters[-1]
            )
            for item in item_tuple
        )
    return str(values)

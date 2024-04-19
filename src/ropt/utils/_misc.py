"""Miscellaneous utilities."""

from copy import deepcopy
from typing import Any, Dict


def update_dict(mapping: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary.

    Args:
        mapping: The dictionary to update
        values:  The values to be updated or added

    Returns:
        The updated dictionary.
    """
    mapping = mapping.copy()
    for key, value in values.items():
        if (
            key in mapping
            and isinstance(mapping[key], dict)
            and isinstance(value, dict)
        ):
            mapping[key] = update_dict(mapping[key], value)
        else:
            mapping[key] = deepcopy(value)
    return mapping

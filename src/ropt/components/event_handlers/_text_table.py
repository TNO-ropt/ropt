from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_GAP = "  "


def format_text_table(
    headers: Sequence[Sequence[str]],
    columns: Sequence[Sequence[object]],
    *,
    digits: int = 6,
) -> str:
    if not columns:
        return ""
    cells, right_aligned = zip(
        *(_format_cells(values, digits) for values in columns), strict=True
    )
    widths = [
        max(len(text) for text in [*header, *column])
        for header, column in zip(headers, cells, strict=True)
    ]
    header_count = max(len(header) for header in headers)
    header_lines = [
        [header[index] if index < len(header) else "" for header in headers]
        for index in range(header_count)
    ]
    rule = _GAP.join("-" * width for width in widths)
    lines = [
        *(_join(line, widths, right_aligned) for line in header_lines),
        rule,
        *(_join(row, widths, right_aligned) for row in zip(*cells, strict=True)),
    ]
    return "\n".join(lines) + "\n"


def _join(texts: Sequence[str], widths: Sequence[int], right: Sequence[bool]) -> str:
    padded = (
        text.rjust(width) if is_right else text.ljust(width)
        for text, width, is_right in zip(texts, widths, right, strict=True)
    )
    return _GAP.join(padded).rstrip()


def _format_cells(values: Sequence[object], digits: int) -> tuple[list[str], bool]:
    if not all(_is_number(value) for value in values if value is not None):
        return ["" if value is None else str(value) for value in values], False
    texts = ["" if value is None else _format_number(value, digits) for value in values]
    points = [_point_position(text) for text in texts]
    before = max(points, default=0)
    after = max(
        (len(text) - point for text, point in zip(texts, points, strict=True)),
        default=0,
    )
    return [
        (" " * (before - point) + text).ljust(before + after)
        for text, point in zip(texts, points, strict=True)
    ], True


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _format_number(value: object, digits: int) -> str:
    return format(value, f".{digits}g") if isinstance(value, float) else str(value)


def _point_position(text: str) -> int:
    # A number without a decimal point is aligned as if it ended in one.
    position = text.find(".")
    return len(text) if position < 0 else position

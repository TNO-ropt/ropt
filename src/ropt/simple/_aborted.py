"""The value an evaluation returns when it was cut off before finishing."""

from __future__ import annotations

from typing import Final


class Aborted:
    """The type of the [`ABORTED`][ropt.simple.ABORTED] sentinel.

    A single instance exists; compare against
    [`ABORTED`][ropt.simple.ABORTED] with `is` rather than constructing one.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Render as the name callers compare against.

        Returns:
            The sentinel's name.
        """
        return "ABORTED"


ABORTED: Final = Aborted()
"""Returned by an evaluation that was cut off before it finished.

An evaluation is a single batch, so there is no partial answer to hand back the
way an optimization hands back its best point so far: either every vector was
evaluated or the batch was abandoned. This marks the second case.

It appears when [`Session.abort`][ropt.simple.Session.abort] is called, or when
another run on the session raises, while the evaluation is in flight.
"""

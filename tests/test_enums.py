"""Tests for the enumerations that reach the public API."""

from __future__ import annotations

import pytest

from ropt.enums import ExitCode


@pytest.mark.parametrize("code", list(ExitCode))
def test_every_exit_code_has_a_message(code: ExitCode) -> None:
    # The mapping is separate from the members, so a new member added without
    # one would otherwise raise only when a run happened to end that way.
    assert code.message


@pytest.mark.parametrize("code", list(ExitCode))
def test_exit_code_message_is_a_phrase_without_a_trailing_period(
    code: ExitCode,
) -> None:
    # The messages compose into a sentence as well as standing alone, which
    # they only do if they agree on capitalization and punctuation.
    assert code.message[0].isupper()
    assert not code.message.endswith(".")


def test_exit_code_messages_are_distinct() -> None:
    messages = [code.message for code in ExitCode]
    assert len(set(messages)) == len(messages)


def test_exit_code_str_is_the_integer_value() -> None:
    # `IntEnum.__str__` is `int.__str__`, so `message` and `name` are the only
    # ways to report a run; this pins that `message` did not change it.
    assert str(ExitCode.MAX_FUNCTIONS_REACHED) == "2"
    assert f"{ExitCode.MAX_FUNCTIONS_REACHED}" == "2"

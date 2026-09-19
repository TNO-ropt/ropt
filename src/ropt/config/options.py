"""Schema models for describing and validating plugin options.

A plugin declares the options that each of its methods accepts. The schema
turns that declaration into a pydantic model that validates the `options` field
of a configuration section, and into a Markdown table for the documentation.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, HttpUrl, create_model, model_validator

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class OptionsSchemaModel(BaseModel):
    """The schema for the options of every method of a plugin.

    The methods are described by a mapping of method names to
    [`MethodSchemaModel`][ropt.config.options.MethodSchemaModel] objects, each
    describing a method supported by the plugin.

    Attributes:
        methods: A mapping of method names to their schemas.
        common:  Optional method schemas defining options shared by all methods.

    **Example**:
    ```py
    from ropt.config.options import OptionsSchemaModel

    schema = OptionsSchemaModel.model_validate(
        {
            "methods": {
                "method": {"options": {"a": float, "b": int | str}},
            }
        }
    )

    options = schema.get_options_model("method")
    print(options.model_validate({"a": 1.0, "b": 1}))  # a=1.0 b=1
    ```
    """

    methods: dict[str, MethodSchemaModel[Any]]
    common: list[MethodSchemaModel[Any]] = []

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _check_common_options(self) -> Self:
        common_options = {option for item in self.common for option in item.options}
        for method_name, method_schema in self.methods.items():
            if method_schema.exclude - common_options:
                msg = (
                    f"Excluded option(s) for {method_name} not in "
                    f"common: {method_schema.exclude - common_options}."
                )
                raise ValueError(msg)
        return self

    def get_options_model(self, method: str) -> type[BaseModel]:
        """Create a pydantic model for validating the options of one method.

        The method is looked up by name, case-insensitively, and its options are
        combined with the common options that it does not exclude.

        Args:
            method: The name of the method for which to create the options model.

        Returns:
            A model that validates the options of the given method.

        Raises:
            ValueError: If the method is not found in the schema.
        """
        method_schema = next(
            (
                schema
                for name, schema in self.methods.items()
                if name.lower() == method.lower()
            ),
            None,
        )
        if method_schema is None:
            msg = f"Method `{method}` not found in schema."
            raise ValueError(msg)

        options: dict[str, Any] = {
            option: (type_ | None, None)
            for option, type_ in method_schema.options.items()
        }
        for common_options in self.common:
            options.update(
                {
                    option: (type_ | None, None)
                    for option, type_ in common_options.options.items()
                    if option not in method_schema.exclude and option not in options
                }
            )

        def _extra_validator(self: Any) -> Any:  # ruff: ignore[any-type]
            if self.__pydantic_extra__:
                unknown_options = ", ".join(
                    f"`{option}`" for option in self.__pydantic_extra__
                )
                msg = f"Unknown or unsupported option(s): {unknown_options}"
                raise ValueError(msg)
            return self

        validator: Callable[..., Any] = model_validator(mode="after")(_extra_validator)  # type: ignore[assignment]

        return create_model(
            "OptionsModel",
            __config__=ConfigDict(extra="allow"),
            __validators__={"_extra_validator": validator},
            **options,
        )


class MethodSchemaModel(BaseModel, Generic[T]):
    """The schema for the options of a single method.

    Attributes:
        options: A dictionary of option names and their types.
        url:     An optional URL for the plugin.
        exclude: A set of common options to exclude for this method.
        title:   An optional title for common method sections.
        doc:     An optional description of the common section.
    """

    options: dict[str, T]
    url: HttpUrl | None = None
    exclude: set[str] = set()
    title: str | None = None
    doc: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


def _get_common_options(options_schema: OptionsSchemaModel) -> str:
    common_string = ""
    for common_schema in options_schema.common:
        common_options = ", ".join(key for key in common_schema.options)
        if common_schema.url is not None:
            common_options = f"[{common_options}]({common_schema.url})"
        title = common_schema.title or "Common Options"
        common_string += f"**{title}:**\n\n{common_options}\n\n"
        if common_schema.doc:
            common_string += f"{common_schema.doc}\n\n"

    return common_string + dedent("""
    **Method-specific Options:**

    | Method | Options |
    |--------|---------|
    """)


def _new_note(notes: dict[str, str], key: str, note: str) -> int:
    if key not in notes:
        notes[key] = note
    return list(notes).index(key) + 1


def gen_options_table(schema: dict[str, Any]) -> str:
    """Generate a Markdown table documenting plugin options.

    Common options are listed first, followed by a table of method-specific
    options, one row per method. A method whose schema provides a URL is
    hyperlinked to it.

    Args:
        schema: A dictionary representing the schema of plugin options.

    Returns:
        A string containing the documented plugin options.
    """
    options_schema = OptionsSchemaModel.model_validate(schema)

    notes: dict[str, str] = {}

    docstring = _get_common_options(options_schema)

    common_options = {
        option for item in options_schema.common for option in item.options
    }

    for method, method_schema in options_schema.methods.items():
        note_numbers: list[int] = []
        method_options = [
            f"*{key}*" if key in common_options else key
            for key in method_schema.options
        ]

        if set(method_schema.options).intersection(common_options):
            note_numbers.append(
                _new_note(
                    notes,
                    "__override_note__",
                    "Options in *italics* override a common option with a different type or behavior.",
                )
            )

        if method_schema.exclude:
            exclude = ", ".join(f"~~{key}~~" for key in method_schema.exclude)
            note_numbers.append(
                _new_note(
                    notes,
                    "__exclude_note__",
                    "Options with ~~strikethrough~~ indicate a common option that is not supported.",
                )
            )
            method_options.append(exclude)

        method_name = (
            method if method_schema.url is None else f"[{method}]({method_schema.url})"
        )
        if note_numbers:
            method_name += (
                "^" + ",".join(str(note) for note in sorted(note_numbers)) + "^"
            )

        docstring += "|" + method_name + "|" + ", ".join(method_options) + "|\n"

    if notes:
        notes_string = "Notes" if len(notes) > 1 else "Note"
        docstring += (
            f"\n**{notes_string}:**\n\n"
            + "\n".join(
                f"{num}. {text}" for num, text in enumerate(notes.values(), start=1)
            )
            + "\n"
        )

    return docstring

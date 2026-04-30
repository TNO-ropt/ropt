"""This module defines utilities for validating plugin options.

This module provides classes and functions to define and validate options for
plugins. It uses Pydantic to create models that represent the schema of
plugin options, allowing for structured and type-safe configuration.

Classes:
    OptionsSchemaModel: Represents the overall schema for plugin options.
    MethodSchemaModel: Represents the schema for a specific method within a
        plugin, including its name and options.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, HttpUrl, create_model, model_validator

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class OptionsSchemaModel(BaseModel):
    """Represents the overall schema for plugin options.

    This class defines the structure for describing the methods and options
    available for a plugin. The methods are described in a list of
    [`MethodSchemaModel][ropt.config.options.MethodSchemaModel`] objects, each
    describing a method supported by the plugin.

    Attributes:
        methods: A list of method schemas.

    **Example**:
    ```py
    from ropt.config.options import OptionsSchemaModel

    schema = OptionsSchemaModel.model_validate(
        {
            "methods": [
                {
                    "options": {"a": float}
                },
                {
                    "options": {"b": int | str},
                },
            ]
        }
    )

    options = schema.get_options_model("method")
    print(options.model_validate({"a": 1.0, "b": 1}))  # a=1.0 b=1
    ```
    """

    methods: dict[str, MethodSchemaModel[Any]]
    common: MethodSchemaModel[Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _check_common_options(self) -> Self:
        if self.common is not None:
            common_options = set(self.common.options)
            for method_name, method_schema in self.methods.items():
                if method_schema.exclude - common_options:
                    msg = (
                        f"Option(s) {method_schema.exclude - common_options} are "
                        f"excluded from `{method_name}` schema but not defined in `common`."
                    )
                    raise ValueError(msg)
        return self

    def get_options_model(self, method: str) -> type[BaseModel]:
        """Creates a Pydantic model for validating options of a specific method.

        This method dynamically generates a Pydantic model tailored to validate
        the options associated with a given method. It iterates through the
        defined methods, collecting option schemas from those matching the
        specified `method` name. The resulting model can then be used to
        validate dictionaries of options against the defined schema.

        Args:
            method: The name of the method for which to create the options model.

        Returns:
            A Pydantic model class capable of validating options for the specified method.

        Raises:
            ValueError: If the specified method is not found in the schema.
        """
        options: dict[str, Any] | None = None
        for method_name, method_schema in self.methods.items():
            options = {
                option: (type_ | None, None)
                for option, type_ in method_schema.options.items()
            }
            if method_name.lower() == method.lower():
                break

        if options is None:
            msg = f"Method `{method}` not found in schema."
            raise ValueError(msg)

        if self.common is not None:
            options.update(
                {
                    option: (type_ | None, None)
                    for option, type_ in self.common.options.items()
                    if option not in method_schema.exclude and option not in options
                }
            )

        def _extra_validator(self: Any) -> Any:  # noqa: ANN401
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
    """Represents the schema for a specific method within a plugin.

    This class defines the structure for describing one or more methods
    supported by a plugin. It contains a dictionary describing an option for
    this method.

    Attributes:
        options: A dictionary of option names and their types.
        url:     An optional URL for the plugin.
        exclude: A set of common options to exclude for this method.
    """

    options: dict[str, T]
    url: HttpUrl | None = None
    exclude: set[str] = set()

    model_config = ConfigDict(extra="forbid", frozen=True)


def gen_options_table(schema: dict[str, Any]) -> str:
    """Generates a Markdown table documenting plugin options.

    This function takes a schema dictionary, validates it against the
    [`OptionsSchemaModel`][ropt.config.options.OptionsSchemaModel], and then
    generates a Markdown document that summarizes the available methods and
    their options. Common options are listed first, followed by a table of
    method-specific options. Each row in the table represents a method, and the
    columns list the method's name and its configurable options. If a URL is
    provided for a method, the method name will be hyperlinked to that URL in
    the table.

    Args:
        schema: A dictionary representing the schema of plugin options.

    Returns:
        A string containing the documented plugin options.
    """
    options_schema = OptionsSchemaModel.model_validate(schema)

    common_string = ""
    common_options = set()
    if options_schema.common is not None:
        options = ", ".join(key for key in options_schema.common.options)
        if options_schema.common.url is not None:
            options = f"[{options}]({options_schema.common.url})"
        common_string = f"**Common Options:**\n\n{options}\n"
        common_options = set(options_schema.common.options)

    docstring = dedent("""
    **Method-specific Options:**

    | Method | Options |
    |--------|---------|
    """)

    override_note: int | None = None
    exclude_note: int | None = None
    notes: list[str] = []
    for method, method_schema in options_schema.methods.items():
        method_string = (
            method if method_schema.url is None else f"[{method}]({method_schema.url})"
        )
        options = ", ".join(
            f"*{key}*" if key in common_options else key
            for key in method_schema.options
        )
        if set(method_schema.options).intersection(common_options):
            if override_note is None:
                override_note = len(notes) + 1
                notes.append(
                    f"{override_note}. Options in *italics* override a common option "
                    "with a different type or behavior."
                )
            method_string += f"^{override_note}^"

        if method_schema.exclude:
            exclude = ", ".join(f"~~{key}~~" for key in method_schema.exclude)
            if exclude_note is None:
                exclude_note = len(notes) + 1
                notes.append(
                    f"{exclude_note}. Options with ~~strikethrough~~ indicate a common "
                    "option that is not supported."
                )
            method_string += f"^{exclude_note}^"
            options_string = f"{options}, {exclude}"
        else:
            options_string = options
        docstring += f"|{method_string}|{options_string}|\n"

    if notes:
        notes_string = "Notes" if len(notes) > 1 else "Note"
        docstring += f"\n**{notes_string}:**\n\n" + "\n".join(notes) + "\n"

    return common_string + docstring

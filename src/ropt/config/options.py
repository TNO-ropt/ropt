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
from typing import Any, Callable, Generic, TypeVar, Union

from pydantic import BaseModel, ConfigDict, HttpUrl, create_model, model_validator

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

    model_config = ConfigDict(extra="forbid")

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
        """
        options: dict[str, Any] | None = None
        for method_name, method_schema in self.methods.items():
            options = {
                option: (Union[type_, None], None)
                for option, type_ in method_schema.options.items()
            }
            if method_name.lower() == method.lower():
                break
        if options is None:
            msg = f"Method `{method}` not found in schema."
            raise ValueError(msg)

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
        options: A list of option dictionaries.
        url:     An optional URL for the plugin.
    """

    options: dict[str, T]
    url: HttpUrl | None = None

    model_config = ConfigDict(extra="forbid")


def gen_options_table(schema: dict[str, Any]) -> str:
    """Generates a Markdown table documenting plugin options.

    This function takes a schema dictionary, validates it against the
    [`OptionsSchemaModel`][ropt.config.options.OptionsSchemaModel], and then
    generates a Markdown table that summarizes the available methods and their
    options. Each row in the table represents a method, and the columns list the
    method's name and its configurable options. If a URL is provided for a
    method, the method name will be hyperlinked to that URL in the table.

    Args:
        schema: A dictionary representing the schema of plugin options.

    Returns:
        A string containing a Markdown table that documents the plugin options.
    """
    OptionsSchemaModel.model_validate(schema)

    docstring = dedent("""
    | Method | Method Options |
    |--------|----------------|
    """)

    for method, method_schema in schema["methods"].items():
        url = MethodSchemaModel.model_validate(method_schema).url
        options = ", ".join(key for key in method_schema["options"])
        if url:
            docstring += f"|[{method}]({url})|{options}|\n"
        else:
            docstring += f"|{method}:|{options}|\n"

    return docstring

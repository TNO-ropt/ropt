"""This module defines utilities for validating plugin options.

This module provides classes and functions to define and validate options for
plugins. It uses Pydantic to create models that represent the schema of
plugin options, allowing for structured and type-safe configuration.

Classes:
    OptionsSchemaModel: Represents the overall schema for plugin options,
        including metadata like type, description, URL, and a list of methods.
    MethodSchemaModel: Represents the schema for a specific method within a
        plugin, including its name, URL, description, and options.
    OptionSchema: Represents the schema for a single option within a method,
        including its name and type.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    create_model,
    field_validator,
    model_validator,
)

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
                    "name": "method",
                    "options": [
                        {"name": "a", "type": float},
                    ],
                },
                {
                    "name": "method",
                    "options": [
                        {"name": "b", "type": int | str},
                    ],
                },
            ]
        }
    )

    options = schema.get_options_model("method")
    print(options.model_validate({"a": 1.0, "b": 1}))  # a=1.0 b=1
    ```
    """

    methods: list[MethodSchemaModel]

    model_config = ConfigDict(extra="forbid")

    def get_options_model(self, method: str) -> type[BaseModel]:
        """Creates a Pydantic model for validating options of a specific method.

        This method dynamically generates a Pydantic model tailored to validate
        the options associated with a given method. It iterates through the
        defined methods, collecting option schemas from those matching the
        specified `method` name. The resulting model can then be used to
        validate dictionaries of options against the defined schema.

        The method returns a model that validates these option types:

        - Optimizer:   Options that are already handled via optimizer options
                       are marked by the `optimizer` type. An error is raised if
                       such an option is encountered.
        - Unsupported: Options that are not supported are marked by the
                       `unsupported` type. An error is raised if such an option
                       is encountered.
        - Supported:   Regular options with a specified type. The values of
                       these are validated against the specified type.
        - Unknown:     Unknown options are reported by raising an error.

        Args:
            method: The name of the method for which to create the options model.

        Returns:
            A Pydantic model class capable of validating options for the specified method.
        """
        options = {}
        validators: dict[str, Any] = {}

        def _optimizer_validator(cls: Any, _1: str, info: ValidationInfo) -> Any:  # noqa: ANN401, ARG001
            msg = f"Option `{info.field_name}` should be handled via a general optimizer option."
            raise ValueError(msg)

        def _unsupported_validator(cls: Any, _1: str, info: ValidationInfo) -> Any:  # noqa: ANN401, ARG001
            msg = f"Option `{info.field_name}` is not supported."
            raise ValueError(msg)

        for item in self.methods:
            if item.name == method or method in item.name:
                for option in item.options:
                    match option.type_:
                        case "optimizer":
                            validators[f"_{option.name}-validator"] = field_validator(
                                option.name, mode="after"
                            )(_optimizer_validator)
                            options[option.name] = (Any, None)
                        case "unsupported":
                            validators[f"_{option.name}-validator"] = field_validator(
                                option.name, mode="after"
                            )(_unsupported_validator)
                            options[option.name] = (Any, None)
                        case _:
                            options[option.name] = (Union[option.type_, None], None)

        def _extra_validator(self: Any) -> Any:  # noqa: ANN401
            if self.__pydantic_extra__:
                unknown_options = ", ".join(
                    f"`{option}`" for option in self.__pydantic_extra__
                )
                msg = f"Unknown options: {unknown_options}"
                raise ValueError(msg)
            return self

        validators["_extra_validator"] = model_validator(mode="after")(_extra_validator)

        return create_model(  # type: ignore[no-any-return, call-overload]
            "OptionsModel",
            __config__=ConfigDict(extra="allow"),
            __validators__=validators,
            **options,
        )


class MethodSchemaModel(BaseModel):
    """Represents the schema for a specific method within a plugin.

    This class defines the structure for describing one or more methods
    supported by a plugin. It contains the name of the method and a list of
    [`OptionSchema`][ropt.config.options.OptionSchema] objects, each describing
    an option for this method.

    Attributes:
        name:    The name of one or multiple methods that support give options.
        options: A list of option schemas.
    """

    name: str | tuple[str, ...]
    options: list[OptionSchema[Any]]

    model_config = ConfigDict(extra="forbid")


class OptionSchema(BaseModel, Generic[T]):
    """Represents the schema for a single option within a method.

    This class defines the structure for describing a single option for a
    method. It includes the option's name and its type.

    Attributes:
        name:  The name of the option.
        type_: The type of the option.
    """

    name: str
    type_: T | Literal["unsupported", "optimizer"] = Field(alias="type")

    model_config = ConfigDict(extra="forbid")

# Transforms

Transforms map values between the user domain and the optimizer domain. Each
transform is initialized with a boolean mask and applied in sequence during
optimization. The base classes define the contract; the built-in defaults
provide linear scale/offset transforms.

See [Transforms](../optimizer_setup/variable_transforms.md) for usage, configuration, and
implementation guidance.

::: ropt.transforms
::: ropt.transforms.VariableTransform
::: ropt.transforms.default.DefaultVariableTransform


# Evaluation Results

Every batch evaluation produces a tuple of [`Results`][ropt.results.Results]
objects: a [`FunctionResults`][ropt.results.FunctionResults] for objective and
constraint values, and/or a [`GradientResults`][ropt.results.GradientResults]
for gradient estimates. Each is a frozen container of
[`ResultField`][ropt.results.ResultField] sub-objects holding NumPy arrays
with axis-name metadata.

See [Working with Results](../running/results.md) for a tour of the access
patterns.

::: ropt.results
    options:
        members: []
::: ropt.results.Results
::: ropt.results.AxisMetadata
::: ropt.results.ResultField
::: ropt.results.FunctionResults
::: ropt.results.ScaledFunctionResults
::: ropt.results.GradientResults
::: ropt.results.ScaledGradientResults
::: ropt.results.Functions
    options:
        members: [from_scaled]
::: ropt.results.Gradients
    options:
        members: [from_scaled]
::: ropt.results.FunctionEvaluations
    options:
        members: [create]
::: ropt.results.GradientEvaluations
    options:
        members: [create]
::: ropt.results.Realizations
    options:
        members: []
::: ropt.results.ConstraintInfo
    options:
        members: [create, from_scaled]
::: ropt.results.results_to_pandas
::: ropt.results.results_to_polars


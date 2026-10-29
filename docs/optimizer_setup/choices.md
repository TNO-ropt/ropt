# Optimization Choices

An optimization configuration can describe more than the variables and the
function to evaluate. You can also choose how realizations contribute, how
quantities are scaled, how gradients are estimated, and which algorithm to use.
This page surveys those choices and links to the detailed guides. For the
smallest complete optimization, see the [Quickstart](../getting_started/quickstart.md).

This page covers what is optimized and how the optimizer works. Where the
evaluations run — evaluation pools, sessions, parallel and multiple
optimizations — is covered in [Running Optimizations](../running/running.md).

## What can the optimizer change?

The `variables` section sets the number of variables and can give each one
bounds, hold selected variables fixed, or mark them as integers. Bounds restrict
the search space; fixed variables retain their starting values. Integer
variables require a method that supports them.

The starting point is an argument of [`optimize`][ropt.optimize], not part of
the configuration. To continue from an earlier result, or to reuse values
already evaluated at the starting point, see
[Restarting from the Best Point](../running/restart.md) and
[Reusing Results at the Starting Point](../running/initial_values.md).

See [Variables](variables.md) and
[Discrete and Mixed-Integer Variables](discrete.md).

## What should the optimizer improve?

By default, `ropt` minimizes one objective. You can define multiple objectives
and combine them with objective weights, or mark individual objectives for
maximization. Objective weights set the balance between objectives; their
effect also depends on the objective scales.

See [Objectives](objectives.md).

## What restrictions must a solution satisfy?

Bounds restrict variables directly. Linear constraints are calculated from
configured coefficients, while nonlinear constraints are returned by the
evaluation function. A solution is feasible when its constraint violations are
within the configured tolerance. Check that the selected optimization method
supports the constraints you need.

See [Constraints](constraints.md) and the
[backend capability reference](../reference/backend.md).

## How should realizations contribute?

Realization weights express the relative contribution of ensemble members.
They are distinct from objective weights, which balance objectives. You can
also use **realization filters** to select or reweight realizations for an
evaluation, for example to focus on poor outcomes.

A **function estimator** combines the per-realization values into the value
used by the optimizer. The default is a weighted mean; other estimators can
target quantities such as variability. Filters change which realizations
contribute, while estimators define how their values are combined.

See [Ensemble-Based Optimization](../getting_started/ensemble.md),
[Realization Filters](realization_filters.md), and
[Function Estimators](function_estimators.md).

## Which quantities need scaling?

Scaling is configured separately for variables, objectives, and constraints:

- **Variables** can be mapped to a different scale and offset before they reach
  the optimizer. This is useful when variable magnitudes differ substantially.
- **Objectives** can be scaled and offset. With multiple objectives, scales
  affect their balance together with objective weights. Automatic objective
  scaling estimates a factor from the first batch; it does not rebalance
  objectives.
- **Constraints** can be scaled before they reach the optimizer, using fixed
  scales or `auto_scale`. For nonlinear constraints, `auto_scale` is set per
  constraint and estimates each scale from the first batch. For linear
  constraints, it is one setting for all equations and estimates the scale of
  each equation from its coefficients, not from evaluations.

### Choose scales from a preliminary evaluation

!!! tip "An alternative to `auto_scale`"

    `auto_scale` applies a fixed rule to the first batch. Choosing scales
    yourself lets you apply a different rule: for example, scale each
    objective individually rather than with one common factor, or base the
    scales on several representative points. You can also choose objective
    offsets; constraints do not support offsets.

    For objectives and nonlinear constraints, evaluate a point with
    [`evaluate`][ropt.evaluate], or several points with
    [`evaluate_batch`][ropt.evaluate_batch]. The points can be anywhere you
    consider representative; they do not have to include the starting point of
    the optimization. Calculate the scales, and any objective offsets, from the
    unscaled `evaluations.objectives` and `evaluations.constraints` in the
    returned [`FunctionResults`][ropt.results.FunctionResults], set them in the
    configuration, and disable `auto_scale`.

    Passing `f0` is optional. If one of the evaluated points is the starting
    point of the optimization, pass its `FunctionResults` as `f0` to
    [`optimize`][ropt.optimize] to avoid evaluating that point again: `ropt`
    reuses the recorded values and recomputes the aggregates with the new
    settings. Results at other points cannot be reused. See
    [Reusing Results at the Starting Point](../running/initial_values.md).

    Linear constraints are not returned by the evaluation function: `ropt`
    calculates them from their coefficients, and their `auto_scale` estimates
    the scale of each equation from those coefficients. See
    [Scaling linear equations](configuration_sections.md#linear-constraint-scales).

See [Scaling the variables](variables.md#scaling),
[Weights and scales](objectives.md#weights-and-scales-determine-the-trade-off-together),
[Scaling constraints](configuration_sections.md#constraint-scales), and
[Scaling linear equations](configuration_sections.md#linear-constraint-scales)
for the field details.

## How should gradients be estimated?

Gradient-based methods use `ropt`'s stochastic gradient estimates. Gradient-free
methods do not use these settings. The main choices are:

- **Perturbations:** their size, whether the size is absolute or relative to
  the bounds, and how perturbed values outside the bounds are corrected.
- **Number of perturbations:** more perturbations can improve the estimate but
  require more evaluations.
- **Per-realization or merged estimates:** by default, a gradient is estimated
  for each realization and the results are combined. With
  `merge_realizations`, all realizations contribute to a single estimate, which
  needs fewer perturbations. The `stddev` function estimator requires
  per-realization gradients.
- **Samplers:** how perturbations are drawn, and whether all realizations use
  the same perturbed values.
- **Seed:** the same seed reproduces the same perturbations. Concurrent or
  nested optimizations need different seeds to draw different perturbations.
- **Evaluation policy:** whether function and gradient evaluations follow the
  optimizer's requests, always run as separate batches, or the gradient is
  evaluated with every function evaluation. Separate evaluation allows
  realization filters to reduce the gradient work.

See [Stochastic Gradients](gradients.md),
[Samplers](samplers.md), and
[Gradient-based or gradient-free](optimizer.md#gradient-based-or-gradient-free).

## What happens when evaluations fail?

An evaluation function indicates a failed realization by returning `NaN`. By
default, every realization and every perturbation must succeed. Two settings
allow some failures:

- `gradient.perturbation_min_success` sets how many perturbations of a
  realization must succeed. If fewer succeed, that realization counts as failed
  for the gradient.
- `realizations.realization_min_success` sets how many realizations must
  succeed. Failed realizations are excluded from the combined value.

If fewer realizations succeed than required, the optimization ends with the
exit code `TOO_FEW_REALIZATIONS`. See
[Tolerating failures](gradients.md#tolerating-failures) and
[Exit Codes](../results/exit_codes.md). An exception raised by the evaluation
function is handled differently; see [Failures and Aborts](../running/failures.md).

## Which algorithm should run, and when should it stop?

The backend selects the optimization algorithm. Methods differ in their support
for variable types and constraints, so check the backend capabilities before
choosing one. You can also set limits on evaluation batches, function
evaluations, iterations, or convergence.

See [Choosing an Optimizer](optimizer.md) and the
[backend capability reference](../reference/backend.md).

For the names, defaults, and types of every configuration field, see
[Configuration Sections](configuration_sections.md).

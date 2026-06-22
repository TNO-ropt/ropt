# Tutorials

This section contains step-by-step tutorials that guide you through complete
working examples. Each tutorial builds on concepts from the [Usage](../usage/background.md)
documentation and demonstrates practical applications of `ropt`.

## Rosenbrock Function Tutorials

These tutorials demonstrate optimization of the multi-dimensional Rosenbrock
function, progressing from a simple deterministic case to stochastic
optimization with different implementation approaches:

- **[Deterministic Optimization](rosenbrock_deterministic.md)** — The simplest
  example: optimizing the classic Rosenbrock function with fixed parameters.
  Start here to understand the basics.

- **[Stochastic Optimization](rosenbrock_basic.md)** — Introduces uncertainty
  by sampling parameters across multiple realizations. Uses
  [`BasicOptimizer`][ropt.workflow.BasicOptimizer] with a batch evaluation
  callback.

- **[Function Evaluator](rosenbrock_function.md)** — Using
  [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator] with a
  per-evaluation function callback. This approach is simpler to write when you
  don't need to handle batches.

- **[Workflow Framework](rosenbrock_workflow.md)** — Using the workflow
  framework directly with
  [`OptimizationStep`][ropt.workflow.compute_steps.OptimizationStep]. This
  approach offers more control and flexibility for complex workflows.

- **[Constrained Optimization](rosenbrock_constrained.md)** — Adding linear
  and nonlinear constraints to the optimization problem.

## Prerequisites

Before starting these tutorials, you should be familiar with:

1. [Installation](../usage/installation.md) — How to install `ropt`
2. [Background](../usage/background.md) — Core concepts of ensemble-based optimization
3. [Quickstart](../usage/quickstart.md) — A minimal working example

Each tutorial provides complete, runnable code that you can copy and experiment
with. The full source code for all examples is available in the
[examples directory](https://github.com/TNO-ropt/ropt/tree/main/examples) of the
repository.

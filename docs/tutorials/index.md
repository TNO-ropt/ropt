# Tutorials

These tutorials walk through the runnable scripts in the
[examples](https://github.com/TNO-ropt/ropt/tree/main/examples) folder. Each one
starts with a link to the full script, then explains it a few lines at a time.
The scripts are short, so keep the file open next to the tutorial.

More tutorials will be added over time.

## Simple API

These use the [simple API](../running/running.md) and are the best place to start.

- [Your First Optimization](optimize.md) — run one optimization.
- [Ensemble Optimization](ensemble.md) — optimize over uncertain realizations.
- [Constrained Optimization](constrained.md) — add constraints.
- [Mixed-Integer Optimization](discrete.md) — integer variables with
  differential evolution.
- [Restarting from the Best Point](restart.md) — restart from the previous
  best point, collecting every result with a handler.

## Low-level API

For advanced users who need full control over the optimization workflow.

- [Building a Workflow](workflow.md) — assemble compute steps and event handlers
  by hand.

# Exit Codes

Every run reports why it ended. The reason is an
[`ExitCode`][ropt.enums.ExitCode], carried on the object the run returns
rather than raised:

```python
from ropt.simple import optimize

result = optimize(config, x0, objective)
print(result.exit_code.name)
```

A run either **stops** or is **aborted**, and the reason indicates which. It
stops when a condition on the run is met — the optimizer converged, a budget ran
out, a [`report` callback](../running/running.md#stopping-early-from-the-callback)
returned `True` — so it ends at a point someone declared acceptable. It is
aborted when something cuts it off without consulting the run at all, and what
comes back is then whatever it had reached rather than a chosen endpoint.

| Exit code               | Meaning                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `FINISHED`              | The run terminated normally. For an optimization this means the backend ended it, by converging or on a limit of its own such as `max_iterations`. |
| `MAX_FUNCTIONS_REACHED` | The configured maximum number of function evaluations was reached.                                   |
| `MAX_BATCHES_REACHED`   | The configured maximum number of evaluation batches was reached.                                     |
| `STOPPED`               | A `report` callback returned `True`, ending the run at the next evaluation boundary.                 |
| `TOO_FEW_REALIZATIONS`  | Too few realizations were evaluated successfully to form an aggregate.                               |
| `ABORTED`               | [`Session.abort`](../running/running.md#stopping-from-outside) cut the run off.                      |
| `ABORTED_ON_ERROR`      | Another run on the same session raised, and brought this one down with it.                           |
| `EXECUTOR_SHUT_DOWN`    | The pool the run was evaluating on could no longer run the work, which in practice means the interpreter was shutting down under it. |
| `UNKNOWN`               | The zero value of the enumeration. No run reports it.                                                |

The first five are stops and the next three aborts; `UNKNOWN` is neither.
`TOO_FEW_REALIZATIONS` ends a run at an evaluation boundary like the other
stops, but unlike them it follows from a failure: one failed realization is
enough to trigger it unless
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations)
is lowered.

## What each kind of run reports

An **optimization** can end for any of the reasons above.

An **evaluation** is a single batch with no optimizer loop around it, so it
reports only `FINISHED`, `ABORTED` or `ABORTED_ON_ERROR`. It is also all or
nothing: either every vector was evaluated, or the batch was abandoned and
`results` is empty. An abort that arrives once the batch has finished costs it
nothing, and the evaluation reports `FINISHED`.

## An exit code is not a result

The two are independent. `results` is `None` when no feasible result was ever
recorded, whatever the reason the run ended, and a run that ends early still
returns the best result it had reached before that. A run can therefore report
`FINISHED` and still have nothing on `results`, because only a result satisfying
every constraint to within `constraint_tolerance` can be returned as the best
one — `1e-10` unless you pass another, and it applies to the bounds and the
linear constraints as well as the nonlinear ones. See
[When something goes wrong](../running/running.md#when-something-goes-wrong).

## Work that has no result object

[`WorkerPool.offload`][ropt.simple.WorkerPool.offload] returns whatever its
callables return, so there is nowhere to carry a reason. A call abandoned by an
abort raises [`AbortedError`][ropt.exceptions.AbortedError] instead, whose
`exit_code` attribute distinguishes an abort that was asked for from one
another run caused.

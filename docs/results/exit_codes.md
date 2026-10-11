# Exit Codes

Every optimization reports why it ended. The reason is an
[`ExitCode`][ropt.enums.ExitCode], carried on the result the optimization
returns:

```python
from ropt import optimize

result = optimize(config, x0, objective)
print(result.exit_code.message)
```

Each code carries a short description in
[`message`][ropt.enums.ExitCode.message], which is what to print or log when
reporting it. Use `name` where the identifier itself is wanted, and compare
against the member to branch on it. Note that `str()` and an f-string give the
integer value, as they do for any `IntEnum`:

```python
code = result.exit_code
code.message  # 'Maximum number of function evaluations reached'
code.name     # 'MAX_FUNCTIONS_REACHED'
str(code)     # '2'
```

An optimization either **stops** or is **aborted**. It stops when a condition on
the optimization is met — the optimizer converged, a budget ran out, a
[`report` callback](../running/running.md#stopping-early-from-the-callback)
returned `True` — so it ends at a point that satisfied a stated criterion, and
returns its result with the code. It is aborted when something ends it
regardless of its state, and it then raises
[`AbortedError`][ropt.exceptions.AbortedError], whose `exit_code` attribute
holds the code:

```python
from ropt import AbortedError

try:
    result = pool.optimize(config, x0, objective)
except AbortedError as error:
    print(error.exit_code.message)
```

| Exit code               | Meaning                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `FINISHED`              | The optimizer ended the optimization, by converging or on a limit of its own such as `max_iterations`. |
| `MAX_FUNCTIONS_REACHED` | The configured maximum number of function evaluations was reached.                                   |
| `MAX_BATCHES_REACHED`   | The configured maximum number of evaluation batches was reached.                                     |
| `STOPPED`               | A `report` callback returned `True`, ending the optimization at the next evaluation boundary.        |
| `TOO_FEW_REALIZATIONS`  | Too few realizations were evaluated successfully to form an aggregate.                               |
| `ABORTED`               | The task was aborted from outside, as closing a session, or a `KeyboardInterrupt`, does to a task still under way. |
| `ABORTED_ON_ERROR`      | Another [task](../running/failures.md) raised, and this one was aborted with it.                    |
| `USER_ABORT`            | [`Session.abort`](../running/running.md#stopping-from-outside) aborted the task.                     |
| `EXECUTOR_SHUT_DOWN`    | The pool the task was evaluating on could no longer run the work, which in practice means the interpreter was shutting down under it. |
| `UNKNOWN`               | The zero value of the enumeration. No task reports it.                                               |

The first five are stops, returned on the result, and the next four are aborts,
raised with `AbortedError`; `UNKNOWN` is neither. `TOO_FEW_REALIZATIONS` ends an
optimization at an evaluation boundary like the other stops, but unlike them it
follows from a failure: one failed realization is enough to trigger it unless
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations)
is lowered.

## What each kind of task reports { #what-each-kind-of-run-reports }

An **optimization** returns any of the stop codes above, and raises with any of
the abort codes.

An **evaluation** is a single batch with no optimizer loop around it, so it has
no criterion to stop on and its result carries no exit code. It is all or
nothing: either every vector was evaluated and the evaluation returns them, or
it was aborted and raises `AbortedError`. An abort that arrives once the batch
has finished leaves it unaffected.

A **`run` call** returns whatever its callables return, and raises
`AbortedError` when it is aborted before all of them have finished.

## An exit code is not a result

The two are independent. `results` is `None` when no feasible result was ever
recorded, whatever the code the optimization ended with, and an optimization
that stops early still returns the best result it had reached before that. An
optimization can therefore report `FINISHED` and still have nothing on
`results`, because only a result satisfying every constraint to within
`constraint_tolerance` can be returned as the best one — `1e-10` unless you pass
another, and it applies to the bounds and the linear constraints as well as the
nonlinear ones. See
[When something goes wrong](../running/running.md#when-something-goes-wrong).

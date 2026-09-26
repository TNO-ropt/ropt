# Implementing a Component

Each component type is an abstract base class providing a public surface and
asking you for one or two private methods.

| Base class                                                          | You implement                    | Callers use                     |
| ------------------------------------------------------------------- | -------------------------------- | ------------------------------- |
| [`Evaluator`][ropt.components.evaluators.Evaluator]                   | `_eval`                          | `eval`                          |
| [`ComputeStep`][ropt.components.compute_steps.ComputeStep]            | `_run`                           | `run`                           |
| [`EventHandler`][ropt.components.event_handlers.EventHandler]         | `event_types`, `_handle_event`   | `handle_event`                  |
| [`ExecutorBase`][ropt.components.executors.ExecutorBase]              | `start`, `_cleanup`              | `submit`, `is_running`, `cancel` |

For the first three the public method holds a concurrency guard, which is what
turns concurrent misuse into a
[`WorkflowError`][ropt.exceptions.WorkflowError] rather than corrupted state, so
override the private method and never the public one.

A backend, sampler, realization filter or function estimator is a **plugin**,
not a component: it is selected by a method string and discovered through an
entry point. See [Writing a Plugin](writing_plugins.md).

## Evaluator

`_eval` receives the batch of variable vectors and the
[`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext], and returns
an [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult]. `eval`
asserts that return type, so a protocol violation surfaces at the boundary
rather than inside the ensemble code.

```python
class MyEvaluator(Evaluator):
    def _eval(self, variables, context):
        return EvaluationBatchResult(objectives=..., batch_id=...)
```

A wrapping evaluator calls the inner evaluator's public `eval`, so the inner
guard applies independently.

## ComputeStep

`_run` receives the [`EnOptContext`][ropt.context.EnOptContext] and the initial
variables, and returns whatever the step produces. Two obligations:

- Emit events with `self._emit_event`, so attached handlers see the run. The
  types are listed under
  [Event types](workflows.md#event-types).
- Poll `self.stopped` at points where stopping is meaningful. `run` clears the
  flag before each call, so a step is reusable after a stopped run.

## EventHandler

`event_types` declares which events reach `_handle_event`; the dispatch machinery
filters on it, so a handler is never called for a type it did not ask for.
Expose accumulated state through `__getitem__`, which is the convention the
built-in handlers follow and what `handler[key]` reads.

`_handle_event` runs on the thread that emitted the event, under a lock the
handler takes for every call. A slow `_handle_event` therefore delays the
emitting run, and every other run waiting on the same handler. See
[Using handlers safely](workflows.md#using-handlers-safely).

## Executor

Subclass [`ExecutorBase`][ropt.components.executors.ExecutorBase] rather than
[`Executor`][ropt.components.executors.Executor]: it provides
[`run`][ropt.components.executors.Executor.run],
[`close`][ropt.components.executors.Executor.close], the `closed` flag, the
refusal of a caller that is one of its own workers, and the split of a batch
into bundles.

You implement two methods. `_run_bundles` runs the bundles and passes each
result to `store` as it arrives; it must release whatever the batch started
before it returns, including when `store` raises. `_release` releases the
executor's own resources, and runs with the lock not held.

```python
class MyExecutor(ExecutorBase):
    def _run_bundles(self, bundles, store):
        for index, bundle in enumerate(bundles):
            results = [
                item.function(*item.args, **item.kwargs) for item in bundle
            ]
            store(index, results)

    def _release(self) -> None:
        ...                       # release the executor's resources
```

Every bundle ends in exactly one of three ways, and the choice determines
whether the executor survives:

| Outcome | Passed to `store` | Effect |
| --- | --- | --- |
| The functions returned | the list of their results | Normal result. |
| The machinery failed | an [`ExecutorFailure`][ropt.components.executors.ExecutorFailure] | Raised as an [`ExecutionError`][ropt.exceptions.ExecutionError] by the evaluator; the executor keeps running. |
| A function raised | `store` is not called; raise the exception out of `_run_bundles` | The exception is re-raised unchanged in the caller; the executor keeps running. |

An [`ExecutorFailure`][ropt.components.executors.ExecutorFailure] is a value, not
an exception — store it, never raise it. Confusing the first two rows is the
common mistake: a bug in user code stored as an `ExecutorFailure` would be
reported as broken infrastructure, and a dead worker raised out of
`_run_bundles` would surface as a user error. See
[Error handling](parallel.md#error-handling) for the distinction.

Two further rules. State of the executor's own that must stay in step with
closing goes under `_lock`, which `ExecutorBase` exposes for that purpose; the
`_on_close` hook runs with it held. `run` refuses work sent from the executor's
own workers — a caller that waits there occupies a worker its own batch needs —
by reading `_thread_state.running_work_item`, a thread-local flag. An executor
whose workers are threads in this process sets that flag for as long as a work
item runs on one; one whose workers run elsewhere never sets it.

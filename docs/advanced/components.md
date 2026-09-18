# Implementing a Component

The four component types share one shape: a public method holding a concurrency
guard, and a private abstract method you implement. Override the private one
only — the guard is what turns concurrent misuse into a
[`WorkflowError`][ropt.exceptions.WorkflowError] instead of corrupted state.

| Base class                                                          | You implement                    | Callers use                     |
| ------------------------------------------------------------------- | -------------------------------- | ------------------------------- |
| [`Evaluator`][ropt.components.evaluators.Evaluator]                   | `_eval`                          | `eval`                          |
| [`ComputeStep`][ropt.components.compute_steps.ComputeStep]            | `_run`                           | `run`                           |
| [`EventHandler`][ropt.components.event_handlers.EventHandler]         | `event_types`, `_handle_event`   | `handle_event`                  |
| [`ExecutorBase`][ropt.components.executors.ExecutorBase]              | `start`, `_cleanup`              | `submit`, `is_running`, `cancel` |

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
        return EvaluationBatchResult(batch_id=..., objectives=...)
```

Wrapping evaluators — [`CachedEvaluator`][ropt.components.evaluators.CachedEvaluator]
is the built-in example — call the inner evaluator's public `eval`, so the inner
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

`_handle_event` must not block for long when the handler is registered with an
[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher]: the
dispatcher serializes events, so a slow handler delays every run sharing it. Use
`run_in_thread=True` for blocking work; see
[Event throughput](workflows.md#event-throughput).

## Executor

Subclass [`ExecutorBase`][ropt.components.executors.ExecutorBase] rather than
[`Executor`][ropt.components.executors.Executor]: it provides `submit`,
`is_running`, `cancel`, submission ownership, and the deadlock guard that
refuses work submitted from the executor's own workers.

You implement two methods. `start` must call `_begin_start` **before** creating
any resources and `_finish_start` once they are in place — the first guards
against a double start and rebinds the asyncio primitives to the current loop,
the second publishes the loop and waits until the executor is ready.
`_cleanup` releases those resources, and runs on the loop thread.

```python
class MyExecutor(ExecutorBase):
    async def start(self, task_group):
        self._begin_start()
        ...                       # create resources
        task_group.create_task(self._run_worker())
        await self._finish_start(task_group)

    def _cleanup(self) -> None:
        ...                       # release them
        self._cleanup_submissions()
```

**The delivery contract is the part to get right.** Every work item ends in
exactly one of three ways, and the choice decides whether the executor survives:

| Outcome | Call | Effect |
| --- | --- | --- |
| The function returned | `_deliver(submission, work_item, value)` | Normal result. |
| The machinery failed | `_deliver(submission, work_item, ExecutorFailure(...))` | Raised as an [`ExecutionError`][ropt.exceptions.ExecutionError] by the evaluator; the executor keeps running. |
| The function raised | `_fail(submission, exc)` | The exception is re-raised unchanged in the caller; the executor keeps running. |

An [`ExecutorFailure`][ropt.components.executors.ExecutorFailure] is a value, not
an exception — deliver it, never raise it. Confusing the two rows is the common
mistake: a bug in user code delivered as an `ExecutorFailure` would be reported
as broken infrastructure, and a dead worker passed to `_fail` would surface as a
user error. See
[Error handling](parallel.md#error-handling) for the distinction.

Two further rules. Skip a submission whose `is_finished` is already `True` — its
caller has left, so running its work items only occupies a worker. And call
`_cleanup_submissions` from `_cleanup`, which aborts whatever is outstanding so
no caller is left blocked in
[`collect`][ropt.components.executors.Submission.collect].

# Building a Workflow

*Audience: advanced users who need full control over the optimization.*

The [simple API](../running/running.md) covers most cases. When you need more —
custom event handling, several optimizers, or nested runs — you assemble the
low-level workflow components yourself. This tutorial follows
[examples/advanced/workflow.py](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/workflow.py).

See [Optimization Workflows](../workflows/workflows.md) for the full reference on the
components used here.

## The evaluator

At the low level the evaluation callback receives the whole batch of variable
vectors at once (a 2-D array) and returns an
[`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult]:

```python
def rosenbrock(variables, context, a, b):
    objectives = np.zeros((variables.shape[0], 1))
    for v_idx, r in enumerate(context.realizations):
        ...
    return EvaluationBatchResult(objectives=objectives)
```

Wrap it in a [`BatchEvaluator`][ropt.components.evaluators.BatchEvaluator]:

```python
from ropt.components.evaluators import BatchEvaluator

evaluator = BatchEvaluator(callback=partial(rosenbrock, a=a, b=b))
```

See [Writing Evaluation Callbacks](../workflows/evaluation_callbacks.md) for the
batch callback signature and the other evaluators.

## The compute step and its handlers

An [`OptimizationStep`][ropt.components.compute_steps.OptimizationStep] runs the
optimization. You attach event handlers to it: a
[`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] to keep the best
result, and a [`CallbackHandler`][ropt.components.event_handlers.CallbackHandler]
to report progress:

```python
from ropt.components.compute_steps import OptimizationStep
from ropt.components.event_handlers import CallbackHandler, ResultsHandler
from ropt.enums import EnOptEventType

step = OptimizationStep(evaluator=evaluator)
result_handler = ResultsHandler()
step.add_event_handler(result_handler)
step.add_event_handler(
    CallbackHandler(callback=report, event_types={EnOptEventType.FINISHED_EVALUATION})
)
```

## Run it and read the result

The step takes the variables and an
[`EnOptContext`][ropt.context.EnOptContext] built from the config. The best result
is on the handler:

```python
from ropt.context import EnOptContext

step.run(variables=INITIAL_VALUES, context=EnOptContext.model_validate(CONFIG))
best = result_handler.result
```

## Next

- The full component reference: [Optimization Workflows](../workflows/workflows.md).
- Writing the evaluation callback:
  [Writing Evaluation Callbacks](../workflows/evaluation_callbacks.md).
- Running evaluations in parallel: [Parallel Evaluation](../workflows/parallel.md).

::: ropt.plugins.operation.default
    options:
        members: False
::: ropt.plugins.event_handler.default
    options:
        members: False
::: ropt.plugins.event_handler.default
    options:
        members: False
::: ropt.plugins.operation.default.DefaultOperationPlugin
    options:
        members: False
::: ropt.plugins.event_handler.default.DefaultEventHandlerPlugin
    options:
        members: False
::: ropt.plugins.evaluator.default.DefaultEvaluatorPlugin
    options:
        members: False

::: ropt.plugins.operation.ensemble_evaluator.DefaultEnsembleEvaluatorOperation
    options:
        members: [run]
::: ropt.plugins.operation.optimizer.DefaultOptimizerOperation
    options:
        members: [run]

::: ropt.plugins.event_handler._tracker.DefaultTrackerHandler
    options:
        members: False
::: ropt.plugins.event_handler._store.DefaultStoreHandler
    options:
        members: False
::: ropt.plugins.event_handler._observer.DefaultObserverHandler
    options:
        members: False

::: ropt.plugins.evaluator._function_evaluator.DefaultFunctionEvaluator
    options:
        members: False

::: ropt.plugins.evaluator.cached_evaluator.DefaultCachedEvaluator
    options:
        members: False

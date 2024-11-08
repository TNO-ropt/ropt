::: ropt.plugins.plan.default.DefaultPlanPlugin
    options:
        members: none

::: ropt.plugins.plan._set.DefaultSetStep
    options:
        members: [DefaultSetStepWith]

::: ropt.plugins.plan._optimizer.DefaultOptimizerStep
    options:
        members: [DefaultOptimizerStepWith, NestedPlanConfig]

::: ropt.plugins.plan._evaluator.DefaultEvaluatorStep
    options:
        members: [DefaultEvaluatorStepWith]

::: ropt.plugins.plan._pickle.DefaultPickleStep
    options:
        members: [DefaultPickleStepWith]

::: ropt.plugins.plan._print.DefaultPrintStep
    options:
        members: [DefaultPrintStepWith]

::: ropt.plugins.plan._repeat.DefaultRepeatStep
    options:
        members: [DefaultRepeatStepWith]

::: ropt.plugins.plan._metadata.DefaultMetadataHandler
    options:
        members: [DefaultMetadataHandlerWith]

::: ropt.plugins.plan._save.DefaultSaveHandler
    options:
        members: [DefaultSaveHandlerWith]

::: ropt.plugins.plan._table.DefaultTableHandler
    options:
        members: [DefaultTableHandlerWith]

::: ropt.plugins.plan._tracker.DefaultTrackerHandler
    options:
        members: [DefaultTrackerHandlerWith]

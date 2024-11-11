::: ropt.plugins.plan.default.DefaultPlanPlugin
    options:
        members: none

::: ropt.plugins.plan._set.DefaultSetStep
    options:
        members: [DefaultSetStepWith]

::: ropt.plugins.plan._load_vars.DefaultLoadStep
    options:
        members: [DefaultLoadStepWith]

::: ropt.plugins.plan._save_vars.DefaultSaveStep
    options:
        members: [DefaultSaveStepWith]

::: ropt.plugins.plan._evaluator.DefaultEvaluatorStep
    options:
        members: [DefaultEvaluatorStepWith]

::: ropt.plugins.plan._optimizer.DefaultOptimizerStep
    options:
        members: [DefaultOptimizerStepWith, NestedPlanConfig]

::: ropt.plugins.plan._print.DefaultPrintStep
    options:
        members: [DefaultPrintStepWith]

::: ropt.plugins.plan._repeat.DefaultRepeatStep
    options:
        members: [DefaultRepeatStepWith]

::: ropt.plugins.plan._metadata.DefaultMetadataHandler
    options:
        members: [DefaultMetadataHandlerWith]

::: ropt.plugins.plan._tracker.DefaultTrackerHandler
    options:
        members: [DefaultTrackerHandlerWith]

::: ropt.plugins.plan._table.DefaultTableHandler
    options:
        members: [DefaultTableHandlerWith]

::: ropt.plugins.plan._save_results.DefaultSaveHandler
    options:
        members: [DefaultSaveHandlerWith]

::: ropt.plugins.plan.default.ExpressionFunctions

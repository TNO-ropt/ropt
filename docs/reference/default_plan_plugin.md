::: ropt.plugins.plan.default.DefaultPlanPlugin
    options:
        members: False

::: ropt.plugins.plan._set.DefaultSetStep
    options:
        members: [DefaultSetStepWith]

::: ropt.plugins.plan._load_data.DefaultLoadStep
    options:
        members: [DefaultLoadStepWith]

::: ropt.plugins.plan._save.DefaultSaveStep
    options:
        members: [DefaultSaveStepWith]

::: ropt.plugins.plan.evaluator.DefaultEvaluatorStep
    options:
        members: [DefaultEvaluatorStepWith]

::: ropt.plugins.plan.optimizer.DefaultOptimizerStep
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

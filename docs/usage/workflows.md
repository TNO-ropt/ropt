# Optimization Workflows

The section [Running a basic optimization task](basic.md) explains how to run a
single optimization using the [`BasicOptimizer`][ropt.workflow.BasicOptimizer]
class. Although this is sufficient for simple optimization tasks, this class may be 
limited for more complex workflows.

Workflows provide a powerful and flexible framework for constructing and
executing optimization workflows. It is designed to handle both simple,
single-run optimizations and more complex, customized scenarios. The framework
is built upon three key components:

- **[`ComputeStep`][ropt.workflow.compute_steps.ComputeStep]**: Defines a
  distinct action within the workflow, such as running an optimization algorithm.
- **[`EventHandler`][ropt.workflow.event_handlers.EventHandler]**: Responds to
  events emitted by `ComputeStep` instances. This allows for real-time
  monitoring, storing results, or triggering custom logic during the workflow.
- **[`Evaluator`][ropt.workflow.evaluators.Evaluator]**: Provides a mechanism
  for `ComputeStep` objects to perform function evaluations, such as running
  simulations on a high-performance computing (HPC) cluster.

Compute steps, event handlers are executed by calling their
[`run`][ropt.workflow.compute_steps.ComputeStep.run] method. During execution,
compute steps may emit [`events`][ropt.events.Event] to communicate intermediate
results. Event handlers can be added to compute steps using the
[`add_event_handler`][ropt.workflow.compute_steps.ComputeStep.add_event_handler]
method. Many compute steps, such as those performing optimizations, will require
the repeated evaluation of a function, which is performed by evaluator objects
passed to the step upon creation.

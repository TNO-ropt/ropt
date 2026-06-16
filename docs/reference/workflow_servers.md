# Servers

Servers dispatch [`Task`][ropt.workflow.servers.Task] objects produced by an
[`AsyncEvaluator`][ropt.workflow.evaluators.AsyncEvaluator] to a concrete
execution backend (asyncio, multiprocessing, or HPC).

See [Parallel Evaluation](../usage/parallel.md) for usage.

::: ropt.workflow.servers.Server
::: ropt.workflow.servers.Task
::: ropt.workflow.servers.ThreadingServer
::: ropt.workflow.servers.MultiprocessingServer
::: ropt.workflow.servers.HPCServer


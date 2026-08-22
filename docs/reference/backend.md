# Optimizer Backends

A backend is the bridge to an external optimization library. `ropt` ships
with a SciPy-based backend and an `external` backend for running optimizers
in a separate Python process; additional backends are provided by plugin
packages (see [Installation](../getting_started/installation.md)).

See [Running the optimizer in a separate
process](../optimizer_setup/configuration.md#external-backend) for when and how
to use the `external` backend.

::: ropt.backend
::: ropt.backend.scipy.SciPyBackend
::: ropt.backend.external.ExternalBackend
::: ropt.backend.utils


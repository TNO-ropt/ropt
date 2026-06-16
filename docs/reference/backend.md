# Optimizer Backends

A backend is the bridge to an external optimization library. `ropt` ships
with a SciPy-based backend and an `external` backend for running optimizers
in a separate Python process; additional backends are provided by plugin
packages (see [Installation](../usage/installation.md)).

::: ropt.backend
::: ropt.backend.scipy.SciPyBackend
::: ropt.backend.external.ExternalBackend
::: ropt.backend.utils


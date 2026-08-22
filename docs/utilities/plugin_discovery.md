# Plugin Discovery and Validation

`ropt` provides helper functions for querying installed plugins at runtime.
These are useful for verifying your environment or building dynamic
configurations.

[`find_backend_plugin`][ropt.workflow.find_backend_plugin] and
[`find_sampler_plugin`][ropt.workflow.find_sampler_plugin] look up which plugin
provides a given method. They accept the same `"plugin/method"` or `"method"`
strings used in the configuration (see
[Method strings](../optimizer_setup/configuration.md#method-strings)) and return
the plugin name, or `None` if no plugin supports the method:

```python
from ropt.workflow import find_backend_plugin, find_sampler_plugin

find_backend_plugin("slsqp")           # "scipy"
find_backend_plugin("scipy/L-BFGS-B")  # "scipy"
find_backend_plugin("unknown")         # None

find_sampler_plugin("scipy/default")   # "scipy"
```

[`validate_backend_options`][ropt.workflow.validate_backend_options] checks
whether a set of backend-specific options is valid for a given method, raising
an error if not. Call it before starting a long optimization run to catch
configuration mistakes early:

```python
from ropt.workflow import validate_backend_options

validate_backend_options("scipy/slsqp", {"maxiter": 200})   # ok
validate_backend_options("scipy/slsqp", {"bogus": 1})       # raises: unknown option
```

## Where to next

- Installing plugin packages: [Installation](../getting_started/installation.md#plugin-packages).
- The `"plugin/method"` naming convention in full:
  [Method strings](../optimizer_setup/configuration.md#method-strings).
- Tracing what `ropt` is doing at runtime: [Logging](logging.md).

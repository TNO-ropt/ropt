# Plugin Discovery and Validation

`ropt` provides helper functions for querying installed plugins at runtime.
These are useful for verifying your environment or building dynamic
configurations.

[`get_plugin_name`][ropt.plugins.manager.get_plugin_name] looks up which plugin
provides a given method. It takes the plugin area and a method string — the same
`"plugin/method"` or `"method"` strings used in the configuration (see
[Method strings](../optimizer_setup/configuration.md#method-strings)) — and
returns the plugin name, or `None` if no plugin supports the method:

```python
from ropt.utils import get_plugin_name

get_plugin_name("backend", "slsqp")           # "scipy"
get_plugin_name("backend", "scipy/L-BFGS-B")  # "scipy"
get_plugin_name("backend", "unknown")         # None

get_plugin_name("sampler", "scipy/default")   # "scipy"
```

The area is one of `"backend"`, `"sampler"`, `"realization_filter"`,
`"function_estimator"` or `"variable_transform"`.

[`validate_backend_options`][ropt.utils.validate_backend_options] checks
whether a set of backend-specific options is valid for a given method, raising
an error if not. Call it before starting a long optimization run to catch
configuration mistakes early:

```python
from ropt.utils import validate_backend_options

validate_backend_options("scipy/slsqp", {"maxiter": 200})   # ok
validate_backend_options("scipy/slsqp", {"bogus": 1})       # raises: unknown option
```

## Where to next

- Implementing a plugin of your own: [Writing a Plugin](writing_plugins.md).
- Installing plugin packages: [Installation](../getting_started/installation.md#plugin-packages).
- The `"plugin/method"` naming convention in full:
  [Method strings](../optimizer_setup/configuration.md#method-strings).
- Tracing what `ropt` is doing at runtime: [Logging](logging.md).

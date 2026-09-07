# Writing a Plugin

A plugin makes your own optimizer, sampler, filter, or estimator selectable
from a configuration by name, in the same way the built-in ones are:

```python
"backend": {"method": "my_package/my_method"}
```

A plugin is the implementation class itself. It declares which method names it
provides, and `ropt` finds it through a Python entry point, so nothing has to be
imported or registered by hand.

!!! note

    You do not need a plugin to use your own component. Every component field
    also accepts an already-constructed object, so a `Sampler` subclass can be
    passed straight into the configuration (see [Providing optimizer
    components](../optimizer_setup/configuration.md#providing-optimizer-components)).
    Write a plugin when the component should be selectable by name — typically
    when you ship it in a package for others to configure.

## The plugin areas

There is one plugin area per component type, each with its own entry-point group
and its own base class:

| Entry-point group | Base class |
| ----------------- | ---------- |
| `ropt.plugins.backend` | [`Backend`][ropt.backend.Backend] |
| `ropt.plugins.sampler` | [`Sampler`][ropt.sampler.Sampler] |
| `ropt.plugins.realization_filter` | [`RealizationFilter`][ropt.realization_filter.RealizationFilter] |
| `ropt.plugins.function_estimator` | [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] |

## What a plugin must declare

Subclass the base class of the area and add one class attribute:

- **`methods`** — the method names the class provides, as a set. The names are
  matched case-insensitively, so they may be written in any case. Include
  `"default"` if the class has a sensible standard choice, so that
  `"my_package/default"` selects it; leave it out if it does not, and the name
  will correctly fail to resolve.

Two things are optional:

- **`discoverable`** — set it to `False` to keep the plugin from being matched
  when a configuration gives a bare method name without a plugin prefix. The
  built-in [`external`][ropt.backend.external.ExternalBackend] backend does
  this, because its method names belong to the backend it delegates to. The
  default is `True`.
- **`methods` as a predicate** — a class that cannot enumerate what it supports
  may instead set `methods` to a function taking a method name and returning a
  `bool`. Use this only when the names are not knowable in advance, for example
  when they are resolved against another installed package. A predicate receives
  the method name exactly as written, so it owns its own casing, and it cannot
  be listed. See [`MethodSpec`][ropt.plugins.MethodSpec].

The object is constructed by calling the class with the validated configuration
object for its area, for example a
[`SamplerConfig`][ropt.config.SamplerConfig] for a sampler, carrying the
`method` string and the `options` given in the configuration.

## An example

A sampler that draws uniform perturbations. First the sampler itself, a
[`Sampler`][ropt.sampler.Sampler] subclass:

```python
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ropt.config import SamplerConfig
from ropt.context import EnOptContext
from ropt.sampler import Sampler


class UniformSampler(Sampler):
    def __init__(self, sampler_config: SamplerConfig) -> None:
        self._config = sampler_config

    def init(
        self, context: EnOptContext, mask: NDArray[np.bool_] | None, rng: Generator
    ) -> None:
        self._rng = rng

    def generate_samples(self) -> NDArray[np.float64]:
        ...
```

The only thing needed to make it selectable is the `methods` attribute, so it
goes on the class itself:

```python
from typing import ClassVar

from ropt.plugins import MethodSpec


class UniformSampler(Sampler):
    methods: ClassVar[MethodSpec] = {"default", "uniform"}

    ...
```

Annotating it as a `ClassVar` is what keeps type checkers and linters happy
about a set assigned at class level; a bare `methods = {...}` works just as
well at runtime.

## Registering it

Declare the class under the entry-point group of its area, in the
`pyproject.toml` of the package that contains it:

```toml
[project.entry-points."ropt.plugins.sampler"]
my_package = "my_package.sampler:UniformSampler"
```

The entry-point name is the plugin name used in method strings. After
installing the package, the sampler is available as `"my_package/uniform"`, or
as `"uniform"` if no other installed plugin claims that name:

```python
"samplers": [{"method": "my_package/uniform"}],
```

Check that the plugin is found with
[`get_plugin_name`][ropt.plugins.manager.get_plugin_name]:

```python
from ropt.utils import get_plugin_name

get_plugin_name("sampler", "my_package/uniform")   # "my_package"
```

## Registering one without an entry point

An entry point needs an installed package, which a class written in a script or
a notebook does not have. Such a class is added by hand with
[`register_plugin`][ropt.plugins.manager.register_plugin]:

```python
from ropt.plugins.manager import register_plugin

register_plugin("sampler", "my_package", UniformSampler)
```

It is then found by exactly the same lookups as an installed one, so
`"my_package/uniform"` works from that point on, for every optimization started
afterwards. The class must declare its `methods` just as an installed plugin
does.

Registering under the name of an installed plugin is an error, since installed
plugins cannot be shadowed. Registering a name that was registered before
replaces it, which is what re-running a cell that defines and registers a class
needs to do.

Plugins are otherwise discovered once: the shared plugin manager scans the entry
points the first time a method is looked up, so a package installed while the
program is running is not picked up.

## Validating options

Backends receive their method-specific settings through the `options` field of
[`BackendConfig`][ropt.config.BackendConfig], which `ropt` passes through
unchecked. A backend reports what it accepts by implementing
`validate_options`, which is what
[`validate_backend_options`][ropt.utils.validate_backend_options] calls so
that users can catch mistakes before starting a long run.

[`OptionsSchemaModel`][ropt.config.options.OptionsSchemaModel] describes the
options of each method in one place, and both validates them and generates the
documentation table for them. The built-in SciPy backend uses it; see
`SCIPY_OPTIONS_SCHEMA` in `ropt.backend.scipy` for a complete example.

## What a backend may not change

A backend runs in the caller's process, alongside whatever else is in it —
including other optimizations, since
[`optimize_many`][ropt.simple.optimize_many] runs several at once, each on its
own thread with its own configuration. Anything a backend changes *per process*
is therefore shared with runs it knows nothing about, and cannot carry per-run
settings.

So while a run is in progress a backend must not change the working directory,
the environment, `sys.stdout` or `sys.stderr`, or file descriptors 1 and 2.
Where the optimizer produces a log, ask the library to write it to a named
file — that is the per-run answer — or to be quiet.

Not every library allows this, and `ropt` does not paper over the ones that do
not. What such a backend must do instead is **say so in its own
documentation**:

- **It needs exclusive process state** — a working directory, a fixed file name,
  or state kept in the library between calls. What that rules out is not "one
  at a time" but "anything at all at the same time": a changed working directory
  breaks another run in the process whatever backend it uses, and the user's
  evaluation function with it. Say that the backend cannot run concurrently
  in-process, and point users at the
  [`external`][ropt.backend.external.ExternalBackend] backend, which gives it a
  process of its own.

Output is the exception to the rule: `ropt` captures it for you, on both levels,
when the user configures
[`stdout` or `stderr`](../optimizer_setup/configuration.md#optimizer). A backend
should not capture its own output, but it does have two things to do.

## Reporting progress { #reporting-progress }

Whether the optimizer reports at all is the user's choice, made through
[`verbose`](../optimizer_setup/configuration.md#backend). Map it onto whatever
your library provides using
[`resolve_verbosity`][ropt.backend.utils.resolve_verbosity], which normalises
the setting so you need not tell `True` from `1` yourself:

```python
level = resolve_verbosity(verbose=self._config.verbose)
```

`None` means "your library's own default level", `0` means silent, and a
positive integer is an explicit level to be clamped to what you support. Note
that `None` calls for opposite actions in different libraries: SciPy prints
nothing unless asked, so it must be switched on, while NOMAD reports by default,
so it is left alone.

Set the corresponding option only if the user has not set it themselves —
`setdefault` for a dictionary, or a check before appending for a list — so that
a specific option such as `DISPLAY_DEGREE 3` refines the generic flag rather
than fighting it.

## Declaring native output { #declaring-native-output }

`ropt` captures the optimizer's output by swapping `sys.stdout` and
`sys.stderr` for the duration of the run. That catches anything printed from
Python — which is most optimizers, including everything in SciPy except `tnc`.
A compiled optimizer, though, commonly writes to file descriptors 1 and 2
directly, and never touches `sys.stdout` at all.

Set
[`bypasses_python_output`][ropt.backend.Backend.bypasses_python_output] to
`True` when the wrapped library does that, and `ropt` redirects the descriptors
as well:

```python
class MyBackend(Backend):
    @property
    def bypasses_python_output(self) -> bool:
        return True
```

The answer may differ per method, in which case return it based on the
configured method — the built-in SciPy backend returns `self._method == "tnc"`.
**When unsure, declare it for the whole backend.** Being wrong that way costs
only a brief rewiring of process-global state; being wrong the other way means
the user's output escapes to the terminal, and nothing announces it.

You do not have to guess.
[`collect_native_output`][ropt.backend.utils.collect_native_output] runs an
optimization with the Python streams diverted and the descriptors captured, so
whatever it returns was written below Python:

```python
from ropt.backend.utils import collect_native_output

def test_backend_declares_native_output_correctly():
    native = collect_native_output(lambda: optimize(config, x0, objective))
    assert native == ""      # ... or `!= ""` if the declaration is True
```

Worth having as a test rather than a one-off check: these properties drift.
SciPy's `disp` silently became a no-op for `l-bfgs-b` and `nelder-mead` in 1.18.

## Where to next

- The registry in full: [Plugin Manager](../reference/plugin_manager.md).
- Looking up and validating installed plugins:
  [Plugin Discovery](plugin_discovery.md).
- How method strings are resolved:
  [Method strings](../optimizer_setup/configuration.md#method-strings).

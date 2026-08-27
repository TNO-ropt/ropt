"""Serialization for work that leaves this process.

Anything sent to a separate process has to be serialized, and the standard
library can only send functions and classes it can look up by name.
`cloudpickle` lifts that limit, so it is used when it is installed and the
standard library is used when it is not. It is never required: everything that
leaves this process works without it, as long as what is sent can be named.

Everything that sends work out of the process serializes here, so that one rule
covers all of them: the executors, which send work items to their workers, and
the external backend, which sends a problem to the process that solves it.

Only writing has to choose. Reading does not: `cloudpickle` reuses the standard
library's unpickler unchanged, and a payload it wrote names `cloudpickle` itself
for the parts only it understands, so the import happens by itself when it is
needed at all.
"""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from importlib.util import find_spec
from typing import IO, TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable

HAVE_CLOUDPICKLE: Final = find_spec("cloudpickle") is not None

# Annotated, because `cloudpickle` ships no types and would otherwise make
# everything written through this module untyped.
dump: Callable[[Any, IO[bytes]], None]
dumps: Callable[[Any], bytes]

if HAVE_CLOUDPICKLE:
    import cloudpickle

    dump = cloudpickle.dump
    dumps = cloudpickle.dumps
else:
    dump = pickle.dump
    dumps = pickle.dumps

load = pickle.load  # ruff: ignore[suspicious-pickle-usage]
loads = pickle.loads  # ruff: ignore[suspicious-pickle-usage]

# What to say when something could not be written. For a worker in this same
# environment, a missing `cloudpickle` here is missing there too, so installing
# it is a real answer rather than a guess. A job on a compute node need not run
# the same environment at all, which is why this only ever advises.
CANNOT_SERIALIZE: Final = (
    "install ropt[cloudpickle] to send closures, lambdas and locally defined "
    "functions and classes, or use module-level ones"
    if not HAVE_CLOUDPICKLE
    else "the object itself cannot be serialized"
)

# What to say when something arrived but could not be rebuilt. This is a
# different failure with a different cause: the write succeeded, because the
# names it stored all resolve *here*. Reading is where a name that does not
# resolve on the far side shows up, so the advice is about the far side, and
# `cloudpickle` is only the most likely of several answers.
CANNOT_DESERIALIZE: Final = (
    "it names something the worker cannot look up — code defined in a notebook "
    "or an interactive session, which ropt[cloudpickle] would send by value "
    "instead of by name, or a package the worker's environment does not have"
    if not HAVE_CLOUDPICKLE
    else "it names something the worker cannot look up — code the worker cannot "
    "import, or a package its environment does not have"
)

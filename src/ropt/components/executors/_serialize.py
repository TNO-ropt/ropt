"""Serialization for work that leaves this process.

Work sent to a separate process has to be serialized, and the standard library
can only send functions it can look up by name. `cloudpickle` lifts that limit,
so it is used when it is installed and the standard library is used when it is
not.

Only writing has to choose. Reading does not: `cloudpickle` reuses the standard
library's unpickler unchanged, and a payload it wrote names `cloudpickle` itself
for the parts only it understands, so the import happens by itself when it is
needed at all.
"""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from importlib.util import find_spec
from typing import Final

HAVE_CLOUDPICKLE: Final = find_spec("cloudpickle") is not None

if HAVE_CLOUDPICKLE:
    import cloudpickle

    dump = cloudpickle.dump
    dumps = cloudpickle.dumps
else:
    dump = pickle.dump
    dumps = pickle.dumps

load = pickle.load  # ruff: ignore[suspicious-pickle-usage]
loads = pickle.loads  # ruff: ignore[suspicious-pickle-usage]

# What to say when something could not be written. Both ends of a job run in the
# same environment, so if `cloudpickle` is missing here it is missing there too,
# and installing it is a real answer rather than a guess.
CANNOT_SERIALIZE: Final = (
    "install ropt[cloudpickle] to send closures, lambdas and locally defined "
    "functions, or use module-level ones"
    if not HAVE_CLOUDPICKLE
    else "the object itself cannot be serialized"
)

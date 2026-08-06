"""The main `ropt` module, a library for ensemble based optimization.

The high-level convenience API lives in the `ropt.simple` module; import it
directly, for example ``from ropt.simple import optimize``.
"""
# ruff: file-ignore[non-empty-init-module]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

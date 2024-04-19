"""Plugin functionality for adding realization filters.

This package contains the protocol that must be followed by realization filter
plugins, and the default realization filters that are part of `ropt`.

Realization filters are used by the optimizer to determine how a set of
realizations should be used to calculate objective and constraint function
values. They do this by calculating the weights that should be used for each
realization when calculating the values of a given set of objectives and
constraints.
"""

"""Plugin functionality for adding optimization plugins.

Optimization plugins are managed by a
[`PluginManager`][ropt.plugins.PluginManager] object, which returns classes or
factory functions to create objects that implement one or more optimization
methods. These objects must adhere to the
[`Optimizer`][ropt.plugins.optimizer.base.Optimizer] abstract base class.
This abstract base class allows `ropt` to provide the optimizer with the
callback used for evaluating functions and gradients and allows it to be started
from an optimizer step in the optimization plan.

To support the implementation of the optimizer classes, the
[`ropt.plugins.optimizer.utils`][ropt.plugins.optimizer.utils] module provides
some utilities.

By default the [`SciPyOptimizer`][ropt.plugins.optimizer.scipy.SciPyOptimizer]
plugin is installed which provides a number of methods from the
[`scipy.optimize`](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
package.
"""

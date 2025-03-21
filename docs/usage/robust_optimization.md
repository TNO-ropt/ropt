# Introduction: Ensemble-based robust optimization
Constraint optimization is the process of optimizing an objective function
$f(\mathbf{x})$ with respect to a vector of variables $\mathbf{x}$ in the
presence of one or more inequality constraints $g_j(\mathbf{x})$ and/or equality
constraints $h_k(\mathbf{x})$.

$$ \begin{align} \textrm{minimize} \quad & f(\mathbf{x}) \\
\textrm{subject to} \quad & g_j(\mathbf{x}) \le 0, \quad j=1, \ldots, J \\ &
h_k(\mathbf{x}) = 0, \quad k=1, \ldots, K \\
& \mathbf{x}^L \le \mathbf{x} \le \mathbf{x}^U \end{align} $$

In this context, the function $f(\mathbf{x})$ is assumed to have a deterministic
nature, meaning it is well-defined for given parameters. However, in realistic
scenarios, $f(\mathbf{x})$ may be part of a larger set of functions, especially
if it depends on uncertain parameters drawn from some, possibly unknown,
probability distribution.

Ensemble-based robust optimization aims to optimize an ensemble of functions
$f_i(\mathbf{x})$ with respect to $\mathbf{x}$. The set of *realizations* $f_i$
captures the uncertainty that may exist in the model, which can be, for
instance, constructed by varying some parameters according to a given
probability distribution. When given a set of realizations, ensemble-based
optimization proceeds by combining the functions $f_i(\mathbf{x})$ into a single
objective function. For example, using a weighted sum, the problem becomes
(ignoring constraints):

$$ \textrm{minimize} \quad \sum_i w_i f_i(\mathbf{x}), $$

where $w_i$ represents the weights assigned to the different realizations. In
more complex settings, the realizations may also be combined in different ways,
and the set of realizations may be modified during optimization. For instance,
risk-aware objectives may be constructed by minimizing the standard deviation of
the functions or by selecting some of the worst-performing realizations at each
iteration.

In practice, the optimization task often becomes complex due to additional
factors. The evaluation of functions might be computationally expensive, and
calculating their gradients analytically can be challenging or even impossible.
For example, the functions may involve lengthy simulations of a physical process
with numerous variables, utilizing numerical calculations that preclude
straightforward analytical differentiation.

`ropt` leverages standard optimization algorithms, such as those available in
the [SciPy](https://www.scipy.org) package. These methods typically follow an
iterative approach, necessitating repeated assessments of the objective function
and, in many cases, its gradient. Currently, it is assumed that the functions
are not easily differentiated analytically. One of the core functions of `ropt`
is to calculate gradients efficiently using stochastic methods.

`ropt` is responsible for configuring and executing the optimization algorithm,
building the overall function and gradient values from individual realizations,
and monitoring both intermediate and final optimization results. It delegates
the actual calculations of functions to external code that is provided by the
user.

While many optimization scenarios involve a single run of a particular
method, there are cases where it proves beneficial to conduct multiple runs
using the same or different algorithms. For example, when dealing with a mix of
continuous and discrete variables, it might be advantageous to employ different
methods for each variable type. `ropt` facilitates this by offering a
mechanism to run a workflow containing multiple optimizers, potentially of
different types, in an alternating or nested fashion.

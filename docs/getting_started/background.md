# Ensemble-based robust optimization
Constrained optimization is the process of optimizing an objective function
$f(\mathbf{x})$ with respect to a vector of variables $\mathbf{x}$ subject to
bounds ($\mathbf{x}^L$, $\mathbf{x}^U$), and/or inequality/equality constraints
($g_j(\mathbf{x})$, $h_k(\mathbf{x})$).

$$ \begin{align*} \textrm{minimize} \quad & f(\mathbf{x}) \\
\textrm{subject to} \quad & g_j(\mathbf{x}) \le 0, \quad \quad j=1, \ldots, J \\ &
h_k(\mathbf{x}) = 0, \quad \quad k=1, \ldots, K \\
& \mathbf{x}^L \le \mathbf{x} \le \mathbf{x}^U \end{align*} $$

Here, $f(\mathbf{x})$ is deterministic: for a given $\mathbf{x}$ it always
returns the same value. In practice, however, $f(\mathbf{x})$ often depends on
uncertain parameters drawn from some — possibly unknown — probability
distribution. In that case, a single evaluation of $f(\mathbf{x})$ is really
just one member of a larger set of possible functions.

Ensemble-based robust optimization optimizes such a set, or *ensemble*, of
functions $f_i(\mathbf{x})$ at once. Each $f_i$ is called a *realization*: one
possible version of the function, for instance obtained by drawing a set of
values for the uncertain parameters from their probability distribution.
Given a set of realizations, ensemble-based optimization combines the
functions $f_i(\mathbf{x})$ into a single objective function. Using a weighted
sum, for example, the problem becomes (ignoring constraints):

$$ \textrm{minimize} \quad \sum_i w_i f_i(\mathbf{x}), $$

where $w_i$ is the weight given to realization $i$. The realizations can also
be combined in other ways, and the set of realizations used can even change
during the optimization. For example, a risk-aware objective may add a term
based on the standard deviation of the $f_i$, or use only the
worst-performing realizations at each iteration.

In practice, evaluating $f_i(\mathbf{x})$ is often expensive, and calculating
its gradient analytically can be difficult or impossible — for instance when
$f_i(\mathbf{x})$ comes from a long numerical simulation of a physical
process.

`ropt` builds on standard optimization algorithms, such as those in the
[SciPy](https://www.scipy.org) package. These algorithms work iteratively,
evaluating the objective function — and usually its gradient — many times
over the course of the optimization. `ropt` assumes that gradients cannot be
calculated analytically, and one of its core features is estimating them
efficiently using stochastic methods.

`ropt` sets up and runs the optimization algorithm, combines the individual
realizations into overall function and gradient values, and keeps track of
intermediate and final results. Calculating the functions themselves — for
example, running a simulation — is left to code that you provide.

Most optimization problems only need a single run of one method. Sometimes,
though, it helps to combine several runs, possibly with different algorithms
— for example, when a problem mixes continuous and discrete variables, each
kind may be best handled by its own method. `ropt` supports this too: several
optimization runs can be combined sequentially, in parallel, or nested within
each other.


## Where to next

- Install and run your first optimization: [Installation](installation.md) and
  [Quickstart](quickstart.md).
- Learn the terms used across the documentation: [Key Concepts](../optimizer_setup/key_concepts.md).
- Learn the configuration format: [Configuration](../optimizer_setup/configuration.md).
- Understand the stochastic gradient (StoSAG) machinery in depth:
  [Stochastic Gradients](../optimizer_setup/gradients.md).
- Build custom optimization workflows beyond a single optimization run:
  [Optimization Workflows](../workflows/workflows.md).

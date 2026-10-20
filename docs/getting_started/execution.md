# Running in Parallel

An optimization calls your evaluation function many times. By default these
calls happen one after another. If each evaluation is slow, you can run several
at the same time using an evaluation **pool**. Pools must part of a
[`session`][ropt.session] context manager object that handles any cleanup
actions that must be taken after the optimization finishes.

Open a session using a `with` statement, build a pool with it, and pass the pool
to `optimize`. The pool employs four threads to evaluate the functions in
parallel:

```python
from ropt import optimize, session

with session() as s:
    pool = s.thread_pool(workers=4)
    result = optimize(config, x0, objective, pool=pool)
```

The session owns the pool and releases its workers when the block ends. The
runnable example script is
[examples/parallel.py](https://github.com/TNO-ropt/ropt/blob/main/examples/parallel.py).

A `thread_pool` is one kind of evaluation pool, others run each evaluation in a
separate process, or as a job on an HPC cluster. [Evaluating in
Parallel](../running/parallel.md) covers all three: which one suits a given
evaluation function, how many workers to ask for, what stopping an optimization
does to each, and the one property of your objective that rules some of them
out.

## See also

- To run several optimizations in parallel, see [Running Multiple
  Optimizations](../running/many_runs.md).
- Sessions are described in [Sessions](../running/sessions.md).

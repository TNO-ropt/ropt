# Running in Parallel

An optimization calls your evaluation function many times. By default these calls
happen one after another, on the same thread that called
[`optimize`][ropt.optimize]. If each call is slow, you can run several at
the same time by evaluating on a **pool**.

Open a [`session`][ropt.session], build a pool on it, and start the run
on that pool:

```python
from ropt import session

with session() as s:
    result = s.thread_pool(workers=4).optimize(config, x0, objective)
```

The session owns the pool and releases its workers when the block ends, so
there is nothing to close. The runnable script is
[examples/parallel.py](https://github.com/TNO-ropt/ropt/blob/main/examples/parallel.py),
which evaluates one optimization on a thread pool, or on a process pool
when it is passed `--multiprocessing`.

A thread pool is one of four kinds. The others run each evaluation in a separate
process, as a local job, or as a job on an HPC cluster.
[Evaluating in Parallel](../running/parallel.md) covers all four: which one
suits a given evaluation function, how many workers to ask for, what stopping a
run does to each, and the one property of your objective that rules some of them
out.

To run several optimizations at the same time, rather than parallelizing within
one, see [Many Runs at Once](../running/many_runs.md).

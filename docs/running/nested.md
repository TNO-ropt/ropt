# Nested Optimization

Some problems split naturally into two layers: a few variables that are awkward
for one method — integers, say, or switches — and the rest, which a gradient
method handles well. **Nested optimization** solves them in two loops. An outer
run varies the awkward variables, and every outer evaluation runs a complete
inner optimization over the remaining ones, returning the best value it reached.

!!! tip "This page also covers four things that are not nesting"
    Nesting itself is a niche, but this page puts four things that are not
    into one program small enough to read end to end: choosing a pool per
    layer, and why one of them has to stay on threads; collecting results from
    runs that overlap in time, in one handler; reusing an evaluation
    the optimizer has already made; and moving the expensive layer to a cluster
    by changing a single line.

Nothing in `ropt` is dedicated to this: the outer evaluation function
calls [`optimize`][ropt.simple.optimize] itself. What needs care is the plumbing
around it — which
variables each layer owns, which pool each layer evaluates on, and how to get
the results out.

```mermaid
flowchart TB
    subgraph proc["your process"]
        direction LR
        outer["outer optimize<br/>(integer variables)"]
        subgraph tpool["outer thread_pool — 2 threads"]
            direction TB
            in1["outer eval →<br/>inner optimize"]
            in2["outer eval →<br/>inner optimize"]
        end
        grp["shared<br/>DataFrameHandler"]
        outer --> in1
        outer --> in2
        in1 -.->|"results"| grp
        in2 -.->|"results"| grp
    end
    subgraph ppool["inner process_pool — 2 workers"]
        direction TB
        w1["worker"]
        w2["worker"]
    end
    in1 ==>|"batch"| w1
    in2 ==>|"batch"| w2
```

Solid arrows start a run, thick arrows carry evaluations out to a worker
process, and dotted arrows carry results back to the handler. Both pools are
drawn: the outer one is inside your process, the inner one is not. The sections
below work through each part.

The runnable script for this page is
[examples/simple/nested_optimization.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/nested_optimization.py).
It needs the `polars` extra.

## Splitting the variables

Both layers describe the *same* variable vector; each is handed the half it may
change. The `mask` field does this, and the two masks are complements:

```python
--8<-- "examples/simple/nested_optimization.py:configs"
```

A masked-out variable keeps the value it was given and is not passed to the
optimizer, so the inner run treats the outer variables as constants and the
outer run never touches the inner ones. The two layers can otherwise be
configured completely differently — here the outer is integer-valued and
gradient-free, while the inner is continuous and uses an ensemble of five
realizations. See [`variables`](../optimizer_setup/configuration_sections.md#variables)
for the field, and
[Discrete and Mixed-Integer Variables](../optimizer_setup/discrete.md) for the
outer method.

## The outer evaluation function

An outer evaluation *is* an inner optimization. The function receives the outer
variables, merges them into a full vector, runs `optimize`, and returns the best
objective it found:

```python
--8<-- "examples/simple/nested_optimization.py:inner"
```

`np.where(MASK, INITIAL_VALUES, variables)` builds the inner start point: the
inner variables begin where they always do, and the outer ones carry the values
being tried. The `metadata` tags every inner result with the outer evaluation
that caused it, which is what makes the collected results traceable. The `memo`
lookup above it is covered in
[Reusing an outer evaluation](#reusing-an-outer-evaluation).

## Two pools, not one

Each layer evaluates on its own pool, and that is a requirement rather than a
preference:

```python
--8<-- "examples/simple/nested_optimization.py:run"
```

The outer pool is a **thread** pool. Outer evaluations therefore stay inside
this process, where the inner pool and the shared handler are live objects; on
a process pool they would arrive as copies, which `ropt` refuses. The inner
pool is a **process** pool, which is where the real work goes.

That is also where a cluster belongs. Swapping the inner pool for an
[`hpc_pool`][ropt.simple.Session.hpc_pool] is the whole change — the inner
evaluations become cluster jobs, and the layer above is untouched:

```python
inner_pool = active.hpc_pool(workers=50)
```

Sending whole *optimizations* to the cluster, by putting the **outer** pool on
processes or HPC, is possible but gives something different. Each outer
evaluation is then copied into a job, and neither the inner pool nor the shared
group can travel with it: the job has to open its own session and collect its
own results, and return them as data for you to combine. Use that when the
outer evaluations are genuinely independent; use the shape above when you want
one pool and one table across all of them.

The inner run passes `bundle_size=0`, which sends a whole inner batch to one
worker at once. Within one inner run that gives up parallelism
completely — its five realizations are evaluated one after another — but the
parallelism here comes from the layer above: two outer evaluations run at once,
each feeding the inner pool one bundle, so both workers are busy and each batch
costs one transfer instead of five. See
[How a batch is split across workers](parallel.md#how-many-workers).

Handing the inner run the pool it is already running on would deadlock, and
`ropt` refuses it rather than hanging — see
[Pools inside an evaluation](parallel.md#pools-inside-an-evaluation) for why.

## Reusing an outer evaluation

The outer variables are integers and the outer method is population-based, so
the optimizer proposes points it has already evaluated. Each repeat would run a
complete inner optimization for a value that is already known. The two lines at
the top of the evaluation function return the stored value instead. The memo is
an ordinary dictionary, created beside the pools and passed to the function
through `partial`.

The key is exact. Integer variables arrive as whole-numbered doubles and are
re-proposed unchanged, so a repeated vector is identical bit for bit. The
realization index is part of the key because a memo keyed on the variables alone
gives the wrong value for a problem with more than one realization.

The dictionary is reachable only because the outer evaluations run on threads,
in this process. On a `process_pool` or an `hpc_pool` the function is copied
into a worker, which starts from an empty dictionary and discards it when it
finishes, without raising. See
[the rule to settle first](../getting_started/execution.md#the-rule-to-settle-first).

Two outer evaluations run at the same time, so both can find the same key
missing and both compute it. The stored value is the same either way, so the
cost is one repeated evaluation. A lock would close that window and serialize
the outer layer with it.

A repeat starts no inner run, so the table described next holds rows for the
outer evaluations that were computed, not for every outer evaluation.

## Collecting results from runs that overlap

The inner runs are concurrent, and [one
handler](handlers.md#sharing-a-handler-across-concurrent-runs) collects them
all: it takes a lock around every call, so the runs wait for each other rather
than interleaving. Give every inner run the same
[`DataFrameHandler`](handlers.md#dataframehandler) and every inner evaluation
from every inner run lands in one table, keyed by the outer evaluation it
belongs to.

Reading the answer takes one more step than usual:

```python
--8<-- "examples/simple/nested_optimization.py:best"
```

The outer result is not the place to look. The outer layer only ever sees its
own variables and holds the inner ones at their initial values, so the optimum
over *all* variables exists only in the collected inner results.

## See also

- The same flow built from workflow components:
  [Parallel Evaluation](../advanced/parallel.md).

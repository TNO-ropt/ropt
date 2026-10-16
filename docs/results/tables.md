# Writing Tables to Disk

A [`DataFrameHandler`][ropt.DataFrameHandler] collects results into tables
while an optimization runs, and its
[`write_tables`][ropt.DataFrameHandler.write_tables] method writes those tables
to files. The example on this page registers the default tables: the optimal
results, the gradients, the evaluation of each realization, the evaluations at perturbed
points, and the constraint values. The tables are rewritten each time they
change.

The example script is
[`examples/tables.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/tables.py).
It needs `polars`, and writes the tables as fixed-width text files.

## The problem

The problem has three variables with bounds, three realizations, one linear
constraint and one nonlinear constraint, so that every table has content:

```python
--8<-- "examples/tables.py:config"
```

`optimizer.output_dir` is the directory that the tables are written to. It is
passed to the callback below. The `names` entry gives the variables, the
objective and the constraints labels, which appear in the column headers. See
[`names`](../optimizer_setup/configuration_sections.md#names).

## The tables

`set_default_tables()` registers five tables. Each is built from function
results or from gradient results:

| Table | Built from | One row per |
| --- | --- | --- |
| `functions` | function results | function evaluation |
| `gradients` | gradient results | gradient evaluation |
| `evaluations` | function results | realization of a function evaluation |
| `perturbations` | gradient results | perturbation of a realization |
| `constraints` | function results | function evaluation |

Their columns have these titles:

| Table | Column titles |
| --- | --- |
| `functions` | `Batch`, `Target-Objective`, `Objective`, `Constraint`, `Variable` |
| `gradients` | `Batch`, `Target-Gradient`, `Grad-objective`, `Grad-constraint` |
| `evaluations` | `Batch`, `Realization`, `Variable`, `Objective`, `Constraint` |
| `perturbations` | `Batch`, `Realization`, `Perturbation`, `Variable`, `Objective`, `Constraint` |
| `constraints` | `Batch`, then `Bound-`, `Linear-` and `Nonlinear-` each followed by `lower`, `upper` and `violation` |

A field that holds a vector, such as the variables, gives one column per
element, labeled from `names`. The `constraints` table holds, for each
constraint, the difference from its lower and upper bound, and the amount by
which it is violated. The field that each title is built from is described in
[Working with Results](results.md).

## Writing the files

The handler registers the default tables, and sets `write_tables` as its
callback:

```python
--8<-- "examples/tables.py:handler"
```

The handler calls the callback after each evaluation that added rows to a table,
with `optimizer.output_dir` as the argument. `write_tables` raises a
`ValueError` if `output_dir` is not set. Each call rewrites every file, so the
files hold the results so far while the optimization runs. The optimization
waits while the files are written.

In a text file, every column has a fixed width. The labels of each column are
printed on a second header line, under the title, and numbers are aligned on
the decimal point. Floating point numbers are written with 6 significant
digits; set another number with `digits=`, for example
`partial(handler.write_tables, digits=10)`.
`write_tables(directory, format="csv")` writes CSV files instead, at full
precision. In a CSV file, the title and the labels are joined with the
separator of the handler (`,` by default, set with `sep=`), giving column names
such as `Variable,x`.

## Running the optimization

The handler is passed to the optimization like any other handler:

```python
--8<-- "examples/tables.py:run"
```

The `functions.txt` file it writes starts like this:

```text
Batch  Target-Objective  Objective  Constraint  Variable  Variable  Variable
                          distance      spread         x         y         z
-----  ----------------  ---------  ----------  --------  --------  --------
    0         0.32       0.32         0         1         1         1
    2         0.0962831  0.0962831    0.627965  0.298846  0.876325  0.926811
```

The batch IDs skip the gradient evaluations, whose results go to the
`gradients` and `perturbations` tables. To read the tables in your own code
instead of from the files, use `handler.get_tables()` or `handler["functions"]`;
see [Result Handlers](handlers.md#dataframehandler).

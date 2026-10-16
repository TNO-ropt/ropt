# Running External Programs

An evaluation function often runs a simulator or a solver as a separate program.
Start it with [`run_program`][ropt.run_program], and the program stops when the
optimization is aborted, on any pool, together with every program it started.
A program started with `subprocess.run` or `os.system` keeps running after an
abort, and on a thread pool the evaluation waits for it.

The example script
[`examples/programs.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/programs.py)
runs a simulator from the objective function. The simulator is
[`examples/simulator.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simulator.py),
a stand-in for a real one: it reads the variables from a JSON input file,
prints its progress, and writes the objective to a JSON output file. When the
input sets `hang`, it sleeps for an hour instead.

## Running the simulator

The objective function writes the input file into a temporary directory, runs
the simulator there, and reads the output file:

```python
--8<-- "examples/programs.py:objective"
```

The `cwd` argument sets the working directory of the simulator, so the
relative file names refer to the temporary directory. Each evaluation uses a
directory of its own, so evaluations that run at the same time do not
overwrite each other's files. The output of the simulator, standard output and
standard error together, goes to `simulator.log` in that directory.

`run_program` blocks until the program exits, and returns its exit status. A
status of zero usually means success; a negative status `-N` means that the
program was ended by signal `N`, as in `subprocess`. The objective returns NaN
for a failed simulation, which marks the evaluation as failed; see
[Failures and Aborts](failures.md).

## Running the optimization

The optimization runs the four realizations on a thread pool, so up to four
simulators run at the same time:

```python
--8<-- "examples/programs.py:run"
```

## Aborting a hanging simulator

In the second run the simulator of realization 2 does not finish. A timer
calls [`Session.abort`][ropt.Session.abort] after two seconds:

```python
--8<-- "examples/programs.py:abort"
```

The abort terminates the hanging simulator, and the optimization raises
[`AbortedError`][ropt.exceptions.AbortedError] about two seconds after it
started. If the objective started the simulator with `subprocess.run`, the
abort would not reach the simulator: the session would not close until the
simulator exited, an hour later. The same applies to an abort by Ctrl-C, or by
an exception elsewhere in the session.

## Arguments

The command is a list of the program and its arguments, as in
`subprocess.run`. A single string raises a `TypeError`: there is no shell, so
a string cannot be split into arguments.

| Argument | Description |
| --- | --- |
| `cwd` | The working directory of the program, by default that of your program. |
| `env` | The environment of the program, by default that of your program. |
| `stdout` | Where standard output goes. |
| `stderr` | Where standard error goes. |

`stdout` and `stderr` take one of:

- `None`, the default: the output goes where the output of your program goes.
- A path, as a string or a `Path`: the file is created, or emptied if it
  exists, and the output is written to it. A relative path is relative to `cwd`.
- An open file, or a file descriptor.
- `subprocess.DEVNULL`: the output is discarded.
- `subprocess.STDOUT`, for `stderr` only: standard error goes where standard
  output goes.

`subprocess.PIPE` raises a `ValueError`: to process the output, write it to a
file and read the file. Standard input is empty.

## When the program is stopped

`run_program` never stops the program on its own account. It stops the program
only when its caller is stopped, in one of three ways:

- **The task that runs the caller is aborted**, for example by Ctrl-C, by
  [`Session.abort`][ropt.Session.abort], or by a failure elsewhere in the
  session. `run_program` then raises
  [`AbortedError`][ropt.exceptions.AbortedError]; see
  [Failures and Aborts](failures.md) for how that reaches the caller. Called
  when the task is already being aborted, `run_program` raises at once and does
  not start the program.
- **An exception interrupts the call**, for example `KeyboardInterrupt` in the
  main thread. The exception is re-raised.
- **The process that called `run_program` ends**, for example when a process
  pool kills a worker that is still busy five seconds after an abort, or when an
  HPC job is cancelled.

In each case the program and every program it started receive `SIGTERM`, and
those still running five seconds later receive `SIGKILL`.

`run_program` behaves the same wherever it is called: in an evaluation function,
an event handler, a report callback, a function given to
[`run`][ropt.Session.run], on every pool, or in your own code outside any
optimization. Only the ways in which the caller can be stopped differ.
[`Session.abort`][ropt.Session.abort] reaches only the tasks of its session.
It does not reach code that no task runs, such as the main thread of your
script inside a `with session()` block or an evaluation function that you call
directly. Nor does it reach an optimization started with the module-level
[`optimize`][ropt.optimize], which is not part of the session. In the main
thread these calls still end on Ctrl-C, through the `KeyboardInterrupt`. In an
HPC job no abort reaches the program, but cancelling the job ends the process
that called `run_program`.

!!! note "Programs that leave the process group"
    On POSIX the program runs in a process group of its own, and an abort
    signals that group. A program that it starts in a new session or process
    group of its own, as a daemon does, is not reached. Outside POSIX only the
    program itself is terminated, and programs it started may keep running.

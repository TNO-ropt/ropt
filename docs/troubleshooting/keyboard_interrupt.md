# Keyboard Interrupts

Some imports can change how the program handles Ctrl-C while it waits. The
`ropt[polars]` extra installs Polars; affected Polars versions change a
process-wide signal setting on import, and `ropt` loads Polars when it is
installed. Until the [Polars issue](https://github.com/pola-rs/polars/issues/21739)
is fixed in a release, installing this extra can trigger the problem when
`ropt` is imported. This is not a `ropt` pool limitation. Other libraries
imported by your program may cause the same problem even without Polars.
Check whether Ctrl-C interrupts your script while it waits for a thread or
process pool. If it does not, call
[`restore_keyboard_interrupt`][ropt.utils.restore_keyboard_interrupt] after
the imports:

```python
from ropt.utils import restore_keyboard_interrupt

restore_keyboard_interrupt()
```

## The symptom

You press Ctrl-C while the program is waiting and it does not respond until
the wait ends. A wait for evaluations may last indefinitely, so the
optimization can appear to ignore Ctrl-C.

It affects the *whole* program, not just the optimization:
anything else that waits stops responding to Ctrl-C too.

## What causes it

Ctrl-C raises `SIGINT`, which CPython turns into a `KeyboardInterrupt` — but
only once the interrupted thread is back in the interpreter. A thread parked in
`Queue.get`, `Event.wait`, `Thread.join`, `Future.result` or `Lock.acquire` is
not there while it waits, and whether the signal breaks that wait depends on a
process-global flag called `SA_RESTART`.

CPython leaves the flag clear, so waits are interruptible. Some third-party
extension modules install their own `SIGINT` handler with the flag **set** when
they are imported. Because the flag is process-global, the effect is not
confined to whoever set it: from that import onwards, Ctrl-C cannot break into
any wait anywhere in the program.

When Polars is installed, a direct `import polars` is not required to trigger
the issue: `ropt` loads it. Without Polars, check the other packages your
program imports: any package that sets the flag can cause the same symptom.

## Where to call it

If Ctrl-C does not interrupt a wait in your program, call the helper **after**
the imports, because a later import can set the flag again. If you import a
package lazily, inside a function, call it after that import instead. You can
also check the behavior without `polars` installed, or after a Polars upgrade;
the workaround is needed only if an imported package still changes the setting.

## Why `ropt` does not do this for you

`SA_RESTART` is process-global state that belongs to your program. A library
that quietly changes it does so for every other part of that program,
including parts that deliberately installed the handler in question — and it
cannot put the flag back afterwards without restoring the very problem it was
called to fix.

There is also no point at which a library could do it reliably. Any import that
happens later can set the flag again, and `ropt` does not control when your
program imports things.

So the decision is left where it can be made: in the application, by
someone who knows whether the trade matters.

## What it does, exactly

It calls `signal.siginterrupt(signal.SIGINT, True)`, which clears the flag and
does nothing else. In particular it does **not** replace the `SIGINT` handler,
so a package that chained its own keeps working; `signal.signal` would replace
it.

It is a one-way switch on purpose. There is no context-manager form, because
restoring the flag on exit would restore the hang, and a clear flag is CPython's
own default in any case.

Calling it more than once is harmless, and it may be called from any thread.

!!! note "Platforms"
    There is no `SA_RESTART` on Windows and therefore no problem to fix; the
    call is a no-op there.

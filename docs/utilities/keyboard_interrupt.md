# Keyboard Interrupts

**You probably do not need this page.** Ctrl-C works normally in most programs
that use `ropt`, and nothing in `ropt` calls the function described here. It is
an escape hatch for one specific failure that is easy to recognise and hard to
diagnose from scratch. Read on only if it has happened to you.

## The symptom

You press Ctrl-C and nothing happens. Not "it takes a while" — nothing at all,
however many times you press it. The run keeps going, and the only way out is
to kill the process from another terminal.

The give-away is that it affects the *whole* program, not just the optimization:
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

This is why the symptom looks so strange. Nothing in your code changed, no
error is reported, and you need not have imported the culprit yourself — a
package `ropt` imports on your behalf is enough. Importing `ropt.simple` alone
sets the flag (at the time of writing by way of polars, which `ropt` loads
whenever it is installed).

## The fix

One line, at the top of your script, **after** the imports:

```python
from ropt.utils import restore_keyboard_interrupt

restore_keyboard_interrupt()
```

After the imports because an import is what sets the flag: calling this first
and importing afterwards leaves you exactly where you started. If you import
lazily, inside a function, call it after that import instead.

## Why `ropt` does not do this for you

`SA_RESTART` is process-global state that belongs to your program. A library
that quietly changes it decides on behalf of every other part of that program,
including parts that deliberately installed the handler in question — and it
cannot put the flag back afterwards without restoring the very problem it was
called to fix.

There is also no point at which a library could do it reliably. Any import that
happens later can set the flag again, and `ropt` has no say over when your
program imports things.

So the decision is left where it can actually be made: in the application, by
someone who knows whether the trade matters.

## What it does, exactly

It calls `signal.siginterrupt(signal.SIGINT, True)`, which clears the flag and
does nothing else. In particular it does **not** replace the `SIGINT` handler,
so a package that chained its own keeps working — which is why this is the right
call and `signal.signal` is not.

It is a one-way switch on purpose. There is no context-manager form, because
restoring the flag on exit would restore the hang, and a clear flag is CPython's
own default in any case.

Calling it more than once is harmless, and it may be called from any thread.

!!! note "Platforms"
    There is no `SA_RESTART` on Windows and therefore no problem to fix; the
    call is a no-op there.

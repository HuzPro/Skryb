"""Helpers for safely releasing tkinter :class:`Variable` instances.

The settings window runs on a fresh daemon thread each time the user
opens it and creates a new ``ctk.CTk()`` root. When the window closes,
that Tk interpreter is destroyed — but the Python ``StringVar`` objects
bound to its widgets linger until Python's garbage collector reaches
them. At that point, ``Variable.__del__`` calls
``self._tk.getboolean(self._tk.call("info", "exists", self._name))``
against the dead interpreter and raises
``RuntimeError: main thread is not in main loop`` (printed as
``Exception ignored``).

``Variable.__del__`` returns early if ``self._tk is None``, so clearing
that attribute before the Variable is collected is a safe, targeted
workaround. Collect every ``StringVar`` / ``IntVar`` / etc. a tab owns
and feed them into :func:`release_tk_variables` from the tab's
``dispose()`` method.
"""

import tkinter


def release_tk_variables(*variables: tkinter.Variable) -> None:
    """Break each variable's reference to its Tk interpreter.

    Silently ignores anything that isn't a :class:`tkinter.Variable` or
    that has already been released — disposal paths shouldn't throw.
    """
    for var in variables:
        if var is None:
            continue
        try:
            var._tk = None
        except AttributeError:
            pass

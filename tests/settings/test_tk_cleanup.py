import tkinter

from voicetotext.settings.tk_cleanup import release_tk_variables


class _FakeVariable:
    """Stand-in that mirrors ``tkinter.Variable``'s ``_tk`` attribute."""

    def __init__(self):
        self._tk = object()


class TestReleaseTkVariables:
    def test_should_clear_tk_attribute_on_each_variable(self):
        v1 = _FakeVariable()
        v2 = _FakeVariable()
        v3 = _FakeVariable()

        release_tk_variables(v1, v2, v3)

        assert v1._tk is None
        assert v2._tk is None
        assert v3._tk is None

    def test_should_ignore_none_entries(self):
        v = _FakeVariable()

        release_tk_variables(None, v, None)

        assert v._tk is None

    def test_should_swallow_attribute_errors(self):
        class _NoTk:
            __slots__ = ()

        # Should not raise even though _NoTk has no ``_tk`` slot to assign.
        release_tk_variables(_NoTk())

    def test_should_be_safe_to_call_twice(self):
        v = _FakeVariable()
        release_tk_variables(v)
        release_tk_variables(v)

        assert v._tk is None

    def test_should_prevent_tkinter_variable_del_from_touching_tk(self):
        # Simulate what tkinter.Variable.__del__ does: it returns early
        # when ``self._tk is None``. Without release_tk_variables, it
        # would proceed to call self._tk.getboolean() and raise.
        root = tkinter.Tk()
        root.withdraw()
        var = tkinter.StringVar(master=root, value="hello")
        assert var._tk is not None

        release_tk_variables(var)

        assert var._tk is None
        root.destroy()
        # After destroy, __del__ on var should be a no-op because _tk is None.
        # We can't force __del__ here, but the attribute state proves
        # the early-return path will be taken.

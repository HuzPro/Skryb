"""``set_titlebar_dark`` is best-effort native glue (the real DWM repaint
is checked by the manual Windows smoke). Here we only pin the contract
that it never propagates an error, so a failure to theme the caption can
never crash the settings window opening."""

from voicetotext.settings.win32_titlebar import set_titlebar_dark


class _NoHandle:
    def winfo_id(self):
        raise RuntimeError("no native handle")


def test_should_swallow_errors_when_handle_unavailable():
    set_titlebar_dark(_NoHandle(), dark=True)


def test_should_swallow_errors_for_non_tk_object():
    set_titlebar_dark(object(), dark=False)

"""Print the caret context of whatever window you click into.

Counts down so you can put the caret, or a selection, inside a real editor
before the read happens, then prints what each reader saw. Nothing is typed
and no window is activated, so the app under test keeps the focus throughout.

Both readers are printed separately so the field log shows which one answered:
UI Automation reaches most frameworks, and the edit-control messages reach the
Win32 ``EDIT`` and ``RichEdit`` windows that publish no text pattern.

    venv\Scripts\python.exe scripts/probe_caret_context.py

Expect ``NO_CONTEXT`` from controls that expose no UI Automation text
pattern.
"""

import logging
import sys
import time

from skryb.output.win32_caret_context import (
    read_caret_context,
    read_ui_automation_context,
)
from skryb.output.win32_edit_control_context import (
    read_edit_control_context,
)


COUNTDOWN_SECONDS = 3


def main() -> int:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    print("Click into a text field and place the caret or a selection.")
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"{remaining}...", flush=True)
        time.sleep(1)
    # UI Automation hands back characters (U+FFFC for an embedded object,
    # say) that a cp1252 console cannot encode, so escape before printing.
    print("ui automation:", ascii(read_ui_automation_context()))
    print("edit control: ", ascii(read_edit_control_context()))
    print("chosen:       ", ascii(read_caret_context()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

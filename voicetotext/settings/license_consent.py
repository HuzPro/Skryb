import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.core.color_scheme import detect_color_scheme
from voicetotext.settings.service import SettingsService
from voicetotext.settings.theme import apply_theme, theme_for
from voicetotext.settings.win32_titlebar import set_titlebar_dark
from voicetotext.transcription.license_policy import LicenseTerms


# Shows *terms* and reports whether the user accepted. The first argument
# is the widget the modal parents itself to, so a page and the window can
# share one prompt without either owning a window reference.
LicenseConsentPrompt = Callable[[tkinter.Misc, LicenseTerms], bool]

_DIALOG_GEOMETRY = "760x620"
_INTRO = (
    "This model is distributed under the licence below. Accepting records "
    "your agreement to these exact terms."
)


def collect_license_acceptance(
    service: SettingsService,
    model_name: str,
    parent: tkinter.Misc,
    prompt: LicenseConsentPrompt,
) -> bool:
    """Ask for consent when *model_name* is gated; say whether to proceed.

    ``True`` means the caller may go ahead: either the model is ungated,
    or the user accepted and the acceptance is already persisted.
    ``False`` means declined, and nothing has been written.
    """
    if not service.license_acceptance_required(model_name):
        return True
    if not prompt(parent, service.license_terms(model_name)):
        return False
    service.accept_license(model_name)
    return True


def ask_license_consent(parent: tkinter.Misc, terms: LicenseTerms) -> bool:
    """Show *terms* in a modal window and wait for the user's answer."""
    dialog = _LicenseConsentDialog(parent, terms)
    parent.wait_window(dialog)
    return dialog.accepted


class _LicenseConsentDialog(tkinter.Toplevel):
    """Modal window showing the exact text an acceptance is keyed to.

    The whole licence is rendered rather than summarised: the digest of
    these bytes is what gets recorded, so the user has to be able to read
    what they are agreeing to.
    """

    def __init__(self, parent: tkinter.Misc, terms: LicenseTerms) -> None:
        super().__init__(parent)
        self.accepted = False
        self._theme = theme_for(detect_color_scheme())
        self.title(f"Licence - {terms.license_id}")
        self.geometry(_DIALOG_GEOMETRY)
        self.protocol("WM_DELETE_WINDOW", self._decline)
        self._build(terms)
        apply_theme(self, self._theme)
        set_titlebar_dark(self, dark=self._theme.is_dark)
        self.transient(parent.winfo_toplevel())
        self.grab_set()

    def _build(self, terms: LicenseTerms) -> None:
        self._build_buttons()
        ttk.Label(self, text=_INTRO, wraplength=700).pack(
            anchor="w", padx=12, pady=(12, 4),
        )
        ttk.Label(self, text=terms.url).pack(anchor="w", padx=12, pady=(0, 8))
        self._build_text(terms.text)

    def _build_text(self, text: str) -> None:
        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 12))
        self._text = tkinter.Text(
            body,
            wrap="word",
            background=self._theme.surface,
            foreground=self._theme.foreground,
            insertbackground=self._theme.foreground,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            body, orient="vertical", command=self._text.yview,
        )
        self._text.configure(yscrollcommand=scrollbar.set)
        self._text.insert("1.0", text)
        self._text.configure(state="disabled")
        scrollbar.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)

    def _build_buttons(self) -> None:
        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x", padx=12, pady=8)
        self._accept_button = ttk.Button(
            footer, text="Accept", command=self._accept,
        )
        self._accept_button.pack(side="right")
        self._decline_button = ttk.Button(
            footer, text="Decline", command=self._decline,
        )
        self._decline_button.pack(side="right", padx=(0, 8))

    def _accept(self) -> None:
        self.accepted = True
        self._close()

    def _decline(self) -> None:
        self.accepted = False
        self._close()

    def _close(self) -> None:
        self.grab_release()
        self.destroy()

    def accept_button(self) -> ttk.Button:
        return self._accept_button

    def decline_button(self) -> ttk.Button:
        return self._decline_button

    def license_text(self) -> str:
        return self._text.get("1.0", "end-1c")

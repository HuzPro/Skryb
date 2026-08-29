from unittest.mock import MagicMock

import pytest

from voicetotext.settings.license_consent import (
    _LicenseConsentDialog,
    collect_license_acceptance,
)
from voicetotext.transcription.license_policy import LicenseTerms


_GATED = "crisperwhisper-2.0-turbo"
_TERMS = LicenseTerms(
    license_id="example-nc",
    url="https://example.invalid/LICENSE.md",
    text="Non-commercial research use only.",
    digest="d1",
)


def _gated_service() -> MagicMock:
    service = MagicMock()
    service.license_acceptance_required.return_value = True
    service.license_terms.return_value = _TERMS
    return service


class TestCollectLicenseAcceptance:
    def test_should_not_prompt_for_an_ungated_model(self) -> None:
        service = MagicMock()
        service.license_acceptance_required.return_value = False
        prompts: list[LicenseTerms] = []

        allowed = collect_license_acceptance(
            service, "parakeet-tdt-0.6b-v3", MagicMock(),
            lambda _parent, terms: prompts.append(terms) or True,
        )

        assert allowed is True
        assert prompts == []
        assert service.accept_license.called is False

    def test_should_record_acceptance_and_allow_when_confirmed(self) -> None:
        service = _gated_service()

        allowed = collect_license_acceptance(
            service, _GATED, MagicMock(), lambda _parent, _terms: True,
        )

        assert allowed is True
        service.accept_license.assert_called_once_with(_GATED)

    def test_should_show_the_terms_the_service_supplies(self) -> None:
        service = _gated_service()
        shown: list[LicenseTerms] = []

        collect_license_acceptance(
            service, _GATED, MagicMock(),
            lambda _parent, terms: bool(shown.append(terms)) or True,
        )

        assert shown == [_TERMS]

    def test_should_record_nothing_when_declined(self) -> None:
        service = _gated_service()

        allowed = collect_license_acceptance(
            service, _GATED, MagicMock(), lambda _parent, _terms: False,
        )

        assert allowed is False
        assert service.accept_license.called is False


class TestLicenseConsentDialog:
    @pytest.fixture
    def dialog(self, tk_root):
        widget = _LicenseConsentDialog(tk_root, _TERMS)
        yield widget
        if widget.winfo_exists():
            widget.grab_release()
            widget.destroy()

    def test_should_render_the_full_licence_text(self, dialog) -> None:
        assert dialog.license_text() == _TERMS.text

    def test_should_report_acceptance_when_accept_is_pressed(self, dialog) -> None:
        dialog.accept_button().invoke()

        assert dialog.accepted is True

    def test_should_report_refusal_when_decline_is_pressed(self, dialog) -> None:
        dialog.decline_button().invoke()

        assert dialog.accepted is False

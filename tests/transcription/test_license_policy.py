"""The licence gate, over two made-up gated models: no model in the catalogue needs a licence now."""

import hashlib
from dataclasses import replace

import pytest

from tests.fakes.config import FakeConfig
from skryb.core.exceptions import LicenseNotAcceptedError
from skryb.transcription import license_policy
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import (
    ALL_MODELS,
    DEFAULT_MODEL,
    ModelInfo,
)

_LICENCE_TEXT = "Example Model Licence\n\n1. Non-commercial research only.\n2. Keep this notice.\n"
_SMALL = ModelInfo(
    name="example-small", size_mb=1, description="Example", repo_id="example/small",
    license_id="example-nc", license_url="https://example.com/small/LICENSE.md",
)
_LARGE = replace(_SMALL, name="example-large", repo_id="example/large", license_url="https://example.com/large/LICENSE.md")
_ACCEPTANCE_KEY = "licenses.example-nc.accepted_digest"


@pytest.fixture(autouse=True)
def _shipped_licence(monkeypatch: pytest.MonkeyPatch) -> None:
    def licence_bytes(license_id: str) -> bytes:
        if license_id != "example-nc":
            raise FileNotFoundError(license_id)
        return _LICENCE_TEXT.encode("utf-8")

    monkeypatch.setattr(license_policy, "_license_bytes", licence_bytes)


def _policy(config: FakeConfig | None = None) -> LicensePolicy:
    return LicensePolicy(config or FakeConfig())


class TestUngatedModels:
    def test_should_never_require_acceptance(self) -> None:
        assert _policy().requires_acceptance(ALL_MODELS[DEFAULT_MODEL]) is False

    def test_should_not_raise_when_asserted(self) -> None:
        _policy().assert_accepted(ALL_MODELS[DEFAULT_MODEL])


class TestGatedModels:
    def test_should_require_acceptance_before_any_consent_is_recorded(self) -> None:
        assert _policy().requires_acceptance(_SMALL) is True

    def test_should_raise_naming_the_model_and_its_licence(self) -> None:
        with pytest.raises(LicenseNotAcceptedError) as failure:
            _policy().assert_accepted(_SMALL)

        assert "example-small" in str(failure.value)
        assert "example-nc" in str(failure.value)

    def test_should_stop_requiring_acceptance_once_accepted(self) -> None:
        policy = _policy()

        policy.accept(_SMALL)

        policy.assert_accepted(_SMALL)


class TestSharedAcceptance:
    def test_accepting_one_model_covers_another_under_the_same_licence(self) -> None:
        policy = _policy()

        policy.accept(_SMALL)

        assert policy.requires_acceptance(_LARGE) is False


class TestAcceptanceRecord:
    def test_should_persist_the_digest_as_a_scalar_under_a_dotted_key(self) -> None:
        config = FakeConfig()
        policy = _policy(config)

        policy.accept(_SMALL)

        assert config.get(_ACCEPTANCE_KEY) == policy.terms(_SMALL).digest

    def test_should_save_immediately_so_a_later_cancel_cannot_lose_it(self) -> None:
        config = FakeConfig()

        _policy(config).accept(_SMALL)

        assert config.save_count == 1

    def test_a_changed_licence_digest_should_invalidate_a_prior_acceptance(self) -> None:
        config = FakeConfig({_ACCEPTANCE_KEY: hashlib.sha256(b"older text").hexdigest()})

        assert _policy(config).requires_acceptance(_SMALL) is True


class TestLicenceTerms:
    def test_terms_should_carry_the_digest_of_the_bytes_it_displays(self) -> None:
        terms = _policy().terms(_SMALL)

        assert terms.digest == hashlib.sha256(terms.text.encode("utf-8")).hexdigest()

    def test_terms_should_carry_the_models_licence_url(self) -> None:
        assert _policy().terms(_LARGE).url == "https://example.com/large/LICENSE.md"

    def test_should_fail_loudly_when_a_licence_resource_is_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            _policy().terms(replace(_SMALL, license_id="not-shipped"))

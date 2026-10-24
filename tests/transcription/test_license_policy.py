import hashlib
from dataclasses import replace

import pytest

from tests.fakes.config import FakeConfig
from skryb.core.exceptions import LicenseNotAcceptedError
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import (
    ALL_MODELS,
    CRISPERWHISPER_MODELS,
    DEFAULT_MODEL,
    ModelInfo,
)


_TURBO = "crisperwhisper-2.0-turbo"
_LARGE = "crisperwhisper-2.0-large"
_ACCEPTANCE_KEY = "licenses.crisperwhisper-nc.accepted_digest"


def _policy(config: FakeConfig | None = None) -> LicensePolicy:
    return LicensePolicy(config or FakeConfig())


class TestUngatedModels:
    def test_should_never_require_acceptance(self) -> None:
        assert _policy().requires_acceptance(ALL_MODELS[DEFAULT_MODEL]) is False

    def test_should_not_raise_when_asserted(self) -> None:
        _policy().assert_accepted(ALL_MODELS[DEFAULT_MODEL])


class TestGatedModels:
    def test_should_require_acceptance_before_any_consent_is_recorded(
        self,
    ) -> None:
        assert _policy().requires_acceptance(CRISPERWHISPER_MODELS[_TURBO]) is True

    def test_should_raise_naming_the_model_and_its_licence(self) -> None:
        with pytest.raises(LicenseNotAcceptedError) as failure:
            _policy().assert_accepted(CRISPERWHISPER_MODELS[_TURBO])

        assert _TURBO in str(failure.value)
        assert "crisperwhisper-nc" in str(failure.value)

    def test_should_stop_requiring_acceptance_once_accepted(self) -> None:
        policy = _policy()

        policy.accept(CRISPERWHISPER_MODELS[_TURBO])

        policy.assert_accepted(CRISPERWHISPER_MODELS[_TURBO])


class TestSharedAcceptance:
    """The licence bytes at both pinned commits are identical, so one
    acceptance must cover both variants - asking twice for the same text
    would be noise, and recording it twice would be two truths.
    """

    def test_should_expose_identical_terms_for_both_variants(self) -> None:
        policy = _policy()

        turbo = policy.terms(CRISPERWHISPER_MODELS[_TURBO])
        large = policy.terms(CRISPERWHISPER_MODELS[_LARGE])

        assert turbo.text == large.text
        assert turbo.digest == large.digest

    def test_accepting_one_variant_should_cover_the_other(self) -> None:
        policy = _policy()

        policy.accept(CRISPERWHISPER_MODELS[_TURBO])

        assert policy.requires_acceptance(CRISPERWHISPER_MODELS[_LARGE]) is False


class TestAcceptanceRecord:
    def test_should_persist_the_digest_as_a_scalar_under_a_dotted_key(
        self,
    ) -> None:
        config = FakeConfig()
        policy = _policy(config)

        policy.accept(CRISPERWHISPER_MODELS[_TURBO])

        assert config.get(_ACCEPTANCE_KEY) == (
            policy.terms(CRISPERWHISPER_MODELS[_TURBO]).digest
        )

    def test_should_save_immediately_so_a_later_cancel_cannot_lose_it(
        self,
    ) -> None:
        config = FakeConfig()

        _policy(config).accept(CRISPERWHISPER_MODELS[_TURBO])

        assert config.save_count == 1

    def test_a_changed_licence_digest_should_invalidate_a_prior_acceptance(
        self,
    ) -> None:
        # Upstream edits its LICENSE.md without touching the version
        # header, so only the digest can tell that the user agreed to
        # wording this build no longer ships.
        config = FakeConfig({_ACCEPTANCE_KEY: hashlib.sha256(b"older text").hexdigest()})

        assert _policy(config).requires_acceptance(
            CRISPERWHISPER_MODELS[_TURBO],
        ) is True


class TestShippedLicenceResources:
    def test_terms_should_carry_the_digest_of_the_bytes_it_displays(self) -> None:
        terms = _policy().terms(CRISPERWHISPER_MODELS[_TURBO])

        assert terms.digest == hashlib.sha256(
            terms.text.encode("utf-8"),
        ).hexdigest()

    def test_terms_should_carry_the_models_pinned_licence_url(self) -> None:
        terms = _policy().terms(CRISPERWHISPER_MODELS[_TURBO])

        assert terms.url == CRISPERWHISPER_MODELS[_TURBO].license_url

    @pytest.mark.parametrize(
        "model_name",
        sorted(name for name, info in ALL_MODELS.items() if info.license_id),
    )
    def test_every_gated_catalog_entry_should_ship_its_licence_text(
        self, model_name: str,
    ) -> None:
        terms = _policy().terms(ALL_MODELS[model_name])

        assert terms.text.strip() != ""

    def test_should_fail_loudly_when_a_licence_resource_is_missing(self) -> None:
        undeclared: ModelInfo = replace(
            CRISPERWHISPER_MODELS[_TURBO], license_id="not-shipped",
        )

        with pytest.raises(FileNotFoundError):
            _policy().terms(undeclared)

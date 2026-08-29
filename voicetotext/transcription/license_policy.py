import hashlib
from dataclasses import dataclass
from importlib import resources

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.exceptions import LicenseNotAcceptedError
from voicetotext.transcription.model_info import ModelInfo


_LICENSE_PACKAGE = "voicetotext"
_LICENSE_DIRECTORY = "licenses"


def _acceptance_key(license_id: str) -> str:
    return f"licenses.{license_id}.accepted_digest"


def _license_bytes(license_id: str) -> bytes:
    # Read as bytes, never as text: the digest the acceptance is keyed to
    # is over the shipped file exactly as it is, and text mode would
    # rewrite its line endings on Windows.
    resource = (
        resources.files(_LICENSE_PACKAGE) / _LICENSE_DIRECTORY / f"{license_id}.md"
    )
    return resource.read_bytes()


@dataclass(frozen=True)
class LicenseTerms:
    """The exact wording a user is asked to agree to, and its digest.

    ``digest`` is the SHA-256 of the shipped bytes, which is what an
    acceptance records. Keying acceptance to the text rather than to a
    version string is deliberate: upstream has already edited this
    licence while leaving its "Version 1.0" header untouched, so only
    the digest identifies what a user actually agreed to.
    """
    license_id: str
    url: str
    text: str
    digest: str


class LicensePolicy:
    """Decides whether a gated model may be acquired or used.

    One instance is built at startup and threaded through the model
    manager, the engine factory, and both settings services, so every
    path that downloads, builds, or loads weights consults the same
    acceptance state. Adding a new entry point cannot silently create a
    bypass, because the check lives here rather than in the UI.

    Acceptance is stored as a scalar under
    ``licenses.<license_id>.accepted_digest``. A licence whose text
    changes therefore invalidates the prior acceptance and is asked
    again.

    Forging that config entry by hand is out of scope: this is a local
    single-user app, and the gate exists to ensure informed consent, not
    to defend against the machine's owner.
    """

    def __init__(self, config: ConfigurationStore) -> None:
        self._config = config

    def requires_acceptance(self, model_info: ModelInfo) -> bool:
        """Whether *model_info* is gated and not yet accepted as shipped."""
        if not model_info.license_id:
            return False
        return self._accepted_digest(model_info.license_id) != (
            self.terms(model_info).digest
        )

    def assert_accepted(self, model_info: ModelInfo) -> None:
        if not self.requires_acceptance(model_info):
            return
        raise LicenseNotAcceptedError(
            f"'{model_info.name}' is licensed under {model_info.license_id} "
            "and its terms have not been accepted. Open Settings, select "
            f"the model, and accept the licence at {model_info.license_url}"
        )

    def terms(self, model_info: ModelInfo) -> LicenseTerms:
        raw = _license_bytes(model_info.license_id)
        return LicenseTerms(
            license_id=model_info.license_id,
            url=model_info.license_url,
            text=raw.decode("utf-8"),
            digest=hashlib.sha256(raw).hexdigest(),
        )

    def accept(self, model_info: ModelInfo) -> None:
        """Record consent to the text this build ships, and persist it now.

        Saving here rather than leaving it to the caller's own save keeps
        the record from being lost when the user accepts and then cancels
        out of the surrounding settings edit.
        """
        self._config.set(
            _acceptance_key(model_info.license_id), self.terms(model_info).digest,
        )
        self._config.save()

    def _accepted_digest(self, license_id: str) -> str:
        stored = self._config.get(_acceptance_key(license_id), "")
        return str(stored) if stored else ""

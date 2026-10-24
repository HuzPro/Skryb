from tests.fakes.config import FakeConfig
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import ModelInfo


class AllowAllLicensePolicy(LicensePolicy):
    """Grants every licence, so tests can build gated engines freely.

    Production never wires this: a permissive default in the real code
    would recreate the bypass the gate exists to close. Tests that
    exercise the gate itself use a real
    :class:`~skryb.transcription.license_policy.LicensePolicy` over
    a :class:`~tests.fakes.config.FakeConfig` instead.
    """

    def __init__(self) -> None:
        super().__init__(FakeConfig())

    def requires_acceptance(self, model_info: ModelInfo) -> bool:
        return False

    def assert_accepted(self, model_info: ModelInfo) -> None:
        return

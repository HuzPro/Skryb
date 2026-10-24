"""The app's own licence acceptance, for tooling that loads real weights.

Benchmarks and opt-in real-model runs download and run the same gated
weights the app does, and the licence covers use and generated outputs,
not only the download. So they consult the same acceptance record the
app writes, read straight from the user's config file: the only way past
the gate is to accept the licence in the app first.
"""

from skryb.core.json_config_adapter import JsonConfigAdapter
from skryb.core.paths import AppPaths
from skryb.transcription.license_policy import LicensePolicy


def user_license_policy() -> LicensePolicy:
    config = JsonConfigAdapter(AppPaths().config_path)
    config.load()
    return LicensePolicy(config)

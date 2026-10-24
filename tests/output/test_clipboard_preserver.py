import pytest

from skryb.output.clipboard_preserver import ClipboardPreserver


class TestClipboardPreserverIsAbstract:
    def test_should_not_be_directly_instantiable(self):
        with pytest.raises(TypeError):
            ClipboardPreserver()

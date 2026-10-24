import pytest
from unittest.mock import MagicMock

from skryb.settings.download_coordinator import DownloadCoordinator


def _make_coordinator(model_manager=None):
    return DownloadCoordinator(model_manager or MagicMock())


class TestDownloadCoordinatorStart:
    def test_should_enqueue_via_model_manager(self):
        manager = MagicMock()
        handle = MagicMock()
        manager.download.return_value = handle
        coord = _make_coordinator(manager)

        coord.start("parakeet-tdt-0.6b-v3")

        manager.download.assert_called_once_with("parakeet-tdt-0.6b-v3")

    def test_should_track_handle_for_later_pause(self):
        manager = MagicMock()
        handle = MagicMock()
        manager.download.return_value = handle
        coord = _make_coordinator(manager)

        coord.start("parakeet-tdt-0.6b-v3")
        coord.pause("parakeet-tdt-0.6b-v3")

        handle.pause.assert_called_once()

    def test_should_replace_handle_when_started_twice(self):
        manager = MagicMock()
        first = MagicMock()
        second = MagicMock()
        manager.download.side_effect = [first, second]
        coord = _make_coordinator(manager)

        coord.start("parakeet-tdt-0.6b-v3")
        coord.start("parakeet-tdt-0.6b-v3")  # resume scenario - re-enqueue
        coord.cancel("parakeet-tdt-0.6b-v3")

        first.cancel.assert_not_called()
        second.cancel.assert_called_once()


class TestDownloadCoordinatorPause:
    def test_should_raise_lookup_error_when_no_active_download(self):
        coord = _make_coordinator()

        with pytest.raises(LookupError):
            coord.pause("parakeet-tdt-0.6b-v3")

    def test_should_forward_pause_to_handle(self):
        manager = MagicMock()
        handle = MagicMock()
        manager.download.return_value = handle
        coord = _make_coordinator(manager)

        coord.start("parakeet-tdt-0.6b-v3")
        coord.pause("parakeet-tdt-0.6b-v3")

        handle.pause.assert_called_once()


class TestDownloadCoordinatorResume:
    def test_should_re_enqueue_via_model_manager(self):
        manager = MagicMock()
        manager.download.return_value = MagicMock()
        coord = _make_coordinator(manager)

        coord.start("parakeet-tdt-0.6b-v3")
        coord.resume("parakeet-tdt-0.6b-v3")

        assert manager.download.call_count == 2
        assert manager.download.call_args_list[0].args == ("parakeet-tdt-0.6b-v3",)
        assert manager.download.call_args_list[1].args == ("parakeet-tdt-0.6b-v3",)


class TestDownloadCoordinatorCancel:
    def test_should_raise_lookup_error_when_no_active_download(self):
        coord = _make_coordinator()

        with pytest.raises(LookupError):
            coord.cancel("parakeet-tdt-0.6b-v3")

    def test_should_forward_cancel_and_drop_handle(self):
        manager = MagicMock()
        handle = MagicMock()
        manager.download.return_value = handle
        coord = _make_coordinator(manager)

        coord.start("parakeet-tdt-0.6b-v3")
        coord.cancel("parakeet-tdt-0.6b-v3")

        handle.cancel.assert_called_once()
        # After cancel, the handle should be gone - second cancel is a LookupError.
        with pytest.raises(LookupError):
            coord.cancel("parakeet-tdt-0.6b-v3")

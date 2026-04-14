from unittest.mock import patch

from voicetotext.output.sleeper import Sleeper, TimeSleeper


class TestTimeSleeperIsASleeper:
    def test_should_implement_sleeper_interface(self):
        sleeper = TimeSleeper()

        assert isinstance(sleeper, Sleeper)


class TestTimeSleeperDelegatesToTimeModule:
    @patch("voicetotext.output.sleeper.time")
    def test_should_call_time_sleep_with_given_duration(self, mock_time):
        sleeper = TimeSleeper()

        sleeper.sleep(0.25)

        mock_time.sleep.assert_called_once_with(0.25)

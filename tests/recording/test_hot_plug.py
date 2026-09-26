from tests.fakes.events import RecordingEventBus
from voicetotext.core.events import MicrophoneConnectedEvent
from voicetotext.recording.hot_plug import (
    SETTLE_MS,
    MicrophoneHotPlug,
    plugged_changes,
)

_MAONO = "Microphone (Maono PD300X Microp"
_MAONO_FULL = "Microphone (Maono PD300X Microphone)"
_BUILT_IN = "Microphone Array (Realtek(R) Au"
_BUILT_IN_FULL = "Microphone Array (Realtek(R) Audio)"
_JABRA = "Headset (Jabra Evolve 20 MS Ste"
_JABRA_FULL = "Headset (Jabra Evolve 20 MS Stereo)"
_FULL_NAMES = {_MAONO: _MAONO_FULL, _BUILT_IN: _BUILT_IN_FULL, _JABRA: _JABRA_FULL}


class TestPluggedChanges:
    def test_reports_new_and_gone_microphones_in_order(self):
        added, removed = plugged_changes(
            ["Mic A", "Mic B", "Mic C"], ["Mic A", "Mic D", "Mic C", "Mic E"],
        )

        assert added == ["Mic D", "Mic E"]
        assert removed == ["Mic B"]

    def test_a_duplicate_is_reported_once(self):
        added, removed = plugged_changes(
            ["Mic A", "Mic B", "Mic B"], ["Mic A", "Mic C", "Mic C", "Mic C"],
        )

        assert added == ["Mic C"]
        assert removed == ["Mic B"]

    def test_blank_names_are_never_a_microphone(self):
        added, removed = plugged_changes(["", "Mic A", "  "], ["Mic A", "", "\t"])

        assert (added, removed) == ([], [])

    def test_the_same_list_twice_changes_nothing(self):
        names = [f"Mikrofon {index} (Gerät für Café ✓)" for index in range(1500)]

        assert plugged_changes(names, list(reversed(names))) == ([], [])

    def test_thousands_of_devices_with_one_swapped(self):
        before = [f"Mic {index}" for index in range(3000)]
        after = [*before[:1200], "USB Mic ✓", *before[1201:]]

        assert plugged_changes(before, after) == (["USB Mic ✓"], ["Mic 1200"])


class FakeRoot:
    def __init__(self) -> None:
        self.pending: dict[str, object] = {}
        self.delays: list[int] = []
        self._next = 0

    def after(self, ms, callback) -> str:
        self._next += 1
        after_id = f"after#{self._next}"
        self.pending[after_id] = callback
        self.delays.append(ms)
        return after_id

    def after_cancel(self, after_id: str) -> None:
        self.pending.pop(after_id, None)

    def run_pending(self) -> None:
        pending, self.pending = self.pending, {}
        for callback in pending.values():
            callback()


class FakeMicrophones:
    """PortAudio keeps the list from its last scan until it is told to look."""

    def __init__(self, seen: list[str]) -> None:
        self.seen = list(seen)
        self.plugged = list(seen)
        self.busy = False
        self.rescans = 0

    def names(self, rescan: bool = False) -> list[str]:
        if rescan and not self.busy:
            self.rescans += 1
            self.seen = list(self.plugged)
        return list(self.seen)

    def labels(self, names):
        return {name: _FULL_NAMES.get(name, name) for name in names}


class Desk:
    def __init__(self, seen: list[str], preferred=(), chosen: str = "") -> None:
        self.root = FakeRoot()
        self.microphones = FakeMicrophones(seen)
        self.bus = RecordingEventBus()
        self.preferred = list(preferred)
        self.chosen = chosen
        self.background: list[object] = []
        self.hot_plug = MicrophoneHotPlug(
            self.root,
            self.microphones,
            preferred=lambda: self.preferred,
            chosen=lambda: self.chosen,
            event_bus=self.bus,
            run_in_background=self._run,
        )

    def _run(self, action) -> None:
        self.background.append(action)
        action()

    def plug_in(self, *names: str) -> None:
        self.microphones.plugged += names
        self.hot_plug.devices_changed()
        self.root.run_pending()

    def unplug(self, name: str) -> None:
        self.microphones.plugged.remove(name)
        self.hot_plug.devices_changed()
        self.root.run_pending()

    def notices(self) -> list[tuple[str, bool]]:
        return [
            (event.name, event.will_be_used)
            for event in self.bus.events_of(MicrophoneConnectedEvent)
        ]


class TestPluggingIn:
    def test_a_preferred_microphone_that_outranks_the_current_one_is_announced(self):
        desk = Desk([_BUILT_IN], preferred=[_MAONO_FULL, _BUILT_IN_FULL])

        desk.plug_in(_MAONO)

        assert desk.notices() == [(_MAONO_FULL, True)]

    def test_a_preferred_microphone_ranked_below_the_current_one_is_quiet(self):
        desk = Desk([_BUILT_IN], preferred=[_BUILT_IN_FULL, _MAONO_FULL])

        desk.plug_in(_MAONO)

        assert desk.notices() == []

    def test_a_microphone_not_in_the_list_is_offered_through_the_tray(self):
        desk = Desk([_BUILT_IN], preferred=[_BUILT_IN_FULL])

        desk.plug_in(_JABRA)

        assert desk.notices() == [(_JABRA_FULL, False)]

    def test_the_chosen_microphone_coming_back_says_it_will_be_used(self):
        desk = Desk([_BUILT_IN], chosen=_MAONO)

        desk.plug_in(_MAONO)

        assert desk.notices() == [(_MAONO_FULL, True)]

    def test_with_no_preferences_any_new_microphone_is_offered(self):
        desk = Desk([_BUILT_IN])

        desk.plug_in(_JABRA, "Mikrofon (Gerät für Café ✓)")

        assert desk.notices() == [
            (_JABRA_FULL, False), ("Mikrofon (Gerät für Café ✓)", False),
        ]

    def test_unplugging_says_nothing_until_the_next_recording_notices(self):
        desk = Desk([_BUILT_IN, _MAONO], chosen=_MAONO)

        desk.unplug(_MAONO)

        assert desk.notices() == []

    def test_other_devices_coming_and_going_say_nothing(self):
        desk = Desk([_BUILT_IN])

        desk.plug_in()

        assert desk.notices() == []

    def test_mid_recording_it_does_not_restart_the_audio_stack(self):
        desk = Desk([_BUILT_IN])
        desk.microphones.busy = True

        desk.plug_in(_JABRA)

        assert desk.microphones.rescans == 0
        assert desk.notices() == []


class TestSettling:
    def test_a_burst_of_device_changes_is_looked_at_once_after_a_second(self):
        desk = Desk([_BUILT_IN])
        desk.microphones.plugged.append(_JABRA)

        for _ in range(6):
            desk.hot_plug.devices_changed()

        assert len(desk.root.pending) == 1
        assert set(desk.root.delays) == {SETTLE_MS}
        desk.root.run_pending()
        assert desk.microphones.rescans == 1
        assert desk.notices() == [(_JABRA_FULL, False)]

    def test_the_rescan_happens_off_the_tk_thread(self):
        desk = Desk([_BUILT_IN])

        desk.plug_in(_JABRA)

        assert len(desk.background) == 1

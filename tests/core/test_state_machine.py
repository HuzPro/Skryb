import threading

import pytest

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import StateChangedEvent
from voicetotext.core.state_machine import AppState, AppStateMachine


class TestAppStateMachineInitialState:
    def test_should_start_in_idle_state(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)

        assert machine.state == AppState.IDLE

    def test_should_accept_loading_as_initial_state(self):
        # ``LOADING`` is the entry state when the engine is being
        # constructed on a background thread at app startup — the
        # state machine must support beginning there rather than
        # transitioning from ``IDLE``.
        bus = SimpleEventBus()
        machine = AppStateMachine(bus, initial_state=AppState.LOADING)

        assert machine.state == AppState.LOADING


class TestAppStateMachineValidTransitions:
    def test_should_transition_from_idle_to_recording(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)

        machine.transition_to(AppState.RECORDING)

        assert machine.state == AppState.RECORDING

    def test_should_transition_from_recording_to_transcribing(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        machine.transition_to(AppState.RECORDING)

        machine.transition_to(AppState.TRANSCRIBING)

        assert machine.state == AppState.TRANSCRIBING

    def test_should_transition_from_transcribing_to_idle(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        machine.transition_to(AppState.RECORDING)
        machine.transition_to(AppState.TRANSCRIBING)

        machine.transition_to(AppState.IDLE)

        assert machine.state == AppState.IDLE

    def test_should_transition_from_recording_to_idle_on_cancel(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        machine.transition_to(AppState.RECORDING)

        machine.transition_to(AppState.IDLE)

        assert machine.state == AppState.IDLE

    def test_should_transition_from_any_state_to_error(self):
        bus = SimpleEventBus()

        for start in [AppState.IDLE, AppState.RECORDING, AppState.TRANSCRIBING]:
            machine = AppStateMachine(bus)
            if start == AppState.RECORDING:
                machine.transition_to(AppState.RECORDING)
            elif start == AppState.TRANSCRIBING:
                machine.transition_to(AppState.RECORDING)
                machine.transition_to(AppState.TRANSCRIBING)

            machine.transition_to(AppState.ERROR)

            assert machine.state == AppState.ERROR

    def test_should_transition_from_error_to_idle(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        machine.transition_to(AppState.ERROR)

        machine.transition_to(AppState.IDLE)

        assert machine.state == AppState.IDLE

    def test_should_transition_from_loading_to_idle(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus, initial_state=AppState.LOADING)

        machine.transition_to(AppState.IDLE)

        assert machine.state == AppState.IDLE

    def test_should_transition_from_loading_to_error(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus, initial_state=AppState.LOADING)

        machine.transition_to(AppState.ERROR)

        assert machine.state == AppState.ERROR


class TestAppStateMachineInvalidTransitions:
    def test_should_reject_idle_to_transcribing(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)

        with pytest.raises(ValueError):
            machine.transition_to(AppState.TRANSCRIBING)

    def test_should_reject_transcribing_to_recording(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        machine.transition_to(AppState.RECORDING)
        machine.transition_to(AppState.TRANSCRIBING)

        with pytest.raises(ValueError):
            machine.transition_to(AppState.RECORDING)

    def test_should_reject_error_to_recording(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        machine.transition_to(AppState.ERROR)

        with pytest.raises(ValueError):
            machine.transition_to(AppState.RECORDING)

    def test_should_reject_idle_to_loading(self):
        # LOADING is strictly the startup entry state, not something
        # the app can fall back into once the engine is live. Allowing
        # the transition would let a buggy hot-swap leave the app in
        # LOADING with no thread responsible for finishing the load.
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)

        with pytest.raises(ValueError):
            machine.transition_to(AppState.LOADING)

    def test_should_reject_loading_to_recording(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus, initial_state=AppState.LOADING)

        with pytest.raises(ValueError):
            machine.transition_to(AppState.RECORDING)


class TestAppStateMachinePublishesEvents:
    def test_should_publish_state_changed_event_on_transition(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        received = []
        bus.subscribe(StateChangedEvent, received.append)

        machine.transition_to(AppState.RECORDING)

        assert len(received) == 1
        assert received[0].old_state == AppState.IDLE.value
        assert received[0].new_state == AppState.RECORDING.value

    def test_should_not_publish_event_on_invalid_transition(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        received = []
        bus.subscribe(StateChangedEvent, received.append)

        with pytest.raises(ValueError):
            machine.transition_to(AppState.TRANSCRIBING)

        assert len(received) == 0


class TestAppStateMachineThreadSafety:
    def test_should_handle_concurrent_transitions_safely(self):
        bus = SimpleEventBus()
        machine = AppStateMachine(bus)
        errors = []
        successes = []

        def try_transition():
            try:
                machine.transition_to(AppState.RECORDING)
                successes.append(True)
            except ValueError:
                errors.append(True)

        threads = [threading.Thread(target=try_transition) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 1
        assert len(errors) == 9
        assert machine.state == AppState.RECORDING

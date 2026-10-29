from unittest.mock import MagicMock, patch

import pytest

from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import (
    MicrophoneConnectedEvent,
    MicrophoneMissingEvent,
    MicrophoneSilentEvent,
    EngineLoadFailedEvent,
    ErrorOccurredEvent,
    RecordingCappedEvent,
    StateChangedEvent,
    TranscriptionEmptyEvent,
    TranscriptionFailedEvent,
)
from skryb.core.state_machine import AppState, AppStateMachine
from skryb.tray.app import SystemTrayApp, format_duration


def _build_app(on_open_settings=None, tray_factory=None):
    use_case = MagicMock()
    event_bus = SimpleEventBus()
    state_machine = AppStateMachine(event_bus=event_bus)
    listener = MagicMock()
    return SystemTrayApp(
        use_case=use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        hotkey_listener=listener,
        tray_factory=tray_factory or (lambda **kwargs: MagicMock()),
        on_open_settings=on_open_settings,
    )


class TestSettingsClick:
    def test_should_call_open_settings_when_click_handled(self):
        call_count = 0

        def on_open() -> None:
            nonlocal call_count
            call_count += 1

        app = _build_app(on_open_settings=on_open)

        app._handle_settings_clicked()

        assert call_count == 1

    def test_should_noop_when_no_settings_callback_registered(self):
        app = _build_app(on_open_settings=None)

        # No callback wired: click must not raise.
        app._handle_settings_clicked()


class TestTrayIconFactoryInvocation:
    def test_should_invoke_factory_with_callbacks_on_run(self):
        captured: dict = {}

        def factory(**kwargs):
            captured.update(kwargs)
            icon = MagicMock()
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(on_open_settings=lambda: None, tray_factory=factory)

        app.run()

        assert callable(captured["get_state_label"])
        assert callable(captured["on_settings_clicked"])
        assert callable(captured["on_quit_clicked"])
        assert captured["initial_state"] == AppState.IDLE


class TestStateLabel:
    def test_the_menu_names_the_running_build_beside_the_state(self):
        app = SystemTrayApp(
            use_case=MagicMock(),
            state_machine=AppStateMachine(event_bus=SimpleEventBus()),
            event_bus=SimpleEventBus(),
            hotkey_listener=MagicMock(),
            tray_factory=lambda **kwargs: MagicMock(),
            build_status=lambda: "main fe502a2, restart for main 1a2b3c4",
        )

        assert app._current_state_label() == (
            "Idle - ready  ·  main fe502a2, restart for main 1a2b3c4"
        )

    def test_the_state_stands_alone_when_the_build_is_unknown(self):
        assert _build_app()._current_state_label() == "Idle - ready"


class TestStateChangeForwarding:
    def test_should_call_update_state_on_icon_when_state_changes(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._handle_state_changed(
            StateChangedEvent(
                old_state=AppState.IDLE.value,
                new_state=AppState.RECORDING.value,
            )
        )

        icon.update_state.assert_called_with(AppState.RECORDING, "Recording...")

    def test_should_label_loading_state_for_user_facing_display(self):
        # Without a label, the tray would render "Unknown" while the
        # engine is loading on a background thread - confusing to a
        # user expecting "the app is starting".
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._handle_state_changed(
            StateChangedEvent(
                old_state=AppState.LOADING.value,
                new_state=AppState.LOADING.value,
            )
        )

        called_with_state, called_with_label = icon.update_state.call_args.args
        assert called_with_state == AppState.LOADING
        assert called_with_label.lower() != "unknown", (
            f"LOADING should have a user-friendly label, got '{called_with_label}'"
        )


class TestPauseResumeListener:
    def test_should_call_pause_on_listener_when_pause_listener_called(self):
        listener = MagicMock()
        app = SystemTrayApp(
            use_case=MagicMock(),
            state_machine=AppStateMachine(event_bus=SimpleEventBus()),
            event_bus=SimpleEventBus(),
            hotkey_listener=listener,
            tray_factory=lambda **kwargs: MagicMock(),
        )

        app.pause_listener()

        listener.pause.assert_called_once()

    def test_should_call_resume_on_listener_when_resume_listener_called(self):
        listener = MagicMock()
        app = SystemTrayApp(
            use_case=MagicMock(),
            state_machine=AppStateMachine(event_bus=SimpleEventBus()),
            event_bus=SimpleEventBus(),
            hotkey_listener=listener,
            tray_factory=lambda **kwargs: MagicMock(),
        )

        app.resume_listener()

        listener.resume.assert_called_once()


class TestFormatDuration:
    """``format_duration`` powers both the recording-cap notification
    and the live elapsed-time tooltip - its output ends up directly
    in user-facing surfaces, so the formatting rules are tested in
    isolation rather than via the consumers.
    """

    def test_should_render_zero_as_minutes_seconds(self):
        assert format_duration(0) == "0:00"

    def test_should_zero_pad_seconds_below_ten(self):
        assert format_duration(7) == "0:07"

    def test_should_round_sub_second_values_down(self):
        # Live tooltip should never claim more time than has actually
        # elapsed - floor rather than round.
        assert format_duration(0.9) == "0:00"
        assert format_duration(1.999) == "0:01"

    def test_should_render_minute_boundary(self):
        assert format_duration(60) == "1:00"

    def test_should_render_arbitrary_minute_value(self):
        assert format_duration(47) == "0:47"
        assert format_duration(600) == "10:00"

    def test_should_render_hour_when_over_one_hour(self):
        # The recording cap is configurable; long caps should not
        # collapse into a confusing "60:00", they should escalate to
        # H:MM:SS.
        assert format_duration(3600) == "1:00:00"
        assert format_duration(3661) == "1:01:01"

    def test_should_treat_negative_values_as_zero(self):
        # Defensive - clock skew between recording-start timestamp
        # and tooltip-tick reads should not produce gibberish.
        assert format_duration(-5) == "0:00"


class TestQuitHandling:
    """Quit must reliably terminate the process even when one of the
    cleanup steps hangs. The strategy is to schedule an unconditional
    hard-exit via a timer *before* running cleanup, so the timer fires
    if cleanup itself never finishes. Cleanup also runs inline on the
    GUI thread because the Qt tray icon operations are not safe from
    background threads on Windows.
    """

    def _build_app_with_mocks(self):
        use_case = MagicMock()
        listener = MagicMock()
        download_manager = MagicMock()
        icon = MagicMock()
        app = SystemTrayApp(
            use_case=use_case,
            state_machine=AppStateMachine(event_bus=SimpleEventBus()),
            event_bus=SimpleEventBus(),
            hotkey_listener=listener,
            tray_factory=lambda **kwargs: icon,
            download_manager=download_manager,
        )
        # Inject the icon as if ``run`` had assigned it. This is the
        # state ``_handle_quit`` operates in.
        app._icon = icon
        return app, use_case, listener, download_manager, icon

    def test_should_end_the_process_after_cleanup_runs(self):
        with (
            patch("skryb.tray.app.threading.Timer"),
            patch("skryb.tray.app._exit_now") as exit_now,
        ):
            app, *_ = self._build_app_with_mocks()

            app._handle_quit()

            exit_now.assert_called_with()

    def test_should_schedule_hard_exit_timer_before_cleanup_starts(self):
        with (
            patch("skryb.tray.app.threading.Timer") as Timer,
            patch("skryb.tray.app._exit_now") as exit_now,
        ):
            app, *_ = self._build_app_with_mocks()

            app._handle_quit()

            Timer.assert_called_once()
            interval, callback = Timer.call_args.args
            assert interval > 0
            # The timer's job is to force-exit if cleanup hangs.
            # Verify the callback actually terminates by invoking it
            # and confirming ``_exit_now`` fires.
            exit_now.reset_mock()
            callback()
            exit_now.assert_called_with()
            Timer.return_value.start.assert_called_once()

    def test_should_run_cleanup_steps_in_documented_order(self):
        with (
            patch("skryb.tray.app.threading.Timer"),
            patch("skryb.tray.app._exit_now"),
        ):
            app, use_case, listener, download_manager, icon = (
                self._build_app_with_mocks()
            )
            call_order: list[str] = []
            listener.stop.side_effect = lambda: call_order.append("listener")
            use_case.dispose.side_effect = lambda: call_order.append("use_case")
            download_manager.shutdown.side_effect = (
                lambda: call_order.append("downloads")
            )
            icon.stop.side_effect = lambda: call_order.append("icon")

            app._handle_quit()

            assert call_order == ["listener", "use_case", "downloads", "icon"]

    def test_should_continue_remaining_cleanup_when_one_step_raises(self):
        with (
            patch("skryb.tray.app.threading.Timer"),
            patch("skryb.tray.app._exit_now"),
        ):
            app, use_case, listener, download_manager, icon = (
                self._build_app_with_mocks()
            )
            use_case.dispose.side_effect = RuntimeError("dispose broke")

            app._handle_quit()

            # Listener.stop ran (it was step 1, before the raising one).
            listener.stop.assert_called_once()
            # download_manager + icon still ran despite the earlier raise.
            download_manager.shutdown.assert_called_once()
            icon.stop.assert_called_once()


class TestLiveRecordingTooltip:
    """While recording, the tray tooltip shows elapsed seconds so the
    user can see they're actually capturing audio (and roughly how
    much). The tray app hands the icon adapter a callable that
    returns ``time-since-recording-started``; the adapter is
    responsible for refreshing the tooltip on a timer. When recording
    ends the provider is cleared so the tooltip falls back to the
    static state label.
    """

    def _build_app_with_icon(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()
        return app, icon

    def test_should_install_elapsed_provider_on_entering_recording(self):
        app, icon = self._build_app_with_icon()

        app._handle_state_changed(
            StateChangedEvent(
                old_state=AppState.IDLE.value,
                new_state=AppState.RECORDING.value,
            )
        )

        icon.set_elapsed_provider.assert_called_once()
        provider = icon.set_elapsed_provider.call_args.args[0]
        assert callable(provider)

    def test_elapsed_provider_should_grow_with_wall_clock_time(self):
        # The provider is a closure over ``time.monotonic`` and a
        # captured start time. Patching ``time.monotonic`` with a
        # known sequence sidesteps Windows' 15.6 ms monotonic-clock
        # resolution that would otherwise make a short ``sleep``
        # unreliable in CI.
        with patch(
            "skryb.tray.app.time.monotonic",
            side_effect=[100.0, 100.5, 101.0],
        ):
            app, icon = self._build_app_with_icon()
            app._handle_state_changed(
                StateChangedEvent(
                    old_state=AppState.IDLE.value,
                    new_state=AppState.RECORDING.value,
                )
            )
            provider = icon.set_elapsed_provider.call_args.args[0]
            first = provider()
            second = provider()

        assert first == 0.5
        assert second == 1.0

    def test_should_clear_elapsed_provider_on_leaving_recording(self):
        app, icon = self._build_app_with_icon()
        app._handle_state_changed(
            StateChangedEvent(
                old_state=AppState.IDLE.value,
                new_state=AppState.RECORDING.value,
            )
        )
        icon.set_elapsed_provider.reset_mock()

        app._handle_state_changed(
            StateChangedEvent(
                old_state=AppState.RECORDING.value,
                new_state=AppState.TRANSCRIBING.value,
            )
        )

        icon.set_elapsed_provider.assert_called_once_with(None)

    def test_should_not_touch_elapsed_provider_for_non_recording_transitions(
        self,
    ):
        # IDLE → ERROR or LOADING → IDLE should NOT publish elapsed
        # updates, otherwise the adapter would needlessly stop and
        # restart a non-existent timer.
        app, icon = self._build_app_with_icon()

        app._handle_state_changed(
            StateChangedEvent(
                old_state=AppState.LOADING.value,
                new_state=AppState.IDLE.value,
            )
        )

        icon.set_elapsed_provider.assert_not_called()


class TestErrorNotification:
    def test_should_notify_on_error_occurred_event(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._event_bus.publish(
            ErrorOccurredEvent(error="Clipboard write failed.")
        )

        icon.notify.assert_called_once_with(
            "Skryb", "Clipboard write failed."
        )

    def test_should_notify_on_transcription_failed_event(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._event_bus.publish(
            TranscriptionFailedEvent(error="model crashed")
        )

        icon.notify.assert_called_once_with("Skryb", "model crashed")

    def test_should_notify_user_when_recording_is_auto_capped(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._event_bus.publish(
            RecordingCappedEvent(max_duration_seconds=600.0)
        )

        icon.notify.assert_called_once()
        args = icon.notify.call_args.args
        assert args[0] == "Skryb"
        assert "10:00" in args[1] or "10 min" in args[1]


class TestEngineLoadFailureNotification:
    """When the background engine load fails, the user must see the
    actionable reason (e.g. "nano-parakeet not installed: pip install
    nano-parakeet") rather than a silently stuck LOADING icon or an
    empty clipboard on the next hotkey press.
    """

    def _build_app_with_icon(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()
        return app, icon

    def test_should_notify_user_when_engine_load_fails(self):
        app, icon = self._build_app_with_icon()

        app._event_bus.publish(
            EngineLoadFailedEvent(
                model_name="nemotron-speech-streaming-en-0.6b",
                error="nano-parakeet not installed: pip install nano-parakeet",
            )
        )

        icon.notify.assert_called_once()
        args = icon.notify.call_args.args
        assert args[0] == "Skryb"
        assert "nano-parakeet not installed" in args[1]

    def test_should_flip_icon_to_error_when_engine_load_fails(self):
        # Tray balloons can be silenced by Focus Assist or DND; the
        # icon is the second channel that must reflect the broken
        # state regardless of notification visibility.
        app, icon = self._build_app_with_icon()
        icon.update_state.reset_mock()

        app._event_bus.publish(
            EngineLoadFailedEvent(model_name="parakeet-tdt-0.6b-v3", error="missing")
        )

        icon.update_state.assert_any_call(AppState.ERROR, "Error")

    def test_should_replay_failure_that_arrived_before_icon_attachment(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app._event_bus.publish(
            EngineLoadFailedEvent(model_name="parakeet-tdt-0.6b-v3", error="load failed early")
        )

        app.run()

        icon.notify.assert_called_once_with("Skryb", "load failed early")
        icon.update_state.assert_called_with(AppState.ERROR, "Error")

    def test_should_replay_ready_state_that_arrived_before_icon_attachment(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app._event_bus.publish(
            StateChangedEvent(old_state="loading", new_state="idle")
        )

        app.run()

        icon.update_state.assert_called_with(AppState.IDLE, "Idle - ready")


class TestTranscriptionEmptyNotification:
    """An empty transcription is the symptom the friend reported. The
    tray must surface "No speech detected" so the user sees a reason
    for the empty clipboard rather than silence.
    """

    def _build_app_with_icon(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()
        return app, icon

    def test_should_notify_user_when_transcription_is_empty(self):
        app, icon = self._build_app_with_icon()

        app._event_bus.publish(
            TranscriptionEmptyEvent(transcription_seconds=0.5)
        )

        icon.notify.assert_called_once()
        args = icon.notify.call_args.args
        assert args[0] == "Skryb"
        message = args[1].lower()
        assert "speech" in message or "empty" in message

    def test_should_not_flip_icon_to_error_on_empty_transcription(self):
        # Empty audio / no speech is not a hard error - the engine
        # ran successfully, it just had nothing to transcribe. The
        # icon must NOT enter the sticky ERROR state for this.
        app, icon = self._build_app_with_icon()
        icon.update_state.reset_mock()

        app._event_bus.publish(
            TranscriptionEmptyEvent(transcription_seconds=0.5)
        )

        for update_call in icon.update_state.call_args_list:
            assert update_call.args[0] != AppState.ERROR


class TestErrorIconStickiness:
    """The tray balloon notification can be silently suppressed by
    Focus Assist, Do Not Disturb, or a hidden notification area. A
    second, persistent visual channel - the tray icon itself - must
    keep the ERROR state visible until the user acknowledges it by
    starting a new recording.
    """

    def _build_app_with_icon(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()
        return app, icon

    def test_should_set_icon_to_error_on_error_event(self):
        # Errors that don't go through ``_report_error`` (e.g.
        # ``_start_recording`` failures, clipboard write errors)
        # publish ``ErrorOccurredEvent`` without ever transitioning
        # the state machine to ERROR. The tray must still flip the
        # icon - relying on the balloon alone is unreliable.
        app, icon = self._build_app_with_icon()
        icon.update_state.reset_mock()

        app._event_bus.publish(ErrorOccurredEvent(error="boom"))

        icon.update_state.assert_any_call(AppState.ERROR, "Error")

    def test_should_keep_error_icon_after_state_reverts_to_idle(self):
        # ``_report_error`` runs ERROR → IDLE back-to-back; without
        # stickiness the red icon would flash for ~10ms and the
        # user would miss it whenever the OS notification was
        # suppressed.
        app, icon = self._build_app_with_icon()
        app._event_bus.publish(ErrorOccurredEvent(error="boom"))
        icon.update_state.reset_mock()

        app._event_bus.publish(
            StateChangedEvent(
                old_state=AppState.ERROR.value,
                new_state=AppState.IDLE.value,
            )
        )

        icon.update_state.assert_not_called()

    def test_should_clear_error_icon_when_user_starts_recording(self):
        # The user pressing the hotkey to start a new recording is
        # an implicit acknowledgement of the previous error. From
        # that point on the icon tracks the state machine again.
        app, icon = self._build_app_with_icon()
        app._event_bus.publish(ErrorOccurredEvent(error="boom"))
        app._event_bus.publish(
            StateChangedEvent(
                old_state=AppState.ERROR.value,
                new_state=AppState.IDLE.value,
            )
        )
        icon.update_state.reset_mock()

        app._event_bus.publish(
            StateChangedEvent(
                old_state=AppState.IDLE.value,
                new_state=AppState.RECORDING.value,
            )
        )

        icon.update_state.assert_called_with(AppState.RECORDING, "Recording...")

    def test_should_still_update_elapsed_provider_while_sticky(self):
        # The icon is decoupled from the state machine while sticky,
        # but the elapsed-time provider must still install on entry
        # to RECORDING and clear on exit - those drive the live
        # tooltip, which is a separate channel from the icon colour.
        app, icon = self._build_app_with_icon()
        app._event_bus.publish(ErrorOccurredEvent(error="boom"))
        icon.set_elapsed_provider.reset_mock()

        app._event_bus.publish(
            StateChangedEvent(
                old_state=AppState.IDLE.value,
                new_state=AppState.RECORDING.value,
            )
        )

        icon.set_elapsed_provider.assert_called_once()
        provider = icon.set_elapsed_provider.call_args.args[0]
        assert callable(provider)


class TestMicrophoneMissingNotification:
    def test_names_the_missing_microphone_and_says_the_default_is_in_use(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._event_bus.publish(MicrophoneMissingEvent(name="Microphone (Maono PD300X Microp"))

        message = icon.notify.call_args.args[1]
        assert "Maono PD300X" in message
        assert "default" in message.lower()
        for update_call in icon.update_state.call_args_list:
            assert update_call.args[0] != AppState.ERROR


class TestMicrophoneNotices:
    def _running_app(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()
        return app, icon

    def _record(self, app) -> None:
        app._event_bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))

    def test_a_silent_microphone_gets_one_balloon_per_recording(self):
        app, icon = self._running_app()
        self._record(app)

        for _ in range(4):
            app._event_bus.publish(MicrophoneSilentEvent(
                device_name="Microphone (Maono PD300X Microphone)", seconds_silent=6.0,
            ))

        icon.notify.assert_called_once_with(
            "Skryb",
            "No audio from Microphone (Maono PD300X Microphone). Check it is "
            "unmuted or pick another microphone from the tray menu.",
        )

    def test_the_next_recording_can_warn_again(self):
        app, icon = self._running_app()
        for _ in range(3):
            self._record(app)
            app._event_bus.publish(MicrophoneSilentEvent(device_name="Windows default"))
            app._event_bus.publish(
                StateChangedEvent(old_state="recording", new_state="idle"),
            )

        assert icon.notify.call_count == 3

    def test_a_preferred_microphone_plugged_in_says_it_will_be_used(self):
        app, icon = self._running_app()

        app._event_bus.publish(MicrophoneConnectedEvent(
            name="Headset (Jabra Evolve 20 MS Stereo)", will_be_used=True,
        ))

        icon.notify.assert_called_once_with(
            "Skryb",
            "Using Headset (Jabra Evolve 20 MS Stereo) for the next dictation.",
        )

    def test_any_other_new_microphone_points_at_the_tray_menu(self):
        app, icon = self._running_app()

        app._event_bus.publish(MicrophoneConnectedEvent(name="Mikrofon (Gerät für Café ✓)"))

        icon.notify.assert_called_once_with(
            "Skryb",
            "New microphone: Mikrofon (Gerät für Café ✓). Choose it from the tray "
            "Microphone menu.",
        )

    def test_nothing_is_shown_before_the_tray_icon_exists(self):
        app = _build_app()

        app._event_bus.publish(MicrophoneConnectedEvent(name="USB Mic"))
        app._event_bus.publish(MicrophoneSilentEvent(device_name="USB Mic"))



class TestNoLastDictation:
    def test_asking_for_the_last_dictation_before_any_shows_a_balloon(self):
        from skryb.core.events import NoLastDictationEvent

        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._event_bus.publish(NoLastDictationEvent())

        icon.notify.assert_called_once_with("Skryb", "No dictation yet.")


class TestRestart:
    def _app(self, call_order: list[str], start_replacement):
        listener = MagicMock()
        listener.stop.side_effect = lambda: call_order.append("cleanup")
        app = SystemTrayApp(
            use_case=MagicMock(),
            state_machine=AppStateMachine(event_bus=SimpleEventBus()),
            event_bus=SimpleEventBus(),
            hotkey_listener=listener,
            tray_factory=lambda **kwargs: MagicMock(),
            start_replacement=start_replacement,
        )
        app._icon = MagicMock()
        return app

    def test_restart_starts_the_replacement_before_cleaning_up_and_exiting(self):
        call_order: list[str] = []
        app = self._app(call_order, lambda: call_order.append("replacement"))
        with (
            patch("skryb.tray.app.threading.Timer"),
            patch(
                "skryb.tray.app._exit_now",
                side_effect=lambda: call_order.append("exit"),
            ),
        ):
            app.restart()

        assert call_order == ["replacement", "cleanup", "exit"]

    def test_a_replacement_that_cannot_start_leaves_this_copy_running(self):
        call_order: list[str] = []

        def cannot_start():
            raise FileNotFoundError("skryb-gui.exe")

        app = self._app(call_order, cannot_start)
        with (
            patch("skryb.tray.app.threading.Timer"),
            patch("skryb.tray.app._exit_now") as exit_now,
            pytest.raises(FileNotFoundError),
        ):
            app.restart()

        exit_now.assert_not_called()
        assert call_order == []


class TestMeetingRecordingOnTheIcon:
    def test_a_meeting_started_before_the_icon_exists_shows_once_it_does(self):
        icons: list = []

        def factory(**kwargs):
            icon = MagicMock()
            icon.run.side_effect = lambda setup: setup(icon)
            icons.append(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.meeting_recording_changed(True)
        app.run()

        icons[0].set_meeting_recording.assert_called_once_with(True)

    def test_starting_and_stopping_a_meeting_marks_and_unmarks_the_icon(self):
        icon = MagicMock()
        icon.run.side_effect = lambda setup: setup(icon)
        app = _build_app(tray_factory=lambda **kwargs: icon)
        app.run()

        app.meeting_recording_changed(True)
        app.meeting_recording_changed(False)

        assert [call.args for call in icon.set_meeting_recording.call_args_list] == [(True,), (False,)]

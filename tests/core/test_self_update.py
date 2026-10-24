from skryb.core.self_update import (
    AFTER_DICTATION,
    CHECKING,
    NOT_PULLED,
    UP_TO_DATE,
    SelfUpdate,
)


class _App:
    """Runs background and Tk-thread work inline, and counts dictations by sleeps."""

    def __init__(
        self,
        *,
        pulled: str | None = None,
        changed: bool = False,
        busy_for_sleeps: int = 0,
    ) -> None:
        self.pulled = pulled
        self.changed = changed
        self.busy_for_sleeps = busy_for_sleeps
        self.pulls = 0
        self.restarts = 0
        self.told: list[str] = []
        self.deferred: list = []
        self.busy_checks_until_dictation: int | None = None

    def updater(self, *, defer_background: bool = False) -> SelfUpdate:
        return SelfUpdate(
            pull=self._pull,
            source_changed=lambda: self.changed,
            is_busy=self._is_busy,
            restart=self._restart,
            tell=self.told.append,
            on_gui_thread=lambda action: action(),
            in_background=self.deferred.append if defer_background else (lambda action: action()),
            sleep=self._sleep,
        )

    def _pull(self) -> str | None:
        self.pulls += 1
        return self.pulled

    def _is_busy(self) -> bool:
        if self.busy_checks_until_dictation is not None:
            self.busy_checks_until_dictation -= 1
            if self.busy_checks_until_dictation == 0:
                self.busy_for_sleeps = 3
        return self.busy_for_sleeps > 0

    def _sleep(self, _seconds: float) -> None:
        self.busy_for_sleeps -= 1

    def _restart(self) -> None:
        assert not self._is_busy(), "restarted in the middle of a dictation"
        self.restarts += 1


class TestClickingTheIconAgain:
    def test_with_nothing_new_it_says_the_app_is_already_up_to_date(self) -> None:
        app = _App()

        app.updater().check(asked=True)

        assert app.told == [CHECKING, UP_TO_DATE]
        assert app.restarts == 0

    def test_with_github_out_of_reach_it_says_why_and_keeps_running(self) -> None:
        app = _App(pulled="GitHub didn't answer in time")

        app.updater().check(asked=True)

        assert app.told[-1] == NOT_PULLED.format(reason="GitHub didn't answer in time")
        assert app.restarts == 0

    def test_new_code_restarts_the_app_straight_away_when_nothing_is_being_dictated(
        self,
    ) -> None:
        app = _App(changed=True)

        app.updater().check(asked=True)

        assert app.restarts == 1
        assert app.told == [CHECKING]

    def test_code_edited_in_this_folder_restarts_the_app_even_off_main(self) -> None:
        app = _App(pulled="this folder is on fix/pill-ünïcode, not main", changed=True)

        app.updater().check(asked=True)

        assert app.restarts == 1

    def test_new_code_waits_for_the_dictation_in_progress_and_says_so(self) -> None:
        app = _App(changed=True, busy_for_sleeps=4)

        app.updater().check(asked=True)

        assert app.told == [CHECKING, AFTER_DICTATION]
        assert app.busy_for_sleeps == 0
        assert app.restarts == 1

    def test_a_dictation_starting_just_before_the_restart_postpones_it_until_it_ends(
        self,
    ) -> None:
        app = _App(changed=True)
        # Idle on the background look, dictating by the time the Tk thread runs.
        app.busy_checks_until_dictation = 2

        app.updater().check(asked=True)

        assert app.restarts == 1
        assert app.busy_for_sleeps == 0

    def test_a_second_click_during_a_check_is_ignored(self) -> None:
        app = _App()
        updater = app.updater(defer_background=True)

        updater.check(asked=True)
        updater.check(asked=True)
        for action in app.deferred:
            action()

        assert app.pulls == 1

    def test_the_next_click_after_a_check_finishes_checks_again(self) -> None:
        app = _App()
        updater = app.updater()

        for _ in range(3):
            updater.check(asked=True)

        assert app.pulls == 3
        assert app.told.count(UP_TO_DATE) == 3


class TestTheStartupCheck:
    def test_it_stays_quiet_when_nothing_changed(self) -> None:
        app = _App()

        app.updater().check(asked=False)

        assert app.told == []
        assert app.restarts == 0

    def test_it_stays_quiet_when_it_couldnt_pull(self) -> None:
        app = _App(pulled="this folder has uncommitted changes")

        app.updater().check(asked=False)

        assert app.told == []

    def test_it_restarts_onto_code_merged_while_the_app_was_closed(self) -> None:
        app = _App(changed=True)

        app.updater().check(asked=False)

        assert app.restarts == 1

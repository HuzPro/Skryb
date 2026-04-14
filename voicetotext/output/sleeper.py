import time
from abc import ABC, abstractmethod


class Sleeper(ABC):

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        pass


class TimeSleeper(Sleeper):

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

import logging
import typing as t
from gamechangerml.api.utils import processmanager

logger = logging.getLogger("gamechanger")


class StatusUpdater:
    def __init__(self, process_key: str, nsteps: int) -> t.Iterable:
        self.key = process_key
        self.current_step = 0
        self.nsteps = nsteps
        self.last_message = None

    def next_step(self, message=""):
        self.last_message = message
        try:
            processmanager.update_status(
                self.key,
                progress=self.current_step,
                total=self.nsteps,
                message=message,
            )
        except Exception as e:
            logger.warn(
                f"StatusUpdater {self.key} failed to update status: {message} \n{e}"
            )

    def current(self):
        return {
            "key": self.key,
            "current_step": self.current_step,
            "nsteps": self.nsteps,
            "last_message": self.last_message,
        }

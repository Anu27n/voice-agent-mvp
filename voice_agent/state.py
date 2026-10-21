"""Call state machine: IDLE -> LISTENING -> THINKING -> SPEAKING -> LISTENING ... -> ENDED."""

from __future__ import annotations

import enum
from typing import Callable


class CallState(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ENDED = "ended"


# Any state can end. THINKING -> LISTENING covers a reply cancelled before it made
# a sound (barge-in while thinking, or nothing to say); SPEAKING -> LISTENING is
# both "reply finished playing" and barge-in.
TRANSITIONS: dict[CallState, frozenset[CallState]] = {
    CallState.IDLE: frozenset({CallState.LISTENING, CallState.ENDED}),
    CallState.LISTENING: frozenset({CallState.THINKING, CallState.ENDED}),
    CallState.THINKING: frozenset({CallState.SPEAKING, CallState.LISTENING, CallState.ENDED}),
    CallState.SPEAKING: frozenset({CallState.LISTENING, CallState.ENDED}),
    CallState.ENDED: frozenset(),
}


class InvalidTransition(Exception):
    def __init__(self, current: CallState, target: CallState) -> None:
        super().__init__(f"cannot go from {current.value} to {target.value}")
        self.current = current
        self.target = target


class StateMachine:
    def __init__(self, on_change: Callable[[CallState, CallState], None] | None = None) -> None:
        self.state = CallState.IDLE
        self._on_change = on_change

    def can(self, target: CallState) -> bool:
        return target is self.state or target in TRANSITIONS[self.state]

    def transition(self, target: CallState) -> bool:
        """Move to ``target``. Returns False if already there; raises if not allowed."""
        if target is self.state:
            return False
        if target not in TRANSITIONS[self.state]:
            raise InvalidTransition(self.state, target)
        previous, self.state = self.state, target
        if self._on_change:
            self._on_change(previous, target)
        return True

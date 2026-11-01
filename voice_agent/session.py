"""Session manager: creates sessions, hands each one to exactly one WebSocket,
and forgets it once the call has ended (ended sessions are served from the DB)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .agent import Conversation
from .config import Settings
from .state import CallState, StateMachine
from .store import Store, utcnow

CONNECT_GRACE_S = 60.0  # a session nobody connects to within this window is dropped


@dataclass
class Session:
    id: str
    started_at: str
    created_monotonic: float
    timezone: str | None = None  # the caller's IANA time zone, if the client sent one
    hangup_requested: bool = False  # set by the end_call tool; the call ends after the reply plays
    conversation: Conversation = field(default_factory=Conversation)
    machine: StateMachine = field(default_factory=StateMachine)
    connected: bool = False
    _turns: int = 0

    @property
    def state(self) -> CallState:
        return self.machine.state

    def next_turn_id(self) -> int:
        self._turns += 1
        return self._turns


class SessionManager:
    def __init__(self, store: Store, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._sessions: dict[str, Session] = {}

    async def create(self, timezone: str | None = None) -> Session:
        await self._expire_unclaimed()
        session = Session(
            id=uuid.uuid4().hex, started_at=utcnow(), created_monotonic=time.monotonic(), timezone=timezone
        )
        config = {**self._settings.public_config(), "caller_timezone": timezone}
        await self._store.create_session(session.id, session.started_at, config)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def claim(self, session_id: str) -> Session | None:
        """Attach a WebSocket to a session. Each session accepts one connection."""
        session = self._sessions.get(session_id)
        if session is None or session.connected or session.state is not CallState.IDLE:
            return None
        session.connected = True
        return session

    def release(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @property
    def active(self) -> int:
        return len(self._sessions)

    async def _expire_unclaimed(self) -> None:
        now = time.monotonic()
        for session in list(self._sessions.values()):
            if not session.connected and now - session.created_monotonic > CONNECT_GRACE_S:
                self._sessions.pop(session.id, None)
                await self._store.end_session(session.id, "never_connected")

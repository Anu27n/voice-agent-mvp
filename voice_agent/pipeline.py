"""The per-call voice pipeline (one instance per WebSocket connection).

    client audio -> VAD -> Utterance (streaming STT) -> transcript
        -> Agent (LLM + tools) -> sentences -> TTS -> Outbox -> client

Everything for one call runs on one event loop, so state is only ever changed
between awaits; after every await the code re-checks the state it depends on.

Barge-in: the VAD keeps running while the agent thinks or speaks, and the
frames of the current speech segment are buffered. When the client reports
that the user talked over playback (``interrupt``), or the user keeps talking
while a reply is still being generated, the reply is cancelled and the buffered
segment becomes the next utterance, so the words that caused the interruption
are not lost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Coroutine, Protocol

from .agent import Agent, AgentRoundEnd, AgentRun, AgentText, AgentToolCall, AgentToolResult
from .audio import pcm16_duration_ms
from .config import Settings
from .metrics import TurnTimer
from .providers.base import LLMError, ProviderError, STTError, STTProvider, TTSError, TTSProvider
from .sentences import SentenceSplitter
from .session import Session
from .speakable import speakable
from .state import CallState
from .store import Store
from .timeline import SpeechTimeline
from .util import aiter_timeout
from .vad import EnergyVAD, VADConfig, VADEvent

log = logging.getLogger(__name__)

FALLBACK_LINE = "Sorry, I didn't catch that, could you repeat?"
SILENCE_GOODBYE = "I haven't heard anything for a while, so I'll end the call here. Goodbye!"
MAX_DURATION_GOODBYE = "We've reached the time limit for this call. Thanks for chatting, goodbye!"
PLAYOUT_MARGIN_S = 0.25  # client-side jitter buffer and scheduling slack
GOODBYE_TIMEOUT_S = 15.0
MAX_TYPED_CHARS = 2000
WATCHDOG_INTERVAL_S = 0.2


class Transport(Protocol):
    async def send_text(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def close(self, code: int = 1000) -> None: ...


def pack_audio(turn_id: int, pcm: bytes) -> bytes:
    """agent_audio frame: uint32 little-endian turn_id, then PCM16LE mono."""
    return struct.pack("<I", turn_id) + pcm


class Outbox:
    """The only writer to a call's WebSocket. Keeps events and audio in order and
    lets a barge-in drop audio that has not gone out yet."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._items: deque[tuple[str, Any, int | None]] = deque()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closing = False
        self._dead = False

    @property
    def open(self) -> bool:
        return not (self._dead or self._closing)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def event(self, kind: str, **payload: Any) -> None:
        if self.open:
            self._items.append(("text", json.dumps({"type": kind, **payload}), None))
            self._wakeup.set()

    def audio(self, turn_id: int, pcm: bytes) -> None:
        if self.open:
            self._items.append(("bytes", pack_audio(turn_id, pcm), turn_id))
            self._wakeup.set()

    def drop_audio(self, turn_id: int) -> None:
        self._items = deque(item for item in self._items if not (item[0] == "bytes" and item[2] == turn_id))

    def mark_disconnected(self) -> None:
        self._dead = True
        self._items.clear()
        self._wakeup.set()

    async def close(self, timeout_s: float = 2.0) -> None:
        self._closing = True
        self._wakeup.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout_s)
            except TimeoutError:
                pass
        self._dead = True
        try:
            await self._transport.close()
        except Exception:
            pass

    async def _run(self) -> None:
        while True:
            while not self._items:
                if self._closing or self._dead:
                    return
                self._wakeup.clear()
                await self._wakeup.wait()
            kind, data, _ = self._items.popleft()
            try:
                if kind == "text":
                    await self._transport.send_text(data)
                else:
                    await self._transport.send_bytes(data)
            except Exception:
                self.mark_disconnected()
                return


class Utterance:
    """One VAD speech segment, streamed to STT by a background task."""

    def __init__(
        self,
        stt: STTProvider,
        sample_rate: int,
        on_partial: Callable[[str], Awaitable[None]],
        timeout_s: float,
    ) -> None:
        self.timer = TurnTimer()
        self._timeout_s = timeout_s
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task = asyncio.create_task(self._run(stt, sample_rate, on_partial))

    def feed(self, frame: bytes) -> None:
        self._queue.put_nowait(frame)

    async def _run(self, stt: STTProvider, sample_rate: int, on_partial: Callable[[str], Awaitable[None]]) -> str:
        stream = await stt.open_stream(sample_rate=sample_rate, on_partial=on_partial)
        try:
            while (frame := await self._queue.get()) is not None:
                await stream.send(frame)
            return await stream.finish()
        finally:
            await stream.aclose()

    async def finish(self) -> str:
        self._queue.put_nowait(None)
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._task
        except TimeoutError as exc:
            raise STTError("stt_timeout", "Speech recognition took too long.") from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise STTError("stt_failed", f"{type(exc).__name__}: {exc}") from exc

    async def discard(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


@dataclass
class Reply:
    turn_id: int
    timer: TurnTimer
    timeline: SpeechTimeline = field(default_factory=SpeechTimeline)
    task: asyncio.Task | None = None
    run: AgentRun | None = None
    audio_started: bool = False
    first_audio_at: float | None = None
    playout_until: float = 0.0  # when the client should finish playing what was sent
    done: bool = False


class VoicePipeline:
    def __init__(
        self,
        session: Session,
        settings: Settings,
        *,
        stt: STTProvider,
        agent: Agent,
        tts: TTSProvider,
        store: Store,
        transport: Transport,
        clock: Callable[[], float] = time.perf_counter,
        on_ended: Callable[[], None] | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self._on_ended = on_ended
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._store = store
        self._clock = clock
        self._outbox = Outbox(transport)
        self._vad = EnergyVAD(
            VADConfig(
                frame_ms=settings.frame_ms,
                start_ms=settings.vad_start_ms,
                silence_ms=settings.vad_silence_ms,
                max_utterance_ms=settings.vad_max_utterance_ms,
                threshold_db=settings.vad_threshold_db,
                margin_db=settings.vad_margin_db,
            )
        )
        self._frame_bytes = settings.frame_bytes
        self._rx = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=max(1, settings.vad_preroll_ms // settings.frame_ms))
        self._segment: list[bytes] = []  # frames of the current speech segment, onset included
        self._utterance: Utterance | None = None
        self._finalizing = 0
        self._reply: Reply | None = None
        self._tasks: set[asyncio.Task] = set()  # cancelled when the call ends
        self._writes: set[asyncio.Task] = set()  # awaited when the call ends
        self._watchdog: asyncio.Task | None = None
        self._started_at = 0.0
        self._last_voiced_at = 0.0
        self._last_activity = 0.0
        self._ending = False
        self.ended = asyncio.Event()

    @property
    def state(self) -> CallState:
        return self.session.state

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        now = self._clock()
        self._started_at = self._last_activity = now
        self._outbox.start()
        self._set_state(CallState.LISTENING)
        self._watchdog = asyncio.create_task(self._watch())

    async def on_disconnect(self) -> None:
        self._outbox.mark_disconnected()
        await self.end("disconnected")
        await self.ended.wait()

    async def end(self, reason: str, goodbye: str | None = None) -> None:
        if self._ending:
            return
        self._ending = True
        log.info("[%s] ending call: %s", self.session.id[:8], reason)
        if self._watchdog is not None and self._watchdog is not asyncio.current_task():
            self._watchdog.cancel()
        if self._reply is not None and not self._reply.done:
            await self._cancel_reply(self._reply, None)
        if self._utterance is not None:
            utterance, self._utterance = self._utterance, None
            await utterance.discard()

        if goodbye and self._outbox.open:
            reply = Reply(self.session.next_turn_id(), TurnTimer())
            try:
                async with asyncio.timeout(GOODBYE_TIMEOUT_S):
                    if self.state is CallState.LISTENING:
                        self._set_state(CallState.THINKING)
                    await self._speak_lines(reply, [goodbye])
                    await self._wait_playout(reply)
            except TimeoutError:
                log.warning("[%s] goodbye timed out", self.session.id[:8])
            await self._persist_reply(reply, goodbye, interrupted=False)

        if self.state is not CallState.ENDED:
            self.session.machine.transition(CallState.ENDED)
        current = asyncio.current_task()
        pending = [t for t in self._tasks if t is not current]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*self._writes, return_exceptions=True)
        await self._store.end_session(self.session.id, reason)
        if self._on_ended is not None:
            self._on_ended()
        # Only now tell the client: everything about the call is saved, and it no
        # longer counts as active, by the time call_ended arrives.
        self._outbox.event("call_ended", reason=reason)
        await self._outbox.close()
        self.ended.set()

    async def _watch(self) -> None:
        settings = self.settings
        while not self._ending:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            now = self._clock()
            if now - self._started_at >= settings.max_call_duration_s:
                self._spawn(self.end("max_duration", goodbye=MAX_DURATION_GOODBYE))
                return
            idle = (
                self.state is CallState.LISTENING
                and self._reply is None
                and self._utterance is None
                and self._finalizing == 0
                and not self._vad.in_speech
            )
            if idle and now - self._last_activity >= settings.silence_timeout_s:
                self._spawn(self.end("silence_timeout", goodbye=SILENCE_GOODBYE))
                return

    # ------------------------------------------------------------- inbound

    async def on_audio(self, data: bytes) -> None:
        if self._ending:
            return
        self._rx.extend(data)
        while len(self._rx) >= self._frame_bytes:
            frame = bytes(self._rx[: self._frame_bytes])
            del self._rx[: self._frame_bytes]
            await self._process_frame(frame)

    async def on_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
            kind = message.get("type")
        except (ValueError, AttributeError):
            self._outbox.event("error", code="bad_message", message="Expected a JSON object with a 'type' field.")
            return
        if kind == "interrupt":
            played = message.get("played_ms")
            played_ms = float(played) if isinstance(played, (int, float)) and not isinstance(played, bool) else None
            turn_id = message.get("turn_id")
            await self.interrupt(turn_id if isinstance(turn_id, int) else None, played_ms)
        elif kind == "user_text":
            await self.user_text(message.get("text"))
        elif kind == "end_call":
            await self.end("user_ended")
        elif kind == "ping":
            self._outbox.event("pong")
        else:
            self._outbox.event("error", code="unknown_event", message=f"Unknown event type {kind!r}.")

    async def user_text(self, text: Any) -> None:
        """A typed message: handled like a spoken turn, minus VAD and STT."""
        if self._ending:
            return
        if not isinstance(text, str) or not text.strip():
            self._outbox.event("error", code="bad_message", message="user_text needs non-empty 'text'.")
            return
        reply = self._reply
        if reply is not None and not reply.done:
            await self._cancel_reply(reply, None)  # typing over the agent works like a barge-in
            self._set_state(CallState.LISTENING, announce=True)
        if self.state is not CallState.LISTENING:
            return
        now = self._clock()
        self._last_activity = now
        self._commit_user_turn(text.strip()[:MAX_TYPED_CHARS], TurnTimer(stt_final=now), source="text")

    async def interrupt(self, turn_id: int | None = None, played_ms: float | None = None) -> None:
        """Barge-in reported by the client: stop the reply, keep what was heard."""
        if self._ending:
            return
        reply = self._reply
        if reply is None or reply.done or (turn_id is not None and turn_id != reply.turn_id):
            self._outbox.event("state", state=self.state.value)  # nothing to cancel; just acknowledge
            return
        log.info("[%s] barge-in on turn %d", self.session.id[:8], reply.turn_id)
        await self._cancel_reply(reply, played_ms)
        self._set_state(CallState.LISTENING, announce=True)

    async def _process_frame(self, frame: bytes) -> None:
        now = self._clock()
        event = self._vad.process(frame)
        if self._vad.last_frame_voiced:
            self._last_voiced_at = now
        self._preroll.append(frame)

        if event is VADEvent.SPEECH_START:
            self._segment = list(self._preroll)
            self._last_activity = now
            if self.state is CallState.LISTENING and self._utterance is None:
                self._open_utterance()
        elif self._vad.in_speech:
            self._segment.append(frame)
            self._last_activity = now
            if self._utterance is not None:
                self._utterance.feed(frame)

        if event is VADEvent.SPEECH_END:
            self._segment = []
            utterance, self._utterance = self._utterance, None
            if utterance is not None:
                long_enough = self._vad.last_segment_voiced_ms >= self.settings.vad_min_utterance_ms
                if self.state is CallState.LISTENING and long_enough:
                    utterance.timer.speech_end = self._last_voiced_at
                    utterance.timer.vad_end = now
                    self._finalizing += 1
                    self._spawn(self._finish_utterance(utterance))
                else:
                    # Too short to be speech, or it happened while the agent was
                    # talking without the client reporting a barge-in (echo).
                    self._spawn(utterance.discard())

        reply = self._reply
        if (
            self.state is CallState.THINKING
            and reply is not None
            and not reply.audio_started
            and self._vad.in_speech
            and self._vad.speech_ms >= self.settings.barge_in_min_ms
        ):
            # The user kept talking while the reply was still being generated.
            log.info("[%s] user kept talking; cancelling turn %d", self.session.id[:8], reply.turn_id)
            await self._cancel_reply(reply, 0.0)
            self._set_state(CallState.LISTENING, announce=True)

    def _open_utterance(self) -> None:
        utterance = Utterance(self._stt, self.settings.input_sample_rate, self._on_partial, self.settings.stt_timeout_s)
        for frame in self._segment:
            utterance.feed(frame)
        self._utterance = utterance

    async def _on_partial(self, text: str) -> None:
        if not self._ending:
            self._outbox.event("transcript_partial", text=text)

    async def _finish_utterance(self, utterance: Utterance) -> None:
        try:
            try:
                text = (await utterance.finish()).strip()
            except STTError as exc:
                if self.state is CallState.LISTENING and self._reply is None and not self._ending:
                    turn_id = self.session.next_turn_id()
                    await self._report_error(turn_id, exc)
                    self._start_reply(turn_id, utterance.timer, fixed_text=FALLBACK_LINE)
                else:
                    await self._report_error(None, exc)
                return
            utterance.timer.stt_final = self._clock()
            if not text:
                log.info("[%s] empty transcript ignored", self.session.id[:8])
                return
            if self.state is not CallState.LISTENING or self._reply is not None or self._ending:
                log.info("[%s] transcript dropped while %s: %r", self.session.id[:8], self.state.value, text)
                return
            self._commit_user_turn(text, utterance.timer, source="voice")
        finally:
            self._finalizing -= 1

    def _commit_user_turn(self, text: str, timer: TurnTimer, *, source: str) -> None:
        turn_id = self.session.next_turn_id()
        self._outbox.event("transcript_final", text=text, turn_id=turn_id, source=source)
        self.session.conversation.add_user(text)
        self._start_reply(turn_id, timer)
        self._persist(self._store.add_turn(self.session.id, turn_id, "user", text))

    # ------------------------------------------------------------- replies

    def _start_reply(self, turn_id: int, timer: TurnTimer, fixed_text: str | None = None) -> None:
        reply = Reply(turn_id, timer)
        self._reply = reply
        self._set_state(CallState.THINKING)
        reply.task = asyncio.create_task(self._run_reply(reply, fixed_text))
        # A finished reply may still be saving its turn; the call's end waits for it.
        self._writes.add(reply.task)
        reply.task.add_done_callback(self._writes.discard)

    async def _run_reply(self, reply: Reply, fixed_text: str | None) -> None:
        try:
            if fixed_text is not None:
                await self._speak_lines(reply, [fixed_text])
            else:
                await self._speak_agent(reply)
            await self._wait_playout(reply)
        except Exception:
            log.exception("[%s] reply %d failed", self.session.id[:8], reply.turn_id)
            self._outbox.event("error", code="internal_error", message="Something went wrong with that reply.")
        reply.done = True
        if reply.run is not None:
            reply.run.commit_completed()
        if self._reply is reply:
            self._reply = None
        hang_up = self.session.hangup_requested and not self._ending
        if self.state in (CallState.THINKING, CallState.SPEAKING) and not self._ending and not hang_up:
            self._set_state(CallState.LISTENING)
        await self._persist_reply(reply, reply.timeline.full_text(), interrupted=False)
        if hang_up:
            # The agent said goodbye and asked to hang up; its goodbye has now played.
            self._spawn(self.end("agent_ended"))

    async def _speak_agent(self, reply: Reply) -> None:
        run = self._agent.start(self.session.conversation)
        reply.run = run
        try:
            await self._stream_reply(reply, run)
        except (LLMError, TTSError) as exc:
            await self._report_error(reply.turn_id, exc)
            # Keep whatever the caller already heard; say the fallback line if nothing.
            self._commit_heard(reply)
            if not reply.audio_started:
                await self._speak_lines(reply, [FALLBACK_LINE])

    async def _stream_reply(self, reply: Reply, run: AgentRun) -> None:
        sentences: asyncio.Queue[tuple[str, int] | None] = asyncio.Queue()
        producer = asyncio.create_task(self._produce_sentences(reply, run, sentences))
        try:
            while (item := await sentences.get()) is not None:
                await self._speak_sentence(reply, *item)
            await producer  # re-raises an LLM error
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _produce_sentences(self, reply: Reply, run: AgentRun, out: asyncio.Queue) -> None:
        splitter = SentenceSplitter()

        def emit(sentences: list[str]) -> None:
            for sentence in sentences:
                spoken = speakable(sentence)
                if spoken:  # a sentence of pure markup has nothing to say
                    out.put_nowait((spoken, run.round_index))

        try:
            async for event in run.events():
                if isinstance(event, AgentText):
                    if reply.timer.llm_first_token is None:
                        reply.timer.llm_first_token = self._clock()
                    emit(splitter.push(event.text))
                elif isinstance(event, AgentRoundEnd):
                    emit(splitter.flush())
                elif isinstance(event, AgentToolCall):
                    call = event.call
                    self._outbox.event("tool_call", turn_id=reply.turn_id, name=call.name, arguments=call.arguments)
                elif isinstance(event, AgentToolResult):
                    result = event.result
                    self._outbox.event(
                        "tool_result",
                        turn_id=reply.turn_id,
                        name=result.name,
                        is_error=result.is_error,
                        result=json.loads(result.content),
                    )
            emit(splitter.flush())
        finally:
            out.put_nowait(None)

    async def _speak_lines(self, reply: Reply, lines: list[str]) -> None:
        for line in lines:
            try:
                await self._speak_sentence(reply, line, 0)
            except TTSError as exc:
                await self._report_error(reply.turn_id, exc)  # the caption still reached the client
                return

    async def _speak_sentence(self, reply: Reply, text: str, round_index: int) -> None:
        if not self._outbox.open:
            return
        segment = reply.timeline.begin(text, round_index)
        self._outbox.event("agent_text", text=text, turn_id=reply.turn_id)
        timeout = self.settings.tts_timeout_s
        try:
            async for chunk in aiter_timeout(self._tts.synthesize(text), timeout, timeout):
                if not chunk:
                    continue
                if not self._outbox.open:
                    return
                now = self._clock()
                if not reply.audio_started:
                    reply.audio_started = True
                    reply.first_audio_at = reply.timer.tts_first_audio = now
                    if self.state is CallState.THINKING:
                        self._set_state(CallState.SPEAKING)
                duration_ms = pcm16_duration_ms(len(chunk), self._tts.sample_rate)
                reply.timeline.add_audio(segment, duration_ms)
                reply.playout_until = max(now, reply.playout_until) + duration_ms / 1000.0
                self._outbox.audio(reply.turn_id, chunk)
            segment.complete = True
        except TimeoutError as exc:
            raise TTSError("tts_timeout", "Speech synthesis took too long.") from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise TTSError("tts_failed", f"{type(exc).__name__}: {exc}") from exc

    async def _wait_playout(self, reply: Reply) -> None:
        if not reply.audio_started or not self._outbox.open:
            return
        delay = reply.playout_until + PLAYOUT_MARGIN_S - self._clock()
        if delay > 0:
            await asyncio.sleep(delay)

    async def _cancel_reply(self, reply: Reply, played_ms: float | None) -> None:
        if reply.task is not None and not reply.task.done():
            reply.task.cancel()
            await asyncio.gather(reply.task, return_exceptions=True)
        reply.done = True
        self.session.hangup_requested = False  # interrupted the goodbye: the caller has more to say
        self._outbox.drop_audio(reply.turn_id)
        heard_ms = self._heard_ms(reply, played_ms)
        self._commit_heard(reply, heard_ms)
        if self._reply is reply:
            self._reply = None
        await self._persist_reply(reply, reply.timeline.spoken_text(heard_ms), interrupted=True)

    def _heard_ms(self, reply: Reply, played_ms: float | None) -> float:
        if played_ms is not None:
            heard = played_ms
        elif reply.first_audio_at is not None:
            heard = (self._clock() - reply.first_audio_at) * 1000.0
        else:
            heard = 0.0
        return max(0.0, min(heard, reply.timeline.total_ms))

    def _commit_heard(self, reply: Reply, heard_ms: float | None = None) -> None:
        """Commit only the heard part of the reply's unfinished LLM round."""
        run = reply.run
        if run is None:
            return
        heard = reply.timeline.total_ms if heard_ms is None else heard_ms
        spoken = "" if run.round_committed else reply.timeline.spoken_text(heard, rounds={run.round_index})
        run.commit_interrupted(spoken)

    # ------------------------------------------------------------- helpers

    def _set_state(self, target: CallState, *, announce: bool = False) -> None:
        changed = self.session.machine.transition(target)
        if changed or announce:
            self._outbox.event("state", state=target.value)
        if changed and target is CallState.LISTENING:
            self._last_activity = self._clock()
            if self._vad.in_speech and self._utterance is None and self._segment:
                self._open_utterance()  # the speech that interrupted becomes the next turn

    async def _persist_reply(self, reply: Reply, text: str, *, interrupted: bool) -> None:
        tool_calls = reply.run.tool_log if reply.run is not None else []
        metrics = reply.timer.as_metrics()
        log.info("[%s] turn %d metrics %s", self.session.id[:8], reply.turn_id, metrics)
        await self._store.add_turn(
            self.session.id, reply.turn_id, "agent", text, interrupted=interrupted, tool_calls=tool_calls
        )
        await self._store.add_metrics(self.session.id, reply.turn_id, metrics)
        # Announced only once saved, so a client that reads the session back sees it.
        self._outbox.event("metrics", turn_id=reply.turn_id, **metrics)

    async def _report_error(self, turn_id: int | None, exc: ProviderError) -> None:
        log.warning("[%s] %s error %s: %s", self.session.id[:8], exc.stage, exc.code, exc.message)
        self._outbox.event("error", code=exc.code, message=exc.message, stage=exc.stage)
        await self._store.add_error(self.session.id, turn_id, exc.stage, exc.code, exc.message)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _persist(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("[%s] background task failed", self.session.id[:8], exc_info=task.exception())

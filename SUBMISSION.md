# Submission notes

Setup and running are in the [README](README.md); the full design (flow, event
schema, interfaces) is in [docs/DESIGN.md](docs/DESIGN.md). This note covers the
main design choices and what I could not complete.

## Design choices

- **One asyncio pipeline per call.** FastAPI serves the page, `POST /sessions` and
  a WebSocket. Each call runs VAD -> streaming STT -> agent (LLM + tools) ->
  sentence-by-sentence TTS on one event loop, so there are no threads or locks in
  the hot path.
- **Every stage streams.** STT emits partials for live captions. LLM tokens are
  split into sentences, and each sentence goes to TTS immediately, so the agent
  starts speaking before the reply is finished.
- **Providers behind small interfaces.** STT, LLM and TTS are abstract classes
  with Deepgram, Claude and ElevenLabs adapters plus offline mocks. Swapping one is
  configuration only (tested). With no API keys, the app runs in a demo mode built
  on the mocks.
- **Pinned audio format.** 16 kHz PCM16 mono in 20 ms frames from the browser
  (an AudioWorklet resamples); 24 kHz PCM16 back.
- **Energy VAD in audio time.** It has an adaptive noise floor, 100 ms to start,
  600 ms of silence to end, and a 300 ms pre-roll. Short or empty segments never
  reach the LLM. Because it counts frames rather than wall-clock time, recorded
  audio tests behave like live audio.
- **Barge-in.** The browser stops playback after 160 ms of speech. Detection
  learns the agent's echo level, so speakers don't trigger it. The server cancels
  the LLM and TTS requests and turns the interrupting speech into the next turn.
  Only the words the caller actually heard are kept in the history.
- **Append-only conversation history.** Earlier turns are never edited. That
  keeps Claude's thinking blocks valid and lets prompt caching work on every turn.
  The model is `claude-opus-5-5` at low effort, and `claude-haiku-5-5` is a
  one-line switch for lower latency.
- **Latency measured from day one.** Four back-to-back stages (VAD end, STT
  final, LLM first token, TTS first audio) add up to the total. They're stored per
  turn in SQLite and shown live in the UI. `python -m voice_agent.benchmark
  --turns 20` produces the median/p90 report.
- **Failures degrade instead of dropping the call.** STT, LLM and TTS errors and
  timeouts speak a fallback line and the call continues. Silence and max-duration
  timeouts say goodbye, and a dropped socket still leaves the saved transcript.
- **Testing.** 159 automated tests, including an end-to-end test that streams a
  recorded WAV over a real WebSocket. CI runs lint and tests on Linux and Windows,
  and builds and smoke-tests the Docker image.

**Beyond the brief:**
- weather and calculator tools, and time answered in the caller's time zone;
- the agent hangs up when you say goodbye;
- typed messages and text-only calls;
- a call history page with Markdown transcripts;
- Prometheus metrics, an optional access code, and a limit on concurrent calls;
- a Docker image and a Render blueprint.

## What I could not complete

- **Demo video.** I couldn't record the 2–3 minute video. I'm happy to record one
  if you'd like to see it.
- **Testing with live provider APIs.** I didn't have Deepgram and Anthropic API
  keys while building this. The adapters are tested against faithful fakes (a fake
  Deepgram WebSocket server, a fake Anthropic client, mocked HTTP), and the full
  pipeline runs end to end with the offline providers. With keys,
  `pytest -m live` and the benchmark check the real services.
- **Latency report from 20 real turns.** Because of the above, I don't have
  real-provider numbers yet. The benchmark produces the report in one command once
  keys are set.
- **Hosted deployment.** The app is ready for Render (Dockerfile, `render.yaml`,
  access code, call limits), but I haven't deployed it.
- **Known limitations.**
  - VAD and barge-in are energy-based; a neural VAD such as Silero would hold up
    better in noisy rooms.
  - The return to listening after the agent speaks is estimated from the audio
    sent, rather than reported by the browser.
  - It is a single process with SQLite.

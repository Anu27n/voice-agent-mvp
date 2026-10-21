from conftest import FIXTURES, silence, speech
from voice_agent.audio import iter_frames, read_wav
from voice_agent.vad import EnergyVAD, VADConfig, VADEvent


def run(vad, frames):
    return [(i, e) for i, frame in enumerate(frames) if (e := vad.process(frame))]


def test_silence_produces_no_events():
    assert run(EnergyVAD(), silence(2000)) == []


def test_utterance_start_and_end():
    vad = EnergyVAD(VADConfig(start_ms=100, silence_ms=600))
    events = run(vad, silence(400) + speech(800) + silence(700))
    assert [e for _, e in events] == [VADEvent.SPEECH_START, VADEvent.SPEECH_END]
    start, end = events[0][0], events[1][0]
    assert start == 20 + 4  # 400 ms of silence, then the 5th voiced frame (100 ms)
    assert end == 20 + 40 + 29  # 600 ms (30 frames) of trailing silence
    assert vad.last_segment_voiced_ms == 800


def test_short_clicks_never_start_an_utterance():
    frames = silence(300) + speech(60) + silence(300) + speech(80) + silence(300)
    assert run(EnergyVAD(VADConfig(start_ms=100)), frames) == []


def test_pauses_shorter_than_the_silence_threshold_do_not_end_speech():
    vad = EnergyVAD(VADConfig(silence_ms=600))
    events = run(vad, speech(500) + silence(400) + speech(500) + silence(700))
    assert [e for _, e in events] == [VADEvent.SPEECH_START, VADEvent.SPEECH_END]


def test_silence_threshold_is_configurable():
    frames = speech(500) + silence(400) + speech(500) + silence(700)
    events = run(EnergyVAD(VADConfig(silence_ms=300)), frames)
    assert [e for _, e in events].count(VADEvent.SPEECH_END) == 2


def test_quiet_background_noise_is_not_speech():
    # Steady noise well under the absolute floor never triggers.
    assert run(EnergyVAD(), silence(3000)) == []


def test_force_ends_a_segment_that_never_goes_quiet():
    vad = EnergyVAD(VADConfig(max_utterance_ms=1000))
    events = run(vad, speech(1500))
    # Ends at the cap, then (still voiced) starts a new segment.
    assert [(i, e) for i, e in events] == [
        (4, VADEvent.SPEECH_START),
        (49, VADEvent.SPEECH_END),  # 50 frames = 1000 ms
        (54, VADEvent.SPEECH_START),
    ]


def test_recorded_fixture_has_two_utterances_and_the_click_is_ignored():
    pcm, rate = read_wav(FIXTURES / "two_turns.wav")
    vad = EnergyVAD()
    events = [e for frame in iter_frames(pcm, rate, 20) if (e := vad.process(frame))]
    assert events == [VADEvent.SPEECH_START, VADEvent.SPEECH_END] * 2

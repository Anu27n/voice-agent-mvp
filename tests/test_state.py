import pytest

from voice_agent.state import TRANSITIONS, CallState, InvalidTransition, StateMachine


def test_starts_idle():
    assert StateMachine().state is CallState.IDLE


def test_full_call_cycle():
    seen = []
    machine = StateMachine(on_change=lambda old, new: seen.append((old.value, new.value)))
    for state in ("listening", "thinking", "speaking", "listening", "thinking", "speaking", "ended"):
        assert machine.transition(CallState(state)) is True
    assert seen[0] == ("idle", "listening")
    assert seen[-1] == ("speaking", "ended")


def test_barge_in_paths():
    machine = StateMachine()
    machine.transition(CallState.LISTENING)
    machine.transition(CallState.THINKING)
    machine.transition(CallState.LISTENING)  # cancelled before speaking
    machine.transition(CallState.THINKING)
    machine.transition(CallState.SPEAKING)
    machine.transition(CallState.LISTENING)  # interrupted while speaking
    assert machine.state is CallState.LISTENING


@pytest.mark.parametrize("state", list(CallState))
def test_every_live_state_can_end(state):
    if state is CallState.ENDED:
        assert TRANSITIONS[state] == frozenset()
    else:
        assert CallState.ENDED in TRANSITIONS[state]


@pytest.mark.parametrize(
    "path",
    [
        ("thinking",),  # IDLE -> THINKING
        ("listening", "speaking"),  # must think first
        ("listening", "thinking", "speaking", "thinking"),
        ("ended", "listening"),  # ENDED is terminal
    ],
)
def test_invalid_transitions_raise(path):
    machine = StateMachine()
    *ok, bad = [CallState(s) for s in path]
    for state in ok:
        machine.transition(state)
    before = machine.state
    with pytest.raises(InvalidTransition):
        machine.transition(bad)
    assert machine.state is before


def test_same_state_is_a_no_op():
    calls = []
    machine = StateMachine(on_change=lambda *a: calls.append(a))
    machine.transition(CallState.LISTENING)
    assert machine.transition(CallState.LISTENING) is False
    assert len(calls) == 1
    assert machine.can(CallState.LISTENING)

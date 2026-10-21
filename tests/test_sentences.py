from voice_agent.sentences import SentenceSplitter


def stream(text, step=1):
    splitter = SentenceSplitter()
    out = []
    for i in range(0, len(text), step):
        out += splitter.push(text[i : i + step])
    return out + splitter.flush()


def test_emits_each_sentence_as_soon_as_it_ends():
    splitter = SentenceSplitter()
    assert splitter.push("Sure! Let me") == ["Sure!"]
    assert splitter.push(" check that. It") == ["Let me check that."]
    assert splitter.flush() == ["It"]


def test_token_by_token_streaming():
    text = "It's 3 PM. Your order shipped on Friday! Anything else? "
    assert stream(text) == ["It's 3 PM.", "Your order shipped on Friday!", "Anything else?"]


def test_abbreviations_initials_and_decimals_do_not_split():
    text = "Dr. Smith said the fee is 3.50 dollars, e.g. per month. J. Doe agreed."
    assert stream(text, step=3) == ["Dr. Smith said the fee is 3.50 dollars, e.g. per month.", "J. Doe agreed."]


def test_pronoun_i_still_ends_a_sentence():
    assert stream("So do I. Thanks.") == ["So do I.", "Thanks."]


def test_quotes_after_punctuation_stay_with_the_sentence():
    assert stream('She said "hello." Then left.') == ['She said "hello."', "Then left."]


def test_newlines_are_boundaries():
    assert stream("First line\nSecond line") == ["First line", "Second line"]


def test_long_run_on_text_is_broken_at_a_comma():
    splitter = SentenceSplitter(max_chars=40)
    out = splitter.push("this goes on and on without stopping, and it keeps going further")
    assert out == ["this goes on and on without stopping,"]
    assert splitter.flush() == ["and it keeps going further"]


def test_flush_on_empty_buffer():
    assert SentenceSplitter().flush() == []

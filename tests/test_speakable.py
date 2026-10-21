import pytest

from voice_agent.speakable import speakable


@pytest.mark.parametrize(
    "text, spoken",
    [
        ("**Sure!** I can help.", "Sure! I can help."),
        ("That's *really* fast, __very__ fast.", "That's really fast, very fast."),
        ("## Order status", "Order status"),
        ("- Shipped on Friday", "Shipped on Friday"),
        ("2. Arrives Monday", "Arrives Monday"),
        ("> quoted", "quoted"),
        ("See [the tracking page](https://ups.com/track?id=1).", "See the tracking page."),
        ("Visit https://www.example.com/help/orders for more.", "Visit example.com for more."),
        ("Run `pip install` now.", "Run pip install now."),
        ("You are ticket #42.", "You are ticket number 42."),
        ("Great news 🎉🚚!", "Great news!"),
        ("2 * 3 is 6.", "2 times 3 is 6."),
        ("Your order_id is A1001.", "Your order_id is A1001."),  # underscores inside words stay
        ("It's 3:42 PM, about 3.5 hours left.", "It's 3:42 PM, about 3.5 hours left."),
        ("```python", ""),
        ("***", ""),
    ],
)
def test_speakable(text, spoken):
    assert speakable(text) == spoken

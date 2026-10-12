"""The price lambda: fitted once over a whole sequence (offline) or steered online.

Online, lambda is fitted on the first ``startup_tokens`` and then nudged after
every eight-token window so that the slot's saving returns to its target. The
rule compares the cumulative saving and a short exponential average with the
target; inside small bands nothing changes, outside them log(lambda) moves by a
clamped step. The under-compression band is widened early in the sequence by one
window's largest possible swing divided by the windows seen.
"""
from dataclasses import dataclass
import math

from compression_topics.spatial.algorithms.formats import HEADER_BITS, WINDOW

SATURATED_PRICE = 1e12


@dataclass(frozen=True)
class Control:
    startup_tokens: int = 32
    memory_pairs: float = 10.
    recent_deadband: float = .02
    cumulative_deadband: float = .03
    recent_gain: float = 4.
    cumulative_gain: float = 8.
    max_log_step: float = .2
    price_span: float = 1e4

    def __post_init__(self):
        if self.startup_tokens < WINDOW or self.startup_tokens % WINDOW:
            raise ValueError('startup must contain whole eight-token windows')
        values = (self.memory_pairs, self.max_log_step, self.price_span)
        if any(not math.isfinite(x) or x <= 0 for x in values) or self.price_span <= 1:
            raise ValueError('invalid controller memory/bounds')
        if any(not math.isfinite(x) or x < 0 for x in
               (self.recent_deadband, self.cumulative_deadband, self.recent_gain, self.cumulative_gain)):
            raise ValueError('invalid feedback parameters')


def stored_bits(choices):
    return sum(c.bits for c in choices) + HEADER_BITS


def fit_price(table, windows, limit_bits):
    """Smallest price whose choices on ``windows`` fit ``limit_bits``.

    Returns (fitting price, last price above the limit) so a caller can keep the
    less-compressed side of a discrete jump.
    """
    def stored(price):
        return stored_bits(table.choose(w, price) for w in windows)
    low, high = 0., 1e-6
    while stored(high) > limit_bits:
        high *= 4
        if high > SATURATED_PRICE:
            raise ValueError('target cannot be met with the available formats')
    for _ in range(60):
        middle = (low + high) / 2
        if stored(middle) <= limit_bits:
            high = middle
        else:
            low = middle
    exact = abs(stored(high) - limit_bits) <= 1e-8
    return high, (high if exact else low)


def _outside(error, under_band, over_band):
    band = under_band if error >= 0 else over_band
    return math.copysign(max(abs(error) - band, 0.), error)


class Online:
    """Causal price for one layer slot; state survives across cache updates."""

    def __init__(self, saving, control=Control(), minimum_storage=False):
        if not math.isfinite(saving) or not 0 < saving < 1:
            raise ValueError('saving must be between zero and one')
        self.saving, self.control, self.minimum_storage = saving, control, minimum_storage
        self.warmup = control.startup_tokens // WINDOW
        self.alpha = 1 - math.exp(-(WINDOW / 2) / control.memory_pairs)
        self.price = SATURATED_PRICE if minimum_storage else None
        self.anchor = self.initial = None
        self.windows = 0
        self.stored = HEADER_BITS
        self.dense = 0
        self.ema = 1 - saving
        self.hits = 0

    @property
    def started(self):
        return self.price is not None

    def start(self, table):
        """Fit the startup price on the first ``warmup`` windows of ``table``."""
        if not self.minimum_storage:
            limit = len(range(self.warmup)) * table.window_bits * (1 - self.saving)
            _, self.price = fit_price(table, range(self.warmup), limit)
        self.anchor = max(self.price, 1e-12)
        self.initial = self.price

    def decide(self, table, window):
        """Choose one window at the current price, then update the price."""
        price = self.price
        choice = table.choose(window, price)
        self.windows += 1
        n = self.windows
        self.dense += table.window_bits
        previous = self.stored
        self.stored += choice.bits
        fraction = self.stored / self.dense
        increment = (self.stored - previous) / table.window_bits
        target = 1 - self.saving
        if n == self.warmup:
            self.ema = fraction
        elif n > self.warmup:
            self.ema = (1 - self.alpha) * self.ema + self.alpha * increment
            if not self.minimum_storage:
                c = self.control
                slack = (table.choose(window, 0.).bits - table.choose(window, SATURATED_PRICE).bits) / (n * table.window_bits)
                correction = c.recent_gain * _outside(self.ema - target, c.recent_deadband + slack, c.recent_deadband)
                correction += c.cumulative_gain * _outside(fraction - target, c.cumulative_deadband + slack,
                                                           c.cumulative_deadband)
                step = (WINDOW / 2) * max(-c.max_log_step, min(c.max_log_step, correction))
                new = (self.price if self.price > 0 else self.anchor / c.price_span) * math.exp(step)
                self.price = max(self.anchor / c.price_span, min(self.anchor * c.price_span, new))
                self.hits += int(self.price != new)
        return choice, price


def offline_price(table, saving, *, tail_bits=0):
    """One price for every window of the slot, meeting the saving including dense tail tokens."""
    if not math.isfinite(saving) or not 0 < saving < 1:
        raise ValueError('saving must be between zero and one')
    dense = len(table) * table.window_bits + tail_bits
    price, _ = fit_price(table, range(len(table)), dense * (1 - saving) - tail_bits)
    return price

"""Pure projection math for Prep Station.

The planner deliberately lives outside the Anki/PyQt integration module so the
calculation can be exercised with ordinary Python.  A card is considered
``mature`` when its scheduled interval reaches 21 days, matching Anki's stats
and the convention used by Anki Simulator.

The FSRS equations below follow the public FSRS-6 model, and the FSRS-7 model
(34 parameters, a fast and a slow memory component) used by the JSchoreels
Anki fork, ported from fsrs-rs ``model_v7.rs``.  The implementation is
independent and deterministic: it follows the expected Good-answer path, while
charging ``1 / desired_retention`` reviews for each successful review.  This
keeps the result stable enough to be a daily goal rather than a Monte Carlo
number that changes each time the dialog opens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


MATURE_INTERVAL_DAYS = 21
DEFAULT_DESIRED_RETENTION = 0.90
DEFAULT_FSRS_DECAY = 0.1542

# FSRS-6 defaults used when a preset has not been optimized.  Anki represents
# an unoptimized preset with an empty parameter array, so a local fallback is
# required for projecting cards that do not exist yet.
DEFAULT_FSRS_PARAMETERS = (
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    DEFAULT_FSRS_DECAY,
)

# FSRS-7 defaults (fsrs-rs ``inference_v7::DEFAULT_PARAMETERS``), used by the
# JSchoreels Anki fork for presets that have not been optimized yet.
FSRS7_PARAMETER_COUNT = 34
DEFAULT_FSRS7_PARAMETERS = (
    0.1104, 2.2395, 3.9221, 11.7841, 6.1686, 0.6457, 3.6807, 1.9795, 0.0,
    1.3826, 0.7024, 0.5999, 0.8146, 0.6398, 1.0, 1.3207, 0.6707, 3.8668,
    0.4416, 0.0934, 1.8631, 0.6162, 1.0869, 0.1567, 0.0801, 0.2421, 0.9464,
    0.1433, 0.7145, 0.0, 0.5667, 0.3734, 0.5333, 0.3048,
)
FSRS7_S_MIN = 0.0001
FSRS7_S_MAX = 36500.0
FSRS7_FAST_STABILITY_RATIO = 0.8
GOOD = 3


@dataclass(frozen=True)
class FsrsSettings:
    """Scheduling inputs shared by a group of cards."""

    desired_retention: float = DEFAULT_DESIRED_RETENTION
    parameters: tuple[float, ...] = DEFAULT_FSRS_PARAMETERS
    learning_steps: tuple[float, ...] = (1.0, 10.0)
    maximum_interval: int = 36500


@dataclass(frozen=True)
class ProjectionCard:
    """Small, serializable card snapshot needed by the projection."""

    interval: int = 0
    due_day: int = 0
    stability: float | None = None
    difficulty: float | None = None
    last_review_day: int | None = None
    is_new: bool = False
    learning_steps_remaining: int = 0
    settings: FsrsSettings = FsrsSettings()
    # FSRS-7 only: the fast memory component ("s_fast" in card data).
    stability_fast: float | None = None
    # Per-card FSRS values stored by Anki in card data ("dr" / "decay").
    # Dynamic desired retention (e.g. the JSchoreels Anki fork) gives each
    # card its own retention, so these take precedence over the preset.
    desired_retention: float | None = None
    decay: float | None = None


@dataclass(frozen=True)
class CardPath:
    matures_on: int | None
    reviews_to_mature: float
    reviews_through_exam: float


def clamp_retention(value: float) -> float:
    """Keep the model inside Anki's supported desired-retention range."""

    return max(0.70, min(0.99, float(value or DEFAULT_DESIRED_RETENTION)))


def card_retention(value: float) -> float:
    """Clamp a per-card retention to the fork's dynamic DR range."""

    return max(0.30, min(0.995, float(value)))


def is_fsrs7(parameters: Sequence[float]) -> bool:
    return len(parameters) == FSRS7_PARAMETER_COUNT


def normalized_settings(settings: FsrsSettings) -> FsrsSettings:
    params = tuple(float(value) for value in settings.parameters)
    if len(params) not in (21, FSRS7_PARAMETER_COUNT) or any(
        not math.isfinite(value) for value in params
    ):
        params = DEFAULT_FSRS_PARAMETERS
    return FsrsSettings(
        desired_retention=clamp_retention(settings.desired_retention),
        parameters=params,
        learning_steps=tuple(max(0.0, float(step)) for step in settings.learning_steps),
        maximum_interval=max(1, int(settings.maximum_interval or 36500)),
    )


def _factor(decay: float) -> float:
    decay = max(0.01, float(decay or DEFAULT_FSRS_DECAY))
    return (0.9 ** (-1.0 / decay)) - 1.0


def interval_for_stability(
    stability: float,
    desired_retention: float,
    decay: float,
    maximum_interval: int,
) -> int:
    """Return FSRS's unfuzzed interval for a memory stability value."""

    decay = max(0.01, float(decay or DEFAULT_FSRS_DECAY))
    retention = clamp_retention(desired_retention)
    interval = (max(0.001, stability) / _factor(decay)) * (
        retention ** (-1.0 / decay) - 1.0
    )
    return max(1, min(int(maximum_interval), round(interval)))


def stability_for_interval(
    interval: int,
    desired_retention: float,
    decay: float,
) -> float:
    """Estimate missing stability by inverting FSRS's interval equation."""

    decay = max(0.01, float(decay or DEFAULT_FSRS_DECAY))
    denominator = clamp_retention(desired_retention) ** (-1.0 / decay) - 1.0
    if denominator <= 0:
        return max(0.001, float(interval))
    return max(0.001, max(1, int(interval)) * _factor(decay) / denominator)


def retrievability(stability: float, elapsed_days: int, decay: float) -> float:
    decay = max(0.01, float(decay or DEFAULT_FSRS_DECAY))
    return (
        1.0 + _factor(decay) * max(0, elapsed_days) / max(0.001, stability)
    ) ** (-decay)


def _initial_difficulty(params: Sequence[float]) -> float:
    value = params[4] - math.exp(params[5] * 2.0) + 1.0
    return max(1.0, min(10.0, value))


def _next_difficulty(difficulty: float, params: Sequence[float]) -> float:
    initial_easy = params[4] - math.exp(params[5] * 3.0) + 1.0
    delta = (10.0 - difficulty) * 0.0 / 9.0  # Good has zero difficulty delta.
    value = params[7] * initial_easy + (1.0 - params[7]) * (difficulty + delta)
    return max(1.0, min(10.0, value))


def _next_good_stability(
    difficulty: float,
    stability: float,
    recall_probability: float,
    params: Sequence[float],
) -> float:
    increase = (
        math.exp(params[8])
        * (11.0 - difficulty)
        * (max(0.001, stability) ** -params[9])
        * (math.exp((1.0 - recall_probability) * params[10]) - 1.0)
    )
    return max(0.001, stability * (1.0 + increase))


def _short_term_good_stability(stability: float, params: Sequence[float]) -> float:
    increase = math.exp(params[17] * params[18]) * (
        max(0.001, stability) ** -params[19]
    )
    return max(0.001, stability * max(1.0, increase))


def _learning_review_days(steps: Sequence[float], remaining: int) -> list[int]:
    """Return day offsets for the expected Good learning sequence.

    Learning steps are stored as minutes.  Sub-day steps remain on the same
    projection day; interday steps move the next answer to a later day.
    """

    if remaining <= 0:
        remaining = max(1, len(steps))
    relevant = list(steps[-remaining:]) if steps else [0.0]
    offsets = []
    day = 0
    for index in range(max(1, remaining)):
        offsets.append(day)
        if index < len(relevant) - 1:
            day += int(max(0.0, relevant[index + 1]) // 1440.0)
    return offsets


def effective_retention(card: ProjectionCard) -> float:
    """The card's own desired retention when known, else its preset's."""

    if card.desired_retention:
        return card_retention(card.desired_retention)
    return clamp_retention(card.settings.desired_retention)


@dataclass(frozen=True)
class Fsrs7State:
    stability: float
    stability_fast: float
    difficulty: float


def _clamp_s(value: float) -> float:
    return max(FSRS7_S_MIN, min(FSRS7_S_MAX, float(value)))


def _clamp_d(value: float) -> float:
    return max(1.0, min(10.0, float(value)))


def _fsrs7_initial_difficulty(params: Sequence[float], rating: int) -> float:
    return params[4] - math.exp(params[5] * (rating - 1)) + 1.0


def fsrs7_initial_state(params: Sequence[float], rating: int = GOOD) -> Fsrs7State:
    stability = _clamp_s(params[max(1, min(4, rating)) - 1])
    return Fsrs7State(
        stability=stability,
        stability_fast=_clamp_s(stability * FSRS7_FAST_STABILITY_RATIO),
        difficulty=_clamp_d(_fsrs7_initial_difficulty(params, rating)),
    )


def _fsrs7_fast_recall(params: Sequence[float], elapsed: float, s_fast: float) -> float:
    s_fast = _clamp_s(s_fast)
    decay = -max(0.01, min(0.95, params[23] * s_fast ** (params[33] - 0.3)))
    factor = math.exp(min(60.0, math.log(params[25]) / decay)) - 1.0
    return (1.0 + factor * (max(0.0, elapsed) / s_fast)) ** decay


def fsrs7_retrievability(
    params: Sequence[float], elapsed_days: float, state: Fsrs7State
) -> float:
    """FSRS-7 forgetting curve: a weighted mix of a fast and a slow curve."""

    elapsed = max(0.0, float(elapsed_days))
    s = max(FSRS7_S_MIN, state.stability)
    s_fast = max(FSRS7_S_MIN, state.stability_fast)
    d = _clamp_d(state.difficulty)
    r1 = _fsrs7_fast_recall(params, elapsed, s_fast)
    decay2 = -max(0.01, min(0.95, params[24]))
    factor2 = params[26] ** (1.0 / decay2) - 1.0
    d_timescale = math.exp((d - 5.0) * (params[32] - 0.3))
    r2 = (1.0 + factor2 * d_timescale * (elapsed / s)) ** decay2
    weight1 = params[27] * s_fast ** (-params[29])
    weight2 = params[28] * s ** params[30] * math.exp((d - 5.0) * (params[31] - 0.5))
    retention = (weight1 * r1 + weight2 * r2) / max(1e-9, weight1 + weight2)
    return retention * (1.0 - 2e-5) + 1e-5


def _fsrs7_stability_for_set(
    params: Sequence[float],
    last_s: float,
    last_d: float,
    recall: float,
    rating: int,
    start: int,
) -> float:
    hard_penalty = params[start + 6] if rating == 2 else 1.0
    easy_bonus = params[start + 7] if rating == 4 else 1.0
    new_s_fail = (
        params[start + 3]
        * ((last_s + 1.0) ** params[start + 4] - 1.0)
        * math.exp((1.0 - recall) * params[start + 5])
    )
    post_lapse = min(last_s, new_s_fail)
    increase = (
        math.exp(params[start] - 1.5)
        * (11.0 - last_d)
        * last_s ** (-params[start + 1])
        * (math.exp((1.0 - recall) * params[start + 2]) - 1.0)
        * hard_penalty
        * easy_bonus
        + 1.0
    )
    new_s = max(post_lapse, last_s * increase) if rating > 1 else post_lapse
    return _clamp_s(new_s)


def _fsrs7_next_difficulty(
    params: Sequence[float], difficulty: float, rating: int, recall: float
) -> float:
    delta = -params[6] * (rating - 3.0)
    if rating == 1:
        delta *= recall + 0.1
    new_d = difficulty + (10.0 - difficulty) * delta / 9.0
    initial_easy = _fsrs7_initial_difficulty(params, 4)
    return _clamp_d(initial_easy * 0.01 + new_d * 0.99)


def fsrs7_next_state(
    params: Sequence[float],
    state: Fsrs7State,
    elapsed_days: float,
    rating: int = GOOD,
) -> Fsrs7State:
    """FSRS-7 memory update; same-day steps use the same equations."""

    state = Fsrs7State(
        _clamp_s(state.stability),
        _clamp_s(state.stability_fast),
        _clamp_d(state.difficulty),
    )
    elapsed = max(0.0, float(elapsed_days))
    recall = fsrs7_retrievability(params, elapsed, state)
    new_s = _fsrs7_stability_for_set(
        params, state.stability, state.difficulty, recall, rating, 7
    )
    fast_recall = _fsrs7_fast_recall(params, elapsed, state.stability_fast)
    new_s_fast = _fsrs7_stability_for_set(
        params, state.stability_fast, state.difficulty, fast_recall, rating, 15
    )
    if rating == 1:
        new_s_fast = min(new_s_fast, new_s * FSRS7_FAST_STABILITY_RATIO)
    return Fsrs7State(
        stability=new_s,
        stability_fast=_clamp_s(new_s_fast),
        difficulty=_fsrs7_next_difficulty(params, state.difficulty, rating, recall),
    )


def fsrs7_interval(
    params: Sequence[float],
    state: Fsrs7State,
    desired_retention: float,
    maximum_interval: int,
) -> int:
    """Unfuzzed FSRS-7 interval in days where recall falls to the target."""

    target = max(0.0001, min(0.9999, float(desired_retention)))
    low = 0.0
    high = max(1.0, state.stability, state.stability_fast)
    while fsrs7_retrievability(params, high, state) > target and high < FSRS7_S_MAX:
        high = min(FSRS7_S_MAX, high * 2.0)
    for _ in range(50):
        mid = (low + high) * 0.5
        if fsrs7_retrievability(params, mid, state) > target:
            low = mid
        else:
            high = mid
    return max(1, min(int(maximum_interval), round((low + high) * 0.5)))


def fsrs7_state_for_interval(
    params: Sequence[float], interval: int, desired_retention: float
) -> Fsrs7State:
    """Memory state whose recall after ``interval`` days equals the target.

    Mirrors fsrs-rs ``memory_state_from_sm2`` for cards without FSRS memory.
    """

    target = max(0.0001, min(0.9999, float(desired_retention)))
    elapsed = max(FSRS7_S_MIN, float(interval))

    def state_at(log_s: float) -> Fsrs7State:
        s = _clamp_s(math.exp(log_s))
        return Fsrs7State(s, _clamp_s(s * FSRS7_FAST_STABILITY_RATIO), 5.0)

    low, high = math.log(FSRS7_S_MIN), math.log(FSRS7_S_MAX)
    for _ in range(50):
        mid = (low + high) * 0.5
        if fsrs7_retrievability(params, elapsed, state_at(mid)) < target:
            low = mid
        else:
            high = mid
    return state_at((low + high) * 0.5)


def _learning_step_gaps(steps: Sequence[float], remaining: int) -> list[float]:
    """Elapsed days before each expected Good answer in learning."""

    if remaining <= 0:
        remaining = max(1, len(steps))
    relevant = list(steps[-remaining:]) if steps else [0.0]
    return [
        max(0.0, float(relevant[index])) / 1440.0 if 0 < index < len(relevant) else 0.0
        for index in range(max(1, remaining))
    ]


def _project_card_fsrs7(card: ProjectionCard, horizon_days: int) -> CardPath:
    """FSRS-7 counterpart of ``project_card`` with the same Good-path rules."""

    horizon = max(0, int(horizon_days))
    settings = normalized_settings(card.settings)
    params = settings.parameters
    retention = effective_retention(card)
    day = max(0, int(card.due_day))
    interval = max(0, int(card.interval))
    reviews = 0.0
    matures_on = 0 if interval >= MATURE_INTERVAL_DAYS else None
    state = None
    if card.stability is not None:
        state = Fsrs7State(
            stability=card.stability,
            stability_fast=card.stability_fast
            or card.stability * FSRS7_FAST_STABILITY_RATIO,
            difficulty=card.difficulty or 5.0,
        )

    if matures_on is None and (card.is_new or card.learning_steps_remaining):
        learning_days = _learning_review_days(
            settings.learning_steps,
            card.learning_steps_remaining,
        )
        gaps = _learning_step_gaps(
            settings.learning_steps, card.learning_steps_remaining
        )
        for offset, gap in zip(learning_days, gaps):
            if day + offset >= horizon:
                return CardPath(None, reviews, reviews)
            reviews += 1.0
            if state is None:
                state = fsrs7_initial_state(params)
            else:
                state = fsrs7_next_state(params, state, gap)
        day += learning_days[-1]
        interval = fsrs7_interval(params, state, retention, settings.maximum_interval)
        if interval >= MATURE_INTERVAL_DAYS:
            matures_on = day
        day += interval
    elif state is None:
        state = fsrs7_state_for_interval(params, max(1, interval), retention)

    reviews_to_mature = reviews
    first_review = True
    while day < horizon:
        elapsed = interval
        if card.last_review_day is not None and first_review:
            elapsed = max(0, day - card.last_review_day)
        recall = fsrs7_retrievability(params, elapsed, state)
        reviews += 1.0 / max(0.05, recall)
        state = fsrs7_next_state(params, state, elapsed)
        interval = fsrs7_interval(params, state, retention, settings.maximum_interval)
        if matures_on is None and interval >= MATURE_INTERVAL_DAYS:
            matures_on = day
            reviews_to_mature = reviews
        day += interval
        first_review = False

    if matures_on == 0 and not (card.is_new or card.learning_steps_remaining):
        # Existing mature cards still create maintenance work before the exam.
        return CardPath(0, 0.0, reviews)
    if matures_on is None:
        reviews_to_mature = reviews
    return CardPath(matures_on, reviews_to_mature, reviews)


def project_card(card: ProjectionCard, horizon_days: int) -> CardPath:
    """Project one card along its expected FSRS Good-answer path."""

    horizon = max(0, int(horizon_days))
    settings = normalized_settings(card.settings)
    params = settings.parameters
    if is_fsrs7(params):
        return _project_card_fsrs7(card, horizon)
    retention = effective_retention(card)
    decay = max(0.01, float(card.decay or params[20]))
    day = max(0, int(card.due_day))
    stability = card.stability
    difficulty = card.difficulty
    interval = max(0, int(card.interval))
    reviews = 0.0
    matures_on = 0 if interval >= MATURE_INTERVAL_DAYS else None

    if matures_on is not None:
        # Existing mature cards still create maintenance work before the exam.
        first_review = True
        while day < horizon:
            elapsed = interval
            if card.last_review_day is not None and first_review:
                elapsed = max(0, day - card.last_review_day)
            stability = stability or stability_for_interval(interval, retention, decay)
            difficulty = difficulty or _initial_difficulty(params)
            recall = retrievability(stability, elapsed, decay)
            reviews += 1.0 / max(0.05, recall)
            stability = _next_good_stability(difficulty, stability, recall, params)
            difficulty = _next_difficulty(difficulty, params)
            interval = interval_for_stability(
                stability, retention, decay, settings.maximum_interval
            )
            day += interval
            first_review = False
        return CardPath(0, 0.0, reviews)

    if card.is_new or card.learning_steps_remaining:
        learning_days = _learning_review_days(
            settings.learning_steps,
            card.learning_steps_remaining,
        )
        for index, offset in enumerate(learning_days):
            review_day = day + offset
            if review_day >= horizon:
                return CardPath(None, reviews, reviews)
            reviews += 1.0
            if stability is None:
                stability = max(0.001, params[2])
                difficulty = _initial_difficulty(params)
            elif index:
                stability = _short_term_good_stability(stability, params)
                difficulty = _next_difficulty(
                    difficulty or _initial_difficulty(params), params
                )
        day += learning_days[-1]
        interval = interval_for_stability(
            stability or params[2], retention, decay, settings.maximum_interval
        )
        if interval >= MATURE_INTERVAL_DAYS:
            matures_on = day
        day += interval
    else:
        stability = stability or stability_for_interval(
            max(1, interval), retention, decay
        )
        difficulty = difficulty or _initial_difficulty(params)

    reviews_to_mature = reviews
    first_review = True
    while day < horizon:
        if card.last_review_day is not None and first_review:
            recall = retrievability(
                stability or params[2],
                max(0, day - card.last_review_day),
                decay,
            )
        else:
            recall = retention
        reviews += 1.0 / max(0.05, recall)
        stability = _next_good_stability(
            difficulty or _initial_difficulty(params),
            stability or params[2],
            recall,
            params,
        )
        difficulty = _next_difficulty(
            difficulty or _initial_difficulty(params), params
        )
        interval = interval_for_stability(
            stability, retention, decay, settings.maximum_interval
        )
        if matures_on is None and interval >= MATURE_INTERVAL_DAYS:
            matures_on = day
            reviews_to_mature = reviews
        day += interval
        first_review = False

    if matures_on is None:
        reviews_to_mature = reviews
    return CardPath(matures_on, reviews_to_mature, reviews)


def evenly_spaced_arrival_days(count: int, horizon_days: int) -> list[int]:
    """Spread user-entered future cards across the remaining study days."""

    count = max(0, int(count))
    horizon = max(1, int(horizon_days))
    return [
        min(horizon - 1, (index * horizon) // max(1, count))
        for index in range(count)
    ]


def calculate_plan_projection(
    cards: Iterable[ProjectionCard],
    future_cards: Iterable[ProjectionCard],
    horizon_days: int,
    target_percent: int,
) -> dict:
    """Calculate the minimum average daily workload for a maturity target."""

    current = list(cards)
    future = list(future_cards)
    horizon = max(1, int(horizon_days))
    target = max(1, min(100, int(target_percent)))
    current_mature = sum(
        1 for card in current if card.interval >= MATURE_INTERVAL_DAYS
    )
    projected_total = len(current) + len(future)
    target_mature = math.ceil(projected_total * target / 100.0)
    mature_gap = max(0, target_mature - current_mature)

    required_existing_reviews = 0.0
    expected_mature = current_mature
    candidates: list[CardPath] = []
    for card in current:
        path = project_card(card, horizon)
        if card.interval >= MATURE_INTERVAL_DAYS:
            required_existing_reviews += path.reviews_through_exam
        elif card.is_new:
            if path.matures_on is not None:
                candidates.append(path)
        else:
            # Scheduled young/learning cards remain part of the deck's normal
            # FSRS workload even when the maturity target is already met.
            required_existing_reviews += path.reviews_through_exam
            if path.matures_on is not None:
                expected_mature += 1
    for card in future:
        path = project_card(card, horizon)
        if path.matures_on is not None:
            candidates.append(path)

    candidates.sort(
        key=lambda item: (
            item.reviews_through_exam,
            item.matures_on if item.matures_on is not None else horizon + 1,
        )
    )
    introductions_needed = max(0, target_mature - expected_mature)
    selected = candidates[:introductions_needed]
    achievable_mature = expected_mature + len(candidates)
    projected_mature = expected_mature + len(selected)
    achievable_percent = (
        (achievable_mature / projected_total * 100.0) if projected_total else 100.0
    )
    total_reviews = required_existing_reviews + sum(
        item.reviews_through_exam for item in selected
    )
    required_per_day = total_reviews / horizon

    return {
        "current_total": len(current),
        "future_new_cards": len(future),
        "projected_total": projected_total,
        "current_mature": current_mature,
        "current_mature_percent": (
            current_mature / len(current) * 100.0 if current else 0.0
        ),
        "target_mature": target_mature,
        "target_mature_percent": target,
        "mature_gap": mature_gap,
        "projected_mature": projected_mature,
        "achievable_mature": achievable_mature,
        "achievable_percent": min(100.0, achievable_percent),
        "feasible": len(candidates) >= introductions_needed,
        "expected_mature_from_scheduled": expected_mature,
        "cards_to_introduce": introductions_needed,
        "maintenance_reviews": required_existing_reviews,
        "workload_reviews": total_reviews,
        "required_per_day": required_per_day,
    }

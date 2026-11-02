import unittest

from study_tools.prep_station_math import (
    DEFAULT_FSRS7_PARAMETERS,
    FsrsSettings,
    ProjectionCard,
    calculate_plan_projection,
    effective_retention,
    evenly_spaced_arrival_days,
    fsrs7_initial_state,
    fsrs7_interval,
    fsrs7_retrievability,
    fsrs7_state_for_interval,
    project_card,
)

FSRS7 = FsrsSettings(parameters=DEFAULT_FSRS7_PARAMETERS)


class Fsrs7MathTests(unittest.TestCase):
    def test_initial_states_match_fsrs_rs_reference(self):
        # fsrs-rs FSRS::next_states() doc example (default FSRS-7 parameters).
        expected = {
            1: (0.1104, 0.08832, 6.1686, 1),
            2: (2.2395, 1.7916001, 5.261278, 1),
            3: (3.9221, 3.13768, 3.5307243, 5),
            4: (11.7841, 9.427279, 1.0, 54),
        }
        for rating, (s, s_fast, d, interval) in expected.items():
            state = fsrs7_initial_state(DEFAULT_FSRS7_PARAMETERS, rating)
            self.assertAlmostEqual(state.stability, s, places=4)
            self.assertAlmostEqual(state.stability_fast, s_fast, places=4)
            self.assertAlmostEqual(state.difficulty, d, places=4)
            self.assertEqual(
                fsrs7_interval(DEFAULT_FSRS7_PARAMETERS, state, 0.9, 36500), interval
            )

    def test_good_interval_hits_desired_retention(self):
        state = fsrs7_initial_state(DEFAULT_FSRS7_PARAMETERS)
        self.assertAlmostEqual(
            fsrs7_retrievability(DEFAULT_FSRS7_PARAMETERS, 4.7777247, state),
            0.9,
            places=4,
        )

    def test_state_for_interval_round_trips(self):
        for interval in (3, 30, 120):
            state = fsrs7_state_for_interval(DEFAULT_FSRS7_PARAMETERS, interval, 0.85)
            self.assertEqual(
                fsrs7_interval(DEFAULT_FSRS7_PARAMETERS, state, 0.85, 36500), interval
            )

    def test_fsrs7_higher_retention_requires_more_reviews(self):
        low = project_card(
            ProjectionCard(is_new=True, settings=FSRS7, desired_retention=0.80), 90
        )
        high = project_card(
            ProjectionCard(is_new=True, settings=FSRS7, desired_retention=0.95), 90
        )

        self.assertGreater(high.reviews_to_mature, low.reviews_to_mature)
        self.assertGreater(high.matures_on, low.matures_on)

    def test_fsrs7_mature_card_only_adds_maintenance(self):
        path = project_card(
            ProjectionCard(
                interval=40, due_day=10, stability=40.0, difficulty=6.0,
                stability_fast=12.0, last_review_day=-30, settings=FSRS7,
            ),
            200,
        )

        self.assertEqual(path.matures_on, 0)
        self.assertGreater(path.reviews_through_exam, 0)


class PrepStationMathTests(unittest.TestCase):
    def test_future_cards_are_part_of_target_denominator(self):
        cards = [ProjectionCard(interval=30, due_day=40) for _ in range(8)]
        future = [ProjectionCard(is_new=True) for _ in range(2)]

        result = calculate_plan_projection(cards, future, 14, 80)

        self.assertEqual(result["projected_total"], 10)
        self.assertEqual(result["target_mature"], 8)
        self.assertTrue(result["feasible"])

    def test_higher_retention_requires_more_reviews_to_mature(self):
        low = project_card(
            ProjectionCard(
                is_new=True,
                settings=FsrsSettings(desired_retention=0.80),
            ),
            60,
        )
        high = project_card(
            ProjectionCard(
                is_new=True,
                settings=FsrsSettings(desired_retention=0.95),
            ),
            60,
        )

        self.assertGreater(high.reviews_to_mature, low.reviews_to_mature)
        self.assertGreater(high.matures_on, low.matures_on)

    def test_late_future_cards_can_make_target_unreachable(self):
        future = [
            ProjectionCard(is_new=True, due_day=day)
            for day in evenly_spaced_arrival_days(10, 21)
        ]

        result = calculate_plan_projection([], future, 21, 80)

        self.assertFalse(result["feasible"])
        self.assertLess(result["achievable_percent"], 80)

    def test_scheduled_young_cards_remain_in_daily_workload(self):
        cards = [
            ProjectionCard(interval=30, due_day=40),
            ProjectionCard(interval=5, due_day=0),
        ]

        result = calculate_plan_projection(cards, [], 30, 50)

        self.assertTrue(result["feasible"])
        self.assertGreater(result["workload_reviews"], 0)
        self.assertGreater(result["expected_mature_from_scheduled"], 1)

    def test_card_desired_retention_overrides_preset(self):
        preset = FsrsSettings(desired_retention=0.80)
        per_card = project_card(
            ProjectionCard(is_new=True, settings=preset, desired_retention=0.95),
            60,
        )
        high = project_card(
            ProjectionCard(
                is_new=True,
                settings=FsrsSettings(desired_retention=0.95),
            ),
            60,
        )

        self.assertEqual(per_card, high)
        self.assertEqual(
            effective_retention(ProjectionCard(settings=preset)), 0.80
        )


if __name__ == "__main__":
    unittest.main()

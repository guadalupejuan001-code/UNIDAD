"""Pure-Python checks for bounded TRAIN sampling and resumable score caches."""

import copy
import json
import math
import random
import unittest

from ia_propia.conversation.adaptive import AdaptiveSampler, MAX_COUNT, validate_config


CONFIG = {"mode": "train_loss_ema"}


def sampler(size=3, bases=None, config=None, state=None):
    return AdaptiveSampler([f"train-window-{i}" for i in range(size)], bases or [1.] * size,
                           CONFIG if config is None else config, state=state)


class AdaptiveSamplingTests(unittest.TestCase):
    def test_option_absence_and_explicit_mode(self):
        self.assertIsNone(validate_config(None))
        self.assertEqual(validate_config(CONFIG)["refresh_every_steps"], 64)
        with self.assertRaises(ValueError):
            validate_config({})
        with self.assertRaises(ValueError):
            AdaptiveSampler(["train-window-0"], [1.], None)

    def test_initial_and_equal_difficulty_keep_provenance_distribution(self):
        subject = sampler(bases=[1., 7., 100.])
        self.assertEqual(subject.weights(), [1., 7., 100.])
        subject.update([0, 1, 2], [8., 8., 8.])
        self.assertEqual(subject.weights(), [1., 7., 100.])
        self.assertEqual(subject.summary()["observedWindows"], 3)

    def test_real_distribution_mixture_has_independent_rational_reference(self):
        subject = sampler(bases=[100., 1., 1.], config={**CONFIG, "exponent": 1.})
        subject.update([0, 1, 2], [0., 0., 1000.])
        # Priorities are .5,.5,2. Hard probabilities are 20/21,1/105,4/105.
        # Mix one quarter of the baseline (100/102,1/102,1/102) with
        # three quarters of that hard distribution, independently of code.
        expected = [25/102 + 5/7, 1/408 + 1/140, 1/408 + 1/35]
        weights = subject.weights()
        self.assertAlmostEqual(sum(weights), 102., places=11)
        for weight, probability in zip(weights, expected):
            self.assertAlmostEqual(weight / sum(weights), probability, places=13)

    def test_rare_irreducible_outlier_cannot_take_all_probability(self):
        subject = sampler(100)
        subject.update(list(range(100)), [0.] * 99 + [1_000_000.])
        weights = subject.weights()
        self.assertGreater(weights[-1], weights[0])
        self.assertLessEqual(weights[-1], 3.25)
        self.assertGreaterEqual(min(weights), .4375)
        self.assertLess(weights[-1] / sum(weights), .033)
        self.assertFalse(subject.summary()["qualityImprovementValidated"])

    def test_unseen_windows_are_neutral_raw_priorities_not_false_scores(self):
        subject = sampler()
        subject.update([0, 1], [0., 1000.])
        state = subject.state_dict()
        self.assertEqual(state["scores"][2], None)
        self.assertEqual(state["counts"][2], 0)
        weights = subject.weights()
        self.assertLess(weights[0], weights[2])
        self.assertLess(weights[2], weights[1])
        self.assertTrue(all(weight > 0 for weight in weights))

    def test_coverage_probability_floor_even_with_extreme_provenance(self):
        base = [1e-6, 1., 100000., 1e12]
        subject = sampler(4, bases=base)
        subject.update([0, 1, 2, 3], [1e6, 200., 1., 0.])
        weights = subject.weights()
        for original, weight in zip(base, weights):
            self.assertGreaterEqual(weight/sum(weights) + 1e-30, .25*original/sum(base))
            self.assertGreaterEqual(weight/original, .4375-1e-12)
            self.assertLessEqual(weight/original, 3.25+1e-12)

    def test_coverage_one_or_exponent_zero_replays_baseline(self):
        for config in ({**CONFIG, "coverage": 1.}, {**CONFIG, "exponent": 0.}):
            subject = sampler(bases=[1., 2., 20.], config=config)
            subject.update([0, 1, 2], [100., 0., 1.])
            self.assertEqual(subject.weights(), [1., 2., 20.])

    def test_first_score_and_ema_are_observed_losses_not_initial_prior(self):
        subject = sampler()
        subject.update([0], [10.])
        self.assertEqual(subject.state_dict()["scores"][0], 10.)
        subject.update([0], [0.])
        self.assertEqual(subject.state_dict()["scores"][0], 9.)
        subject.update([0, 0], [2., 6.])
        self.assertEqual(subject.state_dict()["scores"][0], 8.5)
        self.assertEqual(subject.state_dict()["counts"][0], 4)

    def test_duplicate_draw_order_does_not_change_one_batch_ema(self):
        left, right = sampler(), sampler()
        left.update([0], [8.]); right.update([0], [8.])
        left.update([0, 0], [1., 9.]); right.update([0, 0], [9., 1.])
        self.assertEqual(left.state_dict(), right.state_dict())
        self.assertAlmostEqual(left.state_dict()["scores"][0], 7.7)
        self.assertEqual(left.state_dict()["counts"][0], 3)

    def test_preflight_does_not_commit_a_failed_optimizer_step(self):
        subject = sampler()
        before = subject.state_dict()
        self.assertIsNone(subject.validate_update([0, 1], [10., 2.]))
        self.assertEqual(subject.state_dict(), before)
        self.assertEqual(subject.weights(), [1., 1., 1.])
        subject.update([0, 1], [10., 2.])
        self.assertEqual(subject.state_dict()["counts"], [1, 1, 0])

    def test_invalid_updates_and_heldout_losses_are_atomic(self):
        subject = sampler()
        subject.update([0], [2.])
        before = subject.state_dict()
        invalid = [([0, 1], [1., math.nan], "train"), ([0, 3], [1., 2.], "train"),
                   ([0, True], [1., 2.], "train"), ([0, 1], [1., True], "train"),
                   ([0], [-1.], "train"), ([0], [math.inf], "train"), ([0], [1e6+1], "train"),
                   ([0], [2.], "validation"), ([0], [2.], "test"), ([], [], "train"),
                   ([0], [], "train"), ([0]*257, [1.]*257, "train")]
        for indices, losses, split in invalid:
            for method in (subject.validate_update, subject.update):
                with self.subTest(indices=indices[:3], split=split, method=method.__name__):
                    with self.assertRaises(ValueError):
                        method(indices, losses, split=split)
                    self.assertEqual(subject.state_dict(), before)

    def test_count_overflow_is_checked_before_any_window_commit(self):
        state = sampler().state_dict()
        state["scores"][0], state["counts"][0] = 1., MAX_COUNT
        subject = sampler(state=state)
        before = subject.state_dict()
        for method in (subject.validate_update, subject.update):
            with self.assertRaises(ValueError):
                method([1, 0], [3., 5.])
            self.assertEqual(subject.state_dict(), before)

    def test_json_restore_preserves_next_cache_weights_and_external_draws(self):
        subject = sampler(bases=[1., 4., 16.])
        subject.update([0, 1, 1], [2., 4., 8.])
        saved = json.loads(json.dumps(subject.state_dict(), allow_nan=False))
        restored = sampler(bases=[1., 4., 16.], state=saved)
        self.assertEqual(subject.state_dict(), restored.state_dict())
        rng = random.Random(73)
        rng.choices(range(3), weights=subject.weights(), k=20)
        resumed_rng = random.Random()
        resumed_rng.setstate(rng.getstate())
        subject.update([2, 0], [12., 1.]); restored.update([2, 0], [12., 1.])
        self.assertEqual(subject.weights(), restored.weights())
        self.assertEqual(rng.choices(range(3), weights=subject.weights(), k=100),
                         resumed_rng.choices(range(3), weights=restored.weights(), k=100))

    def test_public_snapshots_and_config_cannot_mutate_internal_cache(self):
        subject = sampler()
        subject.update([0], [2.])
        before = subject.state_dict()
        exposed = subject.state_dict()
        exposed["scores"][0] = 999.
        exposed["counts"][0] = 99
        exposed["config"]["coverage"] = 1.
        exposed["windowIds"][0] = "different"
        exposed["baseWeights"][0] = 999.
        config = subject.config
        config["coverage"] = 1.
        self.assertEqual(subject.state_dict(), before)

    def test_restore_rejects_corruption_and_new_corpus_bindings(self):
        valid = sampler().state_dict()
        mutations = [lambda s: s.update(schemaVersion=True), lambda s: s.update(extra=1),
                     lambda s: s["windowIds"].reverse(), lambda s: s["baseWeights"].__setitem__(0, 2.),
                     lambda s: s["baseWeights"].__setitem__(0, True), lambda s: s["config"].update(coverage=.5),
                     lambda s: s["config"].pop("epsilon"), lambda s: s.update(scores=[]),
                     lambda s: s["counts"].__setitem__(0, True), lambda s: s["counts"].__setitem__(0, -1),
                     lambda s: s["counts"].__setitem__(0, MAX_COUNT+1),
                     lambda s: s["counts"].__setitem__(0, 1), lambda s: s["scores"].__setitem__(0, 0.),
                     lambda s: (s["scores"].__setitem__(0, math.nan), s["counts"].__setitem__(0, 1))]
        for mutation in mutations:
            invalid = copy.deepcopy(valid)
            mutation(invalid)
            with self.assertRaises(ValueError):
                sampler(state=invalid)

    def test_ids_baseweights_and_config_extremes_fail_clearly(self):
        for ids, bases in (([], []), (["x", "x"], [1., 1.]), ([""], [1.]), (["x\n"], [1.]),
                           (["\ud800"], [1.]), (["x"*513], [1.]), (["x"], []),
                           (["x"], [0.]), (["x"], [True]), (["x"], [math.inf]), (["x"], [1e12+1]), (["x"], [10**10000])):
            with self.assertRaises(ValueError):
                AdaptiveSampler(ids, bases, CONFIG)
        for key, value in (("mode", "test_loss"), ("ema_decay", 1.), ("exponent", 3.),
                           ("coverage", .24), ("min_priority", 0.), ("max_priority", 9.),
                           ("epsilon", 0.), ("refresh_every_steps", True), ("refresh_every_steps", 100001),
                           ("unknown", 1), ("coverage", math.nan), ("exponent", "0.5"), ("exponent", 10**10000)):
            with self.assertRaises(ValueError):
                validate_config({**CONFIG, key: value})


if __name__ == "__main__":
    unittest.main()

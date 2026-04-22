import unittest

try:
    from fraud_sentinel.model.training import _select_threshold
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on lightweight local envs.
    _IMPORT_ERROR = exc
    _select_threshold = None
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_select_threshold is None, f"runtime dependencies unavailable: {_IMPORT_ERROR}")
class TrainingThresholdTests(unittest.TestCase):
    def test_select_threshold_respects_target_recall(self) -> None:
        precision = [0.20, 0.40, 0.80, 1.00]
        recall = [0.95, 0.88, 0.70, 0.00]
        thresholds = [0.10, 0.20, 0.70]

        selected = _select_threshold(precision, recall, thresholds, target_recall=0.85)

        self.assertEqual(selected, 0.20)


if __name__ == "__main__":
    unittest.main()

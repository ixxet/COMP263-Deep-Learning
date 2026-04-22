from __future__ import annotations

import unittest

from fraud_sentinel.feature_schema import FEATURE_COLUMNS, check_columns, coerce_transaction


class FeatureSchemaTests(unittest.TestCase):
    def test_valid_transaction_coerces_to_float(self) -> None:
        payload = {column: "1.5" for column in FEATURE_COLUMNS}
        result = coerce_transaction(payload)
        self.assertEqual(set(result), set(FEATURE_COLUMNS))
        self.assertTrue(all(value == 1.5 for value in result.values()))

    def test_missing_column_fails(self) -> None:
        payload = {column: 1 for column in FEATURE_COLUMNS if column != "V28"}
        with self.assertRaisesRegex(ValueError, "missing columns: V28"):
            coerce_transaction(payload)

    def test_non_numeric_value_fails(self) -> None:
        payload = {column: 1 for column in FEATURE_COLUMNS}
        payload["Amount"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "Amount must be numeric"):
            coerce_transaction(payload)

    def test_non_finite_value_fails(self) -> None:
        payload = {column: 1 for column in FEATURE_COLUMNS}
        payload["V1"] = float("inf")
        with self.assertRaisesRegex(ValueError, "V1 must be finite"):
            coerce_transaction(payload)

    def test_extra_label_rejected_at_inference(self) -> None:
        payload = {column: 1 for column in FEATURE_COLUMNS}
        payload["Class"] = 0
        with self.assertRaisesRegex(ValueError, "unexpected columns: Class"):
            coerce_transaction(payload)

    def test_optional_label_allowed_for_batch_uploads(self) -> None:
        payload = {column: 1 for column in FEATURE_COLUMNS}
        payload["Class"] = 0
        result = coerce_transaction(payload, allow_label=True)
        self.assertEqual(set(result), set(FEATURE_COLUMNS))
        self.assertNotIn("Class", result)

    def test_training_schema_accepts_label(self) -> None:
        check = check_columns([*FEATURE_COLUMNS, "Class"], include_label=True)
        self.assertTrue(check.ok)


if __name__ == "__main__":
    unittest.main()

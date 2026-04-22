from __future__ import annotations

import unittest

from fraud_sentinel.agent.grounding import validate_brief


class GroundingTests(unittest.TestCase):
    def test_requires_citation_when_context_exists(self) -> None:
        result = validate_brief("The model supports analyst review.", 2)
        self.assertFalse(result.ok)
        self.assertIn("brief has no citations", result.reasons)

    def test_rejects_prohibited_fraud_claim(self) -> None:
        result = validate_brief("This is definitely fraud [1].", 1)
        self.assertFalse(result.ok)
        self.assertTrue(any("prohibited claim" in reason for reason in result.reasons))

    def test_rejects_empty_brief(self) -> None:
        result = validate_brief("   ", 0)
        self.assertFalse(result.ok)
        self.assertIn("brief is empty", result.reasons)

    def test_rejects_out_of_range_citation(self) -> None:
        result = validate_brief("Analyst should review this case [3].", 2)
        self.assertFalse(result.ok)
        self.assertIn("brief cites context outside retrieved policy documents", result.reasons)

    def test_accepts_grounded_brief(self) -> None:
        result = validate_brief("The model can prioritize review but does not prove fraud [1].", 1)
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from fraud_sentinel.agent.rag import load_policy_documents


class RagTests(unittest.TestCase):
    def test_load_policy_documents_uses_explicit_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_dir = Path(temp_dir)
            (policy_dir / "policy.md").write_text("# Test Policy\nUse citations.", encoding="utf-8")

            docs = load_policy_documents(policy_dir)

        self.assertEqual(len(docs), 1)
        self.assertIsInstance(uuid.UUID(docs[0]["id"]), uuid.UUID)
        self.assertEqual(docs[0]["title"], "Test Policy")
        self.assertEqual(docs[0]["source"], "policy.md")


if __name__ == "__main__":
    unittest.main()

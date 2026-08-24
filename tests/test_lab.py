import importlib
import json
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step1 = importlib.import_module("01_langsmith_rag_pipeline")
        cls.step2 = importlib.import_module("02_prompt_hub_ab_routing")
        cls.step3 = importlib.import_module("03_ragas_evaluation")
        cls.step4 = importlib.import_module("04_guardrails_validator")
        cls.qa = importlib.import_module("qa_pairs")

    def test_dataset_has_exactly_50_aligned_questions(self):
        self.assertEqual(50, len(self.qa.SAMPLE_QUESTIONS))
        self.assertEqual(50, len(self.qa.QA_PAIRS))
        self.assertEqual(
            self.qa.SAMPLE_QUESTIONS,
            [item["question"] for item in self.qa.QA_PAIRS],
        )

    def test_prompts_receive_context_and_question(self):
        self.assertEqual(
            {"context", "question"}, set(self.step1.RAG_PROMPT.input_variables)
        )
        for prompt in (self.step2.PROMPT_V1, self.step2.PROMPT_V2):
            self.assertEqual({"context", "question"}, set(prompt.input_variables))

    def test_prompt_versions_are_semantically_distinct(self):
        self.assertNotEqual(self.step2.SYSTEM_V1, self.step2.SYSTEM_V2)
        self.assertIn("2-4", self.step2.SYSTEM_V1)
        self.assertIn("3-5", self.step2.SYSTEM_V2)

    def test_router_is_deterministic_and_uses_both_variants(self):
        request_ids = [f"req-{i:04d}" for i in range(50)]
        first = [self.step2.get_prompt_version(value) for value in request_ids]
        second = [self.step2.get_prompt_version(value) for value in request_ids]
        self.assertEqual(first, second)
        self.assertEqual(
            {self.step2.PROMPT_V1_NAME, self.step2.PROMPT_V2_NAME}, set(first)
        )

    def test_ragas_dataset_shape(self):
        results = [{
            "question": "q",
            "answer": "a",
            "contexts": ["c1", "c2"],
            "reference": "r",
        }]
        dataset = self.step3.build_ragas_dataset(results)
        self.assertEqual(1, len(dataset))

    def test_pii_patterns_cover_all_required_types(self):
        detector = self.step4.PIIDetector()
        self.assertEqual(
            {"EMAIL", "PHONE", "SSN", "CREDIT_CARD"},
            set(detector.PII_PATTERNS),
        )

    def test_json_repair_and_fallback_are_valid_json(self):
        formatter = self.step4.JSONFormatter()
        repaired = formatter._repair("```json\n{'key': 'value',}\n```")
        self.assertEqual({"key": "value"}, json.loads(repaired))
        failed = formatter.validate("not json {]", {})
        self.assertEqual("Không thể phân tích JSON", json.loads(failed.fix_value)["error"])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock, patch

from eval_utils import eval_single_sample


class TestRealWorldQAJudge(unittest.TestCase):
    def setUp(self):
        self.item = {
            "index": 1, "question": "What color is the car?",
            "A": "red", "B": "blue", "prediction": "green", "answer": "A",
        }

    def test_judge_can_retry_an_inconclusive_reply(self):
        for reply in ("Cannot determine the answer", "I am unsure about Z."):
            with self.subTest(reply=reply):
                judge = Mock(model="test-judge")
                judge.generate.side_effect = [reply, "A"]
                result = eval_single_sample((judge, self.item))
                self.assertEqual(result["extracted_answer"], "A")
                self.assertEqual(result["hit"], 1)
                self.assertTrue(result["extraction_success"])
                self.assertEqual(judge.generate.call_count, 2)

    def test_no_matching_option(self):
        judge = Mock()
        judge.model = "test-judge"
        judge.generate.return_value = " Z\n"
        # Let a random fallback pick the ground truth to expose false positives.
        with patch("eval_utils.random.choice", return_value="A") as guess:
            result = eval_single_sample((judge, self.item))

        self.assertEqual(result["hit"], 0)
        self.assertEqual(result["extracted_answer"], "Z")
        self.assertFalse(result["extraction_success"])
        self.assertEqual(result["extraction_method"], judge.model)
        judge.generate.assert_called_once()
        guess.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Model answers and choices are data, not executable Python."""
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
try:
    import common_utils
    import eval_utils
finally:
    sys.path.pop(0)


class SafeGradingTest(unittest.TestCase):
    def test_model_response_cannot_execute_code(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            # is_equal lowercases responses before grading, so this sentinel
            # deliberately uses only lowercase callable names.
            answer = f"(__import__('builtins').open({str(marker)!r}, 'w').write('bad'), 1)[1]"
            self.assertFalse(eval_utils.is_equal(answer, "1"))
            self.assertFalse(marker.exists())

    def test_choice_data_cannot_execute_code(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            choices = f"(__import__('pathlib').Path({str(marker)!r}).write_text('bad'), ['red', 'blue'])[1]"
            row = {"choices": choices, "answer": "A", "prediction": "A", "res": "A"}
            self.assertFalse(eval_utils.post_check(row))
            self.assertFalse(marker.exists())

    def test_list_data_cannot_execute_code(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            choices = f"[__import__('pathlib').Path({str(marker)!r}).write_text('bad')]"
            with self.assertRaises(ValueError):
                common_utils.toliststr(choices)
            self.assertFalse(marker.exists())

    def test_arithmetic_and_literal_choices_are_preserved(self):
        for response, answer in [
            ("1/2", "0.5"), ("2 ** 3", "8"), ("-(1 + 2)", "-3"),
            ("1e-3", "0.001"), ("5 % 3", "2"), ("7 // 2", "3"),
        ]:
            with self.subTest(response=response):
                self.assertTrue(eval_utils.is_equal(response, answer))
        self.assertFalse(eval_utils.is_equal("1 / 0", "0"))
        self.assertFalse(eval_utils.is_equal("float('nan')", "0"))
        row = {"choices": "['red', 'blue']", "answer": "B", "prediction": "B", "res": "B"}
        self.assertTrue(eval_utils.post_check(row))
        self.assertEqual(common_utils.toliststr("['a', 2]"), ["a", "2"])
        self.assertEqual(common_utils.toliststr(""), [""])

    @unittest.skipIf(eval_utils.latex2sympy is None, "latex2sympy2 is optional")
    def test_latex_radical_comparison_does_not_eval_sympy_text(self):
        self.assertTrue(eval_utils.is_equal(r"\sqrt{2}", "1.4142135623730951"))
        self.assertFalse(eval_utils.is_equal(r"\sqrt{2}", "1.5"))

    def test_large_integer_arithmetic_keeps_exact_precision(self):
        self.assertFalse(eval_utils.is_equal("9007199254740993", "9007199254740992"))
        self.assertTrue(eval_utils.is_equal("9007199254740993 % 2", "1"))
        self.assertTrue(eval_utils.is_equal("9007199254740993 // 3", "3002399751580331"))
        self.assertFalse(eval_utils.is_equal("9007199254740993 // 3", "3002399751580330"))


if __name__ == "__main__":
    unittest.main()

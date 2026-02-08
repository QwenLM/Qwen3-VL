"""
Qwen3-VL Evaluation Script

A comprehensive evaluation script for Qwen3-VL models with timing metrics,
performance tracking, and batch evaluation support.
"""

import sys
from pathlib import Path

# Add local_transformers to path and redirect transformers imports
_project_root = Path(__file__).parent.parent.resolve()
_local_transformers = _project_root / "local_transformers"
sys.path.insert(0, str(_project_root))

# Import local_transformers as transformers
import local_transformers as transformers

sys.modules["transformers"] = transformers

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run."""

    sample_id: str
    preprocessing_time: float  # Time to process inputs
    generation_time: float  # Time for model generation
    decoding_time: float  # Time to decode outputs
    total_time: float  # Total inference time
    input_tokens: int  # Number of input tokens
    output_tokens: int  # Number of generated tokens
    tokens_per_second: float  # Generation speed
    max_new_tokens: int
    temperature: float
    output_text: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary statistics for evaluation run."""

    total_samples: int
    successful_samples: int
    failed_samples: int
    avg_preprocessing_time: float
    avg_generation_time: float
    avg_decoding_time: float
    avg_total_time: float
    avg_tokens_per_second: float
    total_tokens_generated: int
    total_time: float
    median_generation_time: float
    p95_generation_time: float
    p99_generation_time: float


class Qwen3VLEvaluator:
    """Evaluator class for Qwen3-VL models."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        device_map: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model_id: HuggingFace model identifier or local path
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            dtype: Data type ('float16', 'float32', 'auto'). Auto-selected if None.
            device_map: Device mapping strategy. None means manual device placement.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or ("float16" if self.device == "cuda" else "float32")
        self.device_map = device_map

        print("=== Initializing Qwen3-VL Evaluator ===")
        print(f"Model ID: {model_id}")
        print(f"Device: {self.device}")
        print(f"Data Type: {self.dtype}")

        self.model: Any = None
        self.processor: Any = None
        self._load_model()
        self._load_processor()

    def _load_model(self):
        """Load the model."""
        print("\n=== Loading Model ===")
        start_time = time.time()

        model_kwargs = {
            "dtype": getattr(torch, self.dtype) if self.dtype != "auto" else "auto",
        }

        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        else:
            # Manual device placement
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            model.to(self.device)
            self.model = model

        if not self.device_map:
            self.model.eval()

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    def _load_processor(self):
        """Load the processor."""
        print("\n=== Loading Processor ===")
        start_time = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        load_time = time.time() - start_time
        print(f"Processor loaded in {load_time:.2f} seconds")

    def run_inference(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        sample_id: str = "sample_0",
    ) -> InferenceMetrics:
        """
        Run a single inference.

        Args:
            messages: List of message dictionaries in the expected format
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            sample_id: Identifier for this sample

        Returns:
            InferenceMetrics object with timing and performance data
        """
        total_start = time.time()
        output_text = ""
        error_message = None
        success = True

        try:
            # Preprocessing
            preprocess_start = time.time()
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.device)
            preprocessing_time = time.time() - preprocess_start

            input_tokens = inputs["input_ids"].shape[1]

            # Generation
            generation_start = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            generation_time = time.time() - generation_start

            # Trimming and decoding
            decode_start = time.time()
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            decoding_time = time.time() - decode_start

            output_tokens = len(generated_ids_trimmed[0])
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            total_time = time.time() - total_start

        except Exception as e:
            success = False
            error_message = str(e)
            preprocessing_time = 0
            generation_time = 0
            decoding_time = 0
            total_time = time.time() - total_start
            input_tokens = 0
            output_tokens = 0
            tokens_per_second = 0

        return InferenceMetrics(
            sample_id=sample_id,
            preprocessing_time=preprocessing_time,
            generation_time=generation_time,
            decoding_time=decoding_time,
            total_time=total_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_text=output_text,
            success=success,
            error_message=error_message,
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        show_progress: bool = True,
    ) -> List[InferenceMetrics]:
        """
        Run inference on a batch of test cases.

        Args:
            test_cases: List of dicts with 'sample_id' and 'messages' keys
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            show_progress: Whether to show progress bar

        Returns:
            List of InferenceMetrics objects
        """
        results: List[InferenceMetrics] = []
        iterator = tqdm(test_cases, desc="Evaluating") if show_progress else test_cases

        for test_case in iterator:
            sample_id = test_case.get("sample_id", f"sample_{len(results)}")
            messages = test_case["messages"]

            metrics = self.run_inference(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                sample_id=sample_id,
            )
            results.append(metrics)

        return results

    def compute_summary(self, metrics_list: List[InferenceMetrics]) -> EvaluationSummary:
        """Compute summary statistics from a list of metrics."""
        successful_metrics = [m for m in metrics_list if m.success]
        failed_metrics = [m for m in metrics_list if not m.success]

        if not successful_metrics:
            return EvaluationSummary(
                total_samples=len(metrics_list),
                successful_samples=0,
                failed_samples=len(failed_metrics),
                avg_preprocessing_time=0,
                avg_generation_time=0,
                avg_decoding_time=0,
                avg_total_time=0,
                avg_tokens_per_second=0,
                total_tokens_generated=0,
                total_time=0,
                median_generation_time=0,
                p95_generation_time=0,
                p99_generation_time=0,
            )

        generation_times = [m.generation_time for m in successful_metrics]
        generation_times_sorted = sorted(generation_times)

        return EvaluationSummary(
            total_samples=len(metrics_list),
            successful_samples=len(successful_metrics),
            failed_samples=len(failed_metrics),
            avg_preprocessing_time=statistics.mean([m.preprocessing_time for m in successful_metrics]),
            avg_generation_time=statistics.mean(generation_times),
            avg_decoding_time=statistics.mean([m.decoding_time for m in successful_metrics]),
            avg_total_time=statistics.mean([m.total_time for m in successful_metrics]),
            avg_tokens_per_second=statistics.mean([m.tokens_per_second for m in successful_metrics]),
            total_tokens_generated=sum([m.output_tokens for m in successful_metrics]),
            total_time=sum([m.total_time for m in metrics_list]),
            median_generation_time=statistics.median(generation_times),
            p95_generation_time=generation_times_sorted[int(len(generation_times_sorted) * 0.95)]
            if generation_times_sorted
            else 0,
            p99_generation_time=generation_times_sorted[int(len(generation_times_sorted) * 0.99)]
            if generation_times_sorted
            else 0,
        )


def create_default_test_case() -> Dict[str, Any]:
    """Create a default test case for quick testing."""
    return {
        "sample_id": "demo_sample",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
    }


def save_results(
    metrics_list: List[InferenceMetrics],
    summary: EvaluationSummary,
    output_dir: Path,
    prefix: str = "evaluation",
):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed metrics as JSON
    metrics_file = output_dir / f"{prefix}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in metrics_list], f, indent=2, ensure_ascii=False)
    print(f"\nDetailed metrics saved to: {metrics_file}")

    # Save summary as JSON
    summary_file = output_dir / f"{prefix}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Samples: {summary.total_samples}")
    print(f"Successful: {summary.successful_samples}")
    print(f"Failed: {summary.failed_samples}")
    print("\nTiming Metrics:")
    print(f"  Average Preprocessing Time: {summary.avg_preprocessing_time * 1000:.2f} ms")
    print(f"  Average Generation Time: {summary.avg_generation_time * 1000:.2f} ms")
    print(f"  Average Decoding Time: {summary.avg_decoding_time * 1000:.2f} ms")
    print(f"  Average Total Time: {summary.avg_total_time * 1000:.2f} ms")
    print("\nPerformance Metrics:")
    print(f"  Average Tokens/Second: {summary.avg_tokens_per_second:.2f}")
    print(f"  Total Tokens Generated: {summary.total_tokens_generated}")
    print(f"  Median Generation Time: {summary.median_generation_time * 1000:.2f} ms")
    print(f"  P95 Generation Time: {summary.p95_generation_time * 1000:.2f} ms")
    print(f"  P99 Generation Time: {summary.p99_generation_time * 1000:.2f} ms")
    print(f"  Total Evaluation Time: {summary.total_time:.2f} seconds")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Evaluation Script with Timing Metrics")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (auto = detect automatically)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "auto"],
        default="auto",
        help="Data type (auto = select based on device)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Device mapping strategy (e.g., 'auto', 'balanced'). None = manual placement",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--test-cases-file",
        type=str,
        default=None,
        help="JSON file with test cases (list of dicts with 'sample_id' and 'messages')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of times to run the default test case (if no test cases file provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="evaluation",
        help="Prefix for output files",
    )

    args = parser.parse_args()

    # Initialize evaluator
    device = None if args.device == "auto" else args.device
    dtype = None if args.dtype == "auto" else args.dtype

    evaluator = Qwen3VLEvaluator(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        device_map=args.device_map,
    )

    # Load test cases
    if args.test_cases_file:
        print(f"\n=== Loading Test Cases from {args.test_cases_file} ===")
        with open(args.test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases")
    else:
        print(f"\n=== Using Default Test Case (running {args.num_samples} times) ===")
        test_cases = [create_default_test_case() for _ in range(args.num_samples)]

    # Run evaluation
    print("\n=== Running Evaluation ===")
    metrics_list = evaluator.evaluate_batch(
        test_cases=test_cases,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        show_progress=True,
    )

    # Compute summary
    summary = evaluator.compute_summary(metrics_list)

    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / timestamp

    # Save results
    save_results(metrics_list, summary, output_dir, args.output_prefix)


if __name__ == "__main__":
    main()

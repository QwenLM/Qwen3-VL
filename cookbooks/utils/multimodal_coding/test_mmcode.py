# Standard library imports
import base64
import gzip
import importlib
import json
import multiprocessing
import os
import platform
import re
import signal
import sys
import typing
import faulthandler
from datetime import datetime
from io import BytesIO, StringIO
from unittest.mock import mock_open, patch

# Third-party imports
import numpy as np
from PIL import Image


# from pyext import RuntimeModule
# Fix for higher versions of python
def create_module_from_string(name, code):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(code, module.__dict__)
    return module


from enum import Enum


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)
timeout = 20  # seconds


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def run_test(sample, test=None, debug=True):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError:
        in_outs = None
    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        return in_outs
    elif test is not None:
        results = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                # tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                # An ugly fix for Gemini: it keeps generating code that uses the `__name__`
                sol = sol.replace("__main__", "tmp_sol")
                tmp_sol = create_module_from_string("tmp_sol", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 0 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
            method_name = "code"
            signal.alarm(timeout)
            try:
                # tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                # An ugly fix for Gemini: it keeps generating code that uses the `__name__`
                sol = sol.replace("__main__", "tmp_sol")
                tmp_sol = create_module_from_string("tmp_sol", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 1 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")
            results.append(-2)
            return results

        for index, inputs in enumerate(in_outs["inputs"]):
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
            except:
                True

            if debug:
                print(f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}")
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp_result = output == in_outs["outputs"][index]
                    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                    except:
                        True
                    results.append(tmp_result)

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    print(f"Standard input runtime error or time limit exceeded error = {e}")
                    results.append(-1)
                    continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs["outputs"][index], list):
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                        results.append(-1)
                    signal.alarm(0)

                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                        else:
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    continue

                if passed and debug:
                    print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

                if custom_compare_(output, in_outs["outputs"][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check1 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [x.strip() for x in in_outs["outputs"][index][tmp_index] if x]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                    in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check2 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check3 exception = {e}")
                    pass

                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs["outputs"][index]]
                    tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                        tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                except Exception as e:
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = output == in_outs["outputs"][index]
                except Exception as e:
                    if debug:
                        print(f"Failed check4 exception = {e}")
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = set(frozenset(s) for s in output) == set(frozenset(s) for s in in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check5 exception = {e}")

                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (set(frozenset(round(float(t), 3) for t in s) for s in output) == set(frozenset(round(float(t), 3) for t in s) for s in in_outs["outputs"][index]))
                except Exception as e:
                    if debug:
                        print(f"Failed check6 exception = {e}")

                if tmp_result == True and debug:
                    print("PASSED")

                results.append(tmp_result)

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    else:
                        print(f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

    sys.stdout.flush()
    return results


def custom_compare_(output, ground_truth):

    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    # __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def extract_last_code_block(text):
    """Extract the last Python code block from the text"""
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return None


def check_correctness(sample, generation, timeout=20, debug=True):
    """Check correctness of code generation with a global timeout (from MMCode eval.py)"""

    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        print(f"global timeout")
        if debug:
            print(f"global timeout")
    return result[0]


def evaluate_mmcode_solution(problem, solution_code, timeout=20, debug=True):
    """
    Evaluate a solution using the exact MMCode testing framework
    """
    if not problem or not solution_code:
        print("❌ Missing problem or solution")
        return None

    print(f"\n🧪 TESTING SOLUTION with MMCode framework:")
    print("-" * 50)

    # Extract the Python code from the solution (handling ```python blocks)
    extracted_code = extract_last_code_block(solution_code)
    if extracted_code:
        clean_solution = extracted_code
    else:
        # If no code block found, use the entire solution
        clean_solution = solution_code.strip()

    if not clean_solution:
        print("❌ No valid Python code found in solution")
        return None

    print(f"📝 Testing code...")

    try:
        # Run the test using MMCode's exact testing framework
        curr_res = check_correctness(problem, clean_solution, timeout=timeout, debug=debug)

        if curr_res is None:
            print("❌ Test execution failed")
            return None

        # Fix numpy arrays and booleans (from eval.py)
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed

        # Analyze results
        total_tests = len(curr_res)
        passed_tests = sum(1 for result in curr_res if result == 1)
        failed_tests = sum(1 for result in curr_res if result == 0)
        error_tests = sum(1 for result in curr_res if result == -1)
        timeout_tests = sum(1 for result in curr_res if result == -2)

        print(f"\n📊 DETAILED RESULTS:")
        for i, result in enumerate(curr_res):
            if result == 1:
                print(f"✅ Test {i+1}: PASSED")
            elif result == 0:
                print(f"❌ Test {i+1}: FAILED")
            elif result == -1:
                print(f"💥 Test {i+1}: ERROR")
            elif result == -2:
                print(f"⏱️  Test {i+1}: TIMEOUT")

        print(f"\n📈 SUMMARY:")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"💥 Errors: {error_tests}")
        print(f"⏱️  Timeouts: {timeout_tests}")
        print(f"📋 Total: {total_tests}")

        accuracy = passed_tests / total_tests if total_tests > 0 else 0

        evaluation = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "timeout_tests": timeout_tests,
            "accuracy": accuracy,
            "test_results": curr_res,
            "all_passed": np.all([r == 1 for r in curr_res]),
        }

        print(f"\n🎯 FINAL SCORE: {passed_tests}/{total_tests} ({accuracy:.1%})")

        if evaluation["all_passed"]:
            print(f"🏆 PERFECT: All tests passed!")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: success rate = {accuracy:.1%}")

        return evaluation

    except Exception as e:
        print(f"❌ Evaluation error: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def convert_base64_to_pil_image(base64_str: str) -> Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


def load_mmcode_problems(file_path):
    """Load problems from the downloaded JSONL file and convert images to PIL"""
    problems = []
    try:
        with gzip.open(file_path, "rt") as f:
            for line in f:
                problem = json.loads(line)

                # Convert base64 images to PIL Images (images are guaranteed to exist)
                pil_images = []
                for img_base64 in problem["images"]:
                    try:
                        pil_img = convert_base64_to_pil_image(img_base64)
                        pil_images.append(pil_img)
                    except Exception as e:
                        print(f"Warning: Failed to decode image: {e}")
                        continue
                problem["images"] = pil_images

                problems.append(problem)

        print(f"✅ Loaded {len(problems)} problems")
        return problems
    except Exception as e:
        print(f"❌ Failed to load problems: {e}")
        return []


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="MMCode Evaluation Script")
    parser.add_argument("--problem_file", type=str, default="mmcode_test.jsonl.gz", required=False, help="Path to the JSON file containing problems")
    parser.add_argument("--problem_index", type=int, default=0, help="Index of the problem to evaluate (default: 0)")
    parser.add_argument("--solution_file", type=str, required=True, help="Path to the text file containing the solution code")
    parser.add_argument("--timeout", type=int, default=20, help="Timeout in seconds for each test case (default: 20)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    problems = load_mmcode_problems(args.problem_file)

    evaluate_mmcode_solution(
        problem=problems[args.problem_index],
        solution_code=open(args.solution_file).read(),
    )

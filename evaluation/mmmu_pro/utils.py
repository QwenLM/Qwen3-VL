import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from io_utils import dump, load
from collections.abc import Iterable

def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for key, value in d.items():
        if key == "split":
            items.append((key, value))
        new_key = parent_key + "/" + str(key) if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def print_progress_bar(completed, total, length=40):
    """Simple console-based progress bar."""
    completed_ratio = completed / total
    bar = ('█' * int(length * completed_ratio)).ljust(length)
    percent = int(completed_ratio * 100)
    print(f'\rProgress: |{bar}| {percent}% Complete', end='\r')

def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        use_print_progress=True,  # New parameter to toggle print progress
        **kwargs) -> list:

    if save is not None:
        assert os.path.exists(os.path.dirname(save)) or os.path.dirname(save) == ''
        if not os.path.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = []
    
    start_time = time.time()
    task_count = len(tasks)

    if use_print_progress:
        # Simple custom print-based progress display
        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = []
            for inputs in tasks:
                if not isinstance(inputs, (tuple, list, dict)):
                    inputs = (inputs,)
                if isinstance(inputs, dict):
                    future = executor.submit(func, **inputs)
                else:
                    future = executor.submit(func, *inputs)
                futures.append(future)

            for i, future in enumerate(futures):
                tmp = future.result()
                if keys is not None:
                    res[keys[i]] = tmp
                    if i % 10 == 0 and save is not None:
                        dump(res, save)
                results.append(tmp)
                completed_tasks += 1
                print_progress_bar(completed_tasks, task_count)

    else: 
        # Using Rich for the progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            auto_refresh=True,
        ) as progress:

            task_id = progress.add_task("Processing", total=task_count)

            with ThreadPoolExecutor(max_workers=nproc) as executor:
                futures = []
                for inputs in tasks:
                    if not isinstance(inputs, (tuple, list, dict)):
                        inputs = (inputs,)
                    if isinstance(inputs, dict):
                        future = executor.submit(func, **inputs)
                    else:
                        future = executor.submit(func, *inputs)
                    futures.append(future)

                for i, future in enumerate(futures):
                    tmp = future.result()
                    if keys is not None:
                        res[keys[i]] = tmp
                        if i % 10 == 0 and save is not None:
                            dump(res, save)
                    results.append(tmp)
                    progress.update(task_id, advance=1)

                    # Force update progress every minute
                    if time.time() - start_time > 60:
                        progress.refresh()
                        start_time = time.time()

    if save is not None:
        dump(res, save)
    return results

def cn_string(s):
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False
import json
import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from pathlib import Path


class Timer:
    def __init__(self):
        """
        Creates a timing instance. To be used as:
        ```python
        timer = Timer()

        @timer
        def func_a(...):
            ...

        @timer("custom_label")
        def func_b(...):
            ...

        func_a(...)
        func_b(...)

        timer.report("<path_to_file>.json")
        ```
        In this example, `func_a` will be named `func_a` in the report and `func_b` will be named `custom_label`.
        """
        self._timings: defaultdict = defaultdict(self._new_node)
        self._node_stack: list[defaultdict] = []

    def __call__(self, func_or_label) -> Callable:
        """Use as @timer or @timer('label')."""
        if callable(func_or_label):
            return self._wrap(func_or_label, func_or_label.__qualname__)

        def decorator(func):
            return self._wrap(func, func_or_label)

        return decorator

    def _wrap(self, func, label: str):
        @wraps(func)
        def wrapper(*args, **kwargs):
            parent = self._node_stack[-1] if self._node_stack else self._timings
            node = parent[label]
            self._node_stack.append(node)
            t0 = time.perf_counter_ns()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter_ns() - t0
            self._node_stack.pop()
            node["total_ns"] += elapsed
            node["count"] += 1
            return result

        return wrapper

    def reset(self):
        self._timings = defaultdict(self._new_node)
        self._node_stack = []

    def report(self, path: str | Path, sim_time: float, dt: float):
        def _to_dict(node: dict) -> dict:
            return {
                "total_s" if k == "total_ns" else k: (
                    _to_dict(v) if isinstance(v, dict) else v / 1e9 if k == "total_ns" else v
                )
                for k, v in node.items()
            }

        _times = _to_dict(self._timings)
        general = {"T_sim": sim_time, "T_compute": sum([v["total_s"] for v in _times.values()])}
        general |= {"n_dt/T_compute": (sim_time / dt) / general["T_compute"]}
        Path(path).write_text(json.dumps({"general": general} | _times, indent=2))

    @staticmethod
    def _new_node() -> defaultdict:
        return defaultdict(Timer._new_node, total_ns=0, count=0)


timer = Timer()

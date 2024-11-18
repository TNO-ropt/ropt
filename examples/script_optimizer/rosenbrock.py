"""Example of using the ScriptOptimizer class."""

import argparse
import json
import sys
from pathlib import Path


def rosenbrock(x: float, y: float, realization: int, coefficients_path: Path) -> float:
    """Rosenbrock function."""
    with coefficients_path.open("r", encoding="utf-8") as f:
        coefficients = json.load(f)
    a, b = coefficients[realization]
    return float((a - x) ** 2 + b * (y - x * x) ** 2)


def _read_point(filename: Path) -> tuple[float, float]:
    with filename.open("r", encoding="utf-8") as f:
        variables = json.load(f)
    return variables["x"]["1"], variables["y"]["1"]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--vars", type=Path)
    arg_parser.add_argument("--coefficients", type=Path)
    arg_parser.add_argument("--realization", type=str)
    arg_parser.add_argument("--out", type=Path)
    options, _ = arg_parser.parse_known_args(args=sys.argv[1:])

    x, y = _read_point(options.vars)

    value = rosenbrock(x, y, options.realization, options.coefficients)

    with options.out.open("w", encoding="utf-8") as f:
        f.write(f"{value:.15e}")

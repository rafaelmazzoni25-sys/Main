#!/usr/bin/env python3
"""Generate the EnumModelType.eum file from the C++ enumerations."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def preprocess_enum_header(root: Path) -> str:
    """Run the C++ preprocessor on _enum.h with project defines."""
    include_code = "\n".join(
        [
            "#include \"Defined_Global.h\"",
            "#include \"_define.h\"",
            "#include \"_enum.h\"",
            "",
        ]
    )

    try:
        result = subprocess.run(
            [
                "g++",
                "-E",
                "-x",
                "c++",
                "-I",
                str(root / "source"),
                "-",
            ],
            input=include_code,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise SystemExit("g++ is required to generate EnumModelType.eum") from exc
    return result.stdout


def iter_enum_lines(preprocessed: str) -> List[str]:
    current_file: str | None = None
    lines: List[str] = []
    for raw_line in preprocessed.splitlines():
        if raw_line.startswith("#"):
            parts = raw_line.split()
            if len(parts) >= 3 and parts[0] == "#":
                current_file = parts[2].strip('"')
            continue
        if not current_file or Path(current_file).name != "_enum.h":
            continue
        lines.append(raw_line)
    return lines


def parse_model_enum(lines: List[str]) -> List[Tuple[int, str]]:
    values: Dict[str, int] = {}
    models: List[Tuple[int, int, str]] = []

    pending_enum = False
    in_enum = False
    last_value: int | None = None

    def process_candidate(candidate: str) -> None:
        nonlocal last_value
        candidate = candidate.strip()
        if not candidate:
            return
        candidate = candidate.split("//", 1)[0].strip()
        if not candidate:
            return
        if candidate.endswith(","):
            candidate = candidate[:-1].rstrip()
        if not candidate:
            return
        match = re.match(r"^(?P<name>[A-Za-z_]\w*)(?:\s*=\s*(?P<expr>.+))?$", candidate)
        if not match:
            return
        name = match.group("name")
        expr = match.group("expr")
        if expr is None:
            value = 0 if last_value is None else last_value + 1
        else:
            expr = expr.strip()
            if expr.endswith(","):
                expr = expr[:-1].rstrip()
            try:
                value = int(eval(expr, {"__builtins__": None}, values))
            except NameError as exc:
                raise RuntimeError(
                    f"Unknown identifier in expression '{expr}' for {name}"
                ) from exc
        values[name] = value
        last_value = value
        if name.startswith("MODEL_"):
            models.append((value, len(models), name))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not in_enum and "enum" in stripped:
            pending_enum = True
            last_value = None
            if "{" in stripped:
                in_enum = True
                pending_enum = False
                after = stripped.split("{", 1)[1]
                if after:
                    process_candidate(after)
            continue
        if pending_enum:
            if "{" in stripped:
                in_enum = True
                pending_enum = False
                last_value = None
                after = stripped.split("{", 1)[1]
                if after:
                    process_candidate(after)
            continue
        if in_enum:
            if "}" in stripped:
                before = stripped.split("}", 1)[0]
                process_candidate(before)
                in_enum = False
                last_value = None
                continue
            process_candidate(stripped)

    models.sort(key=lambda item: (item[0], item[1]))
    return [(value, name) for value, _, name in models]


def write_output(models: List[Tuple[int, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        for value, name in models:
            fh.write(f"{value},\"{name}\"\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/EnumModelType.eum"),
        help="Target file to write (default: data/EnumModelType.eum)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    preprocessed = preprocess_enum_header(root)
    enum_lines = iter_enum_lines(preprocessed)
    models = parse_model_enum(enum_lines)
    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(models, output_path)
    print(f"Generated {output_path} with {len(models)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

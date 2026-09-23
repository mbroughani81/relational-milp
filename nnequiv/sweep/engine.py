"""Shared sweep plumbing for the ``*-experiment`` grids.

Both experiments build one command per grid cell and either print the plan
(``--dry-run``, consumed by ``scripts/worker.py``) or run the cells whose result
CSV is missing. Everything that does NOT differ between experiments lives here;
each experiment module supplies only its grid (``iter_cells``) and its per-cell
``build_command``. The emitted commands are still ``python -m benchmarks.run_*``
so the fleet keeps producing the existing (old-schema) result CSVs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# A cell is any tuple whose LAST element is its tag; the preceding elements are
# exactly the positional args the experiment's build_command takes before ``out``.
Cell = tuple
BuildCommand = Callable[..., list[str]]
IterCells = Callable[[], Iterable[Cell]]


def venv_python(repo_root: Path) -> str:
    """The repo's ``.venv`` python if present, else the current interpreter."""
    venv_py = repo_root / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else (sys.executable or "python3")


def load_skip_patterns(path: Path) -> list[str]:
    """Glob patterns from skip.conf (one per line, # comments and blanks ignored)."""
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            patterns.append(line)
    return patterns


def tag_skipped(tag: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(tag, pattern) for pattern in patterns)


def iter_jobs(
    cells: Iterable[Cell],
    build_command: BuildCommand,
    results_dir: Path,
    skip_patterns: list[str],
) -> Iterable[tuple[Path, list[str]]]:
    """Yield (out_path, argv) for every cell not excluded by skip.conf."""
    for cell in cells:
        *cell_args, tag = cell
        if tag_skipped(tag, skip_patterns):
            continue
        out = results_dir / f"{tag}.csv"
        yield out, build_command(*cell_args, out)


def build_plan(
    cells: Iterable[Cell],
    build_command: BuildCommand,
    results_dir: Path,
    skip_file: Path,
    skip_patterns: list[str] | None = None,
) -> list[dict]:
    """The plan as a list of {"result", "command"} objects (skip.conf applied)."""
    if skip_patterns is None:
        skip_patterns = load_skip_patterns(skip_file)
    return [
        {"result": str(out), "command": shlex.join(cmd)}
        for out, cmd in iter_jobs(cells, build_command, results_dir, skip_patterns)
    ]


def run(
    iter_cells: IterCells,
    build_command: BuildCommand,
    results_dir: Path,
    logs_dir: Path,
    skip_file: Path,
    repo_root: Path,
) -> int:
    """Run every not-skipped cell whose result CSV is missing, sequentially."""
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    skip_patterns = load_skip_patterns(skip_file)
    for out, cmd in iter_jobs(iter_cells(), build_command, results_dir, skip_patterns):
        if out.exists():
            print(f"--- skip (exists) {out.name}")
            continue
        log = logs_dir / f"{out.stem}.log"
        print(f">>> {out.stem}")
        with open(log, "w", encoding="utf-8") as log_file:
            rc = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT
            ).returncode
        if rc == 0:
            print(f"    done -> {out.name}")
        else:
            failures += 1
            print(f"    FAILED (rc={rc}); see {log.relative_to(repo_root)}")
    return 1 if failures else 0


@dataclass(frozen=True)
class Experiment:
    name: str
    base_dir: Path
    repo_root: Path
    iter_cells: IterCells
    build_command: BuildCommand

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def skip_file(self) -> Path:
        return self.base_dir / "skip.conf"

    def build_plan(self, skip_patterns: list[str] | None = None) -> list[dict]:
        return build_plan(
            list(self.iter_cells()),
            self.build_command,
            self.results_dir,
            self.skip_file,
            skip_patterns,
        )

    def run(self) -> int:
        return run(
            self.iter_cells,
            self.build_command,
            self.results_dir,
            self.logs_dir,
            self.skip_file,
            self.repo_root,
        )


def run_cli(experiment: Experiment, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"{experiment.name} experiment sweep runner",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan as a JSON array of {result, command} objects and exit",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(experiment.build_plan(), indent=2))
        return 0
    return experiment.run()

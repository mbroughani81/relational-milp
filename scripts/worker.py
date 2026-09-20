#!/usr/bin/env python3
"""Fleet worker for the pruning experiment.

Gets the plan from ``prune-experiment/recreate.py --dry-run`` and runs each cell
whose result CSV is missing and whose lock is unclaimed. Every node runs this
IDENTICAL script; an atomic ``mkdir`` lock on the shared filesystem
self-balances the work, so scaling is just "add more nodes".

Prereqs on each node (see README + setup.sh):
  * RUNTIME_DIR exported (required; base dir for data/third_party/artifacts)
  * shared volume mounted at $SHARED (default /mnt/exp-data)
  * CPLEX subtree copied to $SHARED/ibm/CPLEX_Studio222/cplex (once)
  * ./setup.sh already run

Usage (from repo root):
  scripts/worker.py            # work the queue until this pass drains it
  PROGRESS=1 scripts/worker.py # print queue progress and exit

Env overrides:
  SHARED   shared-FS mount            (default /mnt/exp-data)
  RUN      run name / subdir under it (default prune-10min)
  RECLAIM_STALE_SEC  reclaim a claimed-but-unfinished cell whose lock is older
           than this many seconds (default 0 = never; a milp cell can
           legitimately run for hours, so only enable this for known-dead nodes).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nn_equivalence.paths import runtime_dir  # noqa: E402

RECREATE = REPO_ROOT / "prune-experiment" / "recreate.py"


def get_plan() -> list[dict]:
    """Run recreate.py --dry-run and parse its JSON plan (skip.conf applied)."""
    proc = subprocess.run(
        [sys.executable, str(RECREATE), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def link_to_shared(local_dir: Path, shared_dir: Path) -> None:
    """Point local_dir at shared_dir via symlink, guarding a precious real dir.

    Never clobber a real, non-empty local results/logs dir (those CSVs are
    gitignored and costly). On a fresh clone the dir is empty/absent -> no-op.
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    if local_dir.is_symlink():
        local_dir.unlink()
    elif local_dir.is_dir():
        if any(local_dir.iterdir()):
            sys.exit(
                f"ERROR: {local_dir} is a non-empty real directory; refusing to "
                "replace it\n       (protecting existing result CSVs). Move it "
                "aside or run on a clean clone."
            )
        local_dir.rmdir()
    elif local_dir.exists():
        local_dir.unlink()
    local_dir.symlink_to(shared_dir)


def claim(lock: Path, result: Path, reclaim_stale_sec: int) -> bool:
    """Atomically claim a cell by creating its lock dir. True if we own it now."""
    try:
        lock.mkdir()  # atomic on the shared FS: succeeds for exactly one node
        return True
    except FileExistsError:
        # Already claimed. Optionally reclaim if the owner looks dead.
        if reclaim_stale_sec > 0 and not result.exists():
            age = time.time() - lock.stat().st_mtime
            if age > reclaim_stale_sec:
                print(f"reclaiming stale cell {result.stem} (lock age {int(age)}s)")
                shutil.rmtree(lock, ignore_errors=True)
                try:
                    lock.mkdir()
                    return True
                except FileExistsError:
                    return False
        return False


def main() -> int:
    os.chdir(REPO_ROOT)
    runtime_dir()  # fail fast if RUNTIME_DIR is unset (subprocesses inherit it)

    shared = Path(os.environ.get("SHARED", "/mnt/exp-data"))
    run = os.environ.get("RUN", "prune-10min")
    out = shared / run
    reclaim_stale_sec = int(os.environ.get("RECLAIM_STALE_SEC", "0"))

    if not shared.is_dir():
        sys.exit(f"ERROR: shared FS not mounted at {shared}")
    for sub in ("results", "logs", "queue"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Route recreate.py's results/ and logs/ onto the shared FS via symlinks.
    link_to_shared(REPO_ROOT / "prune-experiment" / "results", out / "results")
    link_to_shared(REPO_ROOT / "prune-experiment" / "logs", out / "logs")

    jobs = get_plan()

    if os.environ.get("PROGRESS") == "1":
        total = len(jobs)
        done = claimed = 0
        for job in jobs:
            result = Path(job["result"])
            if result.exists():
                done += 1
            elif (out / "queue" / f"{result.stem}.lock").is_dir():
                claimed += 1
        print(
            f"run={run}  done={done}/{total}  in-progress={claimed}  "
            f"pending={total - done - claimed}"
        )
        return 0

    host = socket.gethostname()
    print(f"worker {host} starting on queue {out}")

    worked = 0
    for job in jobs:
        result = Path(job["result"])
        if result.exists():  # already have the result? nothing to do.
            continue
        tag = result.stem
        lock = out / "queue" / f"{tag}.lock"
        if not claim(lock, result, reclaim_stale_sec):
            continue
        (lock / "owner").write_text(
            f"{host}\t{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        )

        print(f">>> {host} claimed {tag}")
        log = out / "logs" / f"{tag}.log"
        with open(log, "w", encoding="utf-8") as log_file:
            rc = subprocess.run(
                shlex.split(job["command"]),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            ).returncode
        if rc == 0:
            worked += 1
            print(f"    {tag} finished")
        else:
            print(f"    {tag} command exited non-zero (see {log})")
        # Leave the lock as a done-marker; the result-file check gates reruns.

    print(f"worker {host} drained the queue; ran {worked} cell(s) this pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

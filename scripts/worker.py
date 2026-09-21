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

A cell is also abandoned early if the first TIMEOUT_PROBE_N instances all time
out: a benchmark that hopeless on its first handful of instances will only burn
hours on the rest, so the worker kills it and records a permanent, fleet-wide
skip marker under ``$SHARED/$RUN/skipped/`` (checked before claiming, so no node
or later pass retries it).

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
import re
import shlex
import shutil
import signal
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

# If the first TIMEOUT_PROBE_N instances of a cell all time out, abandon the
# cell early (the rest of the sweep would only time out too).
TIMEOUT_PROBE_N = 10

# Per-instance progress lines streamed by the runners look like
#   "[3/100] mnist_relu_3_100_global_2: status=timeout expected=- runtime_sec=..."
# (see benchmarks.common.print_progress). Capture the instance status.
_PROGRESS_STATUS_RE = re.compile(r"^\[\d+/\d+\].*?\bstatus=(\S+)")


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


def _terminate_group(proc: subprocess.Popen) -> None:
    """Kill the cell's whole process group (SIGTERM, then SIGKILL).

    The subprocess is started in its own session, so this also takes down
    grandchildren the runner spawned (e.g. the external ``cplex`` process).
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):  # up to ~2s for a graceful exit
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_cell(command: list[str], log_file, probe_n: int) -> tuple[str, int | None]:
    """Run one cell, teeing its output to log_file and watching for dead cells.

    Streams the subprocess's combined stdout/stderr into ``log_file`` while
    parsing per-instance ``status=`` lines. If the first ``probe_n`` instances
    all report ``timeout``, kill the process group early and return
    ``("skipped_timeout", None)``. Otherwise return ``("finished", returncode)``.
    """
    proc = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        text=True,
        bufsize=1,
        start_new_session=True,  # own process group -> we can kill children too
    )
    assert proc.stdout is not None
    statuses: list[str] = []
    tripped = False
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        if not tripped and len(statuses) < probe_n:
            match = _PROGRESS_STATUS_RE.match(line)
            if match:
                statuses.append(match.group(1))
                if len(statuses) == probe_n and all(s == "timeout" for s in statuses):
                    tripped = True
                    _terminate_group(proc)  # keep draining the pipe until it closes
    proc.wait()
    if tripped:
        return "skipped_timeout", None
    return "finished", proc.returncode


def main() -> int:
    os.chdir(REPO_ROOT)
    runtime_dir()  # fail fast if RUNTIME_DIR is unset (subprocesses inherit it)

    shared = Path(os.environ.get("SHARED", "/mnt/exp-data"))
    run = os.environ.get("RUN", "prune-10min")
    out = shared / run
    reclaim_stale_sec = int(os.environ.get("RECLAIM_STALE_SEC", "0"))

    if not shared.is_dir():
        sys.exit(f"ERROR: shared FS not mounted at {shared}")
    for sub in ("results", "logs", "queue", "skipped"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Route recreate.py's results/ and logs/ onto the shared FS via symlinks.
    link_to_shared(REPO_ROOT / "prune-experiment" / "results", out / "results")
    link_to_shared(REPO_ROOT / "prune-experiment" / "logs", out / "logs")

    jobs = get_plan()

    if os.environ.get("PROGRESS") == "1":
        total = len(jobs)
        done = skipped = claimed = 0
        for job in jobs:
            result = Path(job["result"])
            tag = result.stem
            if result.exists():
                done += 1
            elif (out / "skipped" / f"{tag}.timeout").exists():
                skipped += 1
            elif (out / "queue" / f"{tag}.lock").is_dir():
                claimed += 1
        print(
            f"run={run}  done={done}/{total}  skipped={skipped}  "
            f"in-progress={claimed}  pending={total - done - skipped - claimed}"
        )
        return 0

    host = socket.gethostname()
    print(f"worker {host} starting on queue {out}")

    worked = skipped = 0
    for job in jobs:
        result = Path(job["result"])
        tag = result.stem
        skip_marker = out / "skipped" / f"{tag}.timeout"
        # already have the result, or a prior node skipped it? nothing to do.
        if result.exists() or skip_marker.exists():
            continue
        lock = out / "queue" / f"{tag}.lock"
        if not claim(lock, result, reclaim_stale_sec):
            continue
        (lock / "owner").write_text(
            f"{host}\t{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        )

        print(f">>> {host} claimed {tag}")
        log = out / "logs" / f"{tag}.log"
        with open(log, "w", encoding="utf-8") as log_file:
            outcome, rc = run_cell(shlex.split(job["command"]), log_file, TIMEOUT_PROBE_N)
        if outcome == "skipped_timeout":
            skipped += 1
            skip_marker.write_text(
                f"{host}\t"
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t"
                f"first {TIMEOUT_PROBE_N}/{TIMEOUT_PROBE_N} instances timed out\n"
            )
            print(
                f"    {tag} SKIPPED: first {TIMEOUT_PROBE_N} instances all timed out"
            )
        elif rc == 0:
            worked += 1
            print(f"    {tag} finished")
        else:
            print(f"    {tag} command exited non-zero (see {log})")
        # Leave the lock as a done-marker; the result/skip check gates reruns.

    print(
        f"worker {host} drained the queue; ran {worked} cell(s), "
        f"skipped {skipped} this pass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
import re


_INTEGER = r"[0-9][0-9,]*"
_FLOAT = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"

_PRESOLVE_TIME_RE = re.compile(
    rf"^\s*Presolve time\s*=\s*({_FLOAT})\s*sec\.?",
    re.MULTILINE,
)
_REDUCED_MODEL_RE = re.compile(
    rf"^\s*Reduced\s+\S+\s+has\s+({_INTEGER})\s+rows?,\s*"
    rf"({_INTEGER})\s+columns?,\s*(?:and\s+)?({_INTEGER})\s+nonzeros?\.",
    re.MULTILINE,
)
_REDUCED_BINARIES_RE = re.compile(
    rf"^\s*Reduced\s+\S+\s+has\s+({_INTEGER})\s+binar(?:y|ies)\b",
    re.MULTILINE,
)
_ELIMINATED_RE = re.compile(
    rf"^\s*(?:MIP\s+)?Presolve eliminated\s+({_INTEGER})\s+rows?\s+and\s+"
    rf"({_INTEGER})\s+columns?\.",
    re.MULTILINE,
)
_ALL_ELIMINATED_RE = re.compile(
    r"^\s*All rows and columns eliminated\.\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CplexPresolveLogStats:
    """Statistics reported by CPLEX during the actual optimization call.

    ``initial_time_sec`` and the reduced-model dimensions correspond to the
    first presolve summary in the CPLEX solve log. CPLEX may report additional
    presolve passes later during root processing; their sum is exposed through
    ``total_reported_time_sec``.
    """

    initial_time_sec: float | None
    total_reported_time_sec: float | None
    time_entries: int
    eliminated_rows: int | None
    eliminated_columns: int | None
    reduced_rows: int | None
    reduced_columns: int | None
    reduced_nonzeros: int | None
    reduced_binary_variables: int | None
    all_rows_and_columns_eliminated: bool


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_cplex_presolve_log(log_text: str) -> CplexPresolveLogStats:
    """Parse initial presolve statistics from an actual CPLEX solve log.

    The parser intentionally uses the text emitted by ``Cplex.solve()``. It
    does not invoke the advanced presolve API and therefore does not perform a
    second presolve pass.
    """

    time_matches = list(_PRESOLVE_TIME_RE.finditer(log_text))
    times = [float(match.group(1)) for match in time_matches]
    initial_time_sec = times[0] if times else None
    total_reported_time_sec = sum(times) if times else None

    # The text before the first "Presolve time" line contains the initial
    # reduction summary. Later parts of the log may contain repeated presolve
    # passes after probing or root processing and must not replace the initial
    # reduced-model dimensions.
    initial_summary_end = time_matches[0].start() if time_matches else len(log_text)
    initial_summary = log_text[:initial_summary_end]

    eliminated_matches = list(_ELIMINATED_RE.finditer(initial_summary))
    eliminated_rows = (
        sum(_parse_int(match.group(1)) for match in eliminated_matches)
        if eliminated_matches
        else None
    )
    eliminated_columns = (
        sum(_parse_int(match.group(2)) for match in eliminated_matches)
        if eliminated_matches
        else None
    )

    all_eliminated = _ALL_ELIMINATED_RE.search(initial_summary) is not None
    if all_eliminated:
        reduced_rows = 0
        reduced_columns = 0
        reduced_nonzeros = 0
        reduced_binary_variables = 0
    else:
        model_match = _REDUCED_MODEL_RE.search(initial_summary)
        binary_match = _REDUCED_BINARIES_RE.search(initial_summary)
        reduced_rows = _parse_int(model_match.group(1)) if model_match else None
        reduced_columns = _parse_int(model_match.group(2)) if model_match else None
        reduced_nonzeros = _parse_int(model_match.group(3)) if model_match else None
        reduced_binary_variables = (
            _parse_int(binary_match.group(1)) if binary_match else None
        )

    return CplexPresolveLogStats(
        initial_time_sec=initial_time_sec,
        total_reported_time_sec=total_reported_time_sec,
        time_entries=len(times),
        eliminated_rows=eliminated_rows,
        eliminated_columns=eliminated_columns,
        reduced_rows=reduced_rows,
        reduced_columns=reduced_columns,
        reduced_nonzeros=reduced_nonzeros,
        reduced_binary_variables=reduced_binary_variables,
        all_rows_and_columns_eliminated=all_eliminated,
    )

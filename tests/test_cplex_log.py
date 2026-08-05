import math

from benchmarks.cplex_log import parse_cplex_presolve_log


def test_parse_initial_reduced_mip_and_repeated_presolve() -> None:
    log = """
Tried aggregator 2 times.
MIP Presolve eliminated 1506 rows and 764 columns.
MIP Presolve modified 308 coefficients.
Reduced MIP has 625 rows, 1310 columns, and 45123 nonzeros.
Reduced MIP has 102 binaries, 0 generals, 0 SOSs, and 0 indicators.
Presolve time = 0.06 sec. (0.96 ticks)
Found incumbent of value 1.0 after 0.07 sec.
Tried aggregator 1 time.
Presolve time = 0.01 sec. (0.88 ticks)
"""

    stats = parse_cplex_presolve_log(log)

    assert stats.initial_time_sec == 0.06
    assert stats.total_reported_time_sec is not None
    assert math.isclose(stats.total_reported_time_sec, 0.07)
    assert stats.time_entries == 2
    assert stats.eliminated_rows == 1506
    assert stats.eliminated_columns == 764
    assert stats.reduced_rows == 625
    assert stats.reduced_columns == 1310
    assert stats.reduced_nonzeros == 45123
    assert stats.reduced_binary_variables == 102
    assert not stats.all_rows_and_columns_eliminated


def test_parse_all_rows_and_columns_eliminated() -> None:
    log = """
MIP Presolve eliminated 4 rows and 2 columns.
Aggregator did 8 substitutions.
All rows and columns eliminated.
Presolve time =    0.00 sec. (0.02 ticks)
"""

    stats = parse_cplex_presolve_log(log)

    assert stats.initial_time_sec == 0.0
    assert stats.reduced_rows == 0
    assert stats.reduced_columns == 0
    assert stats.reduced_nonzeros == 0
    assert stats.reduced_binary_variables == 0
    assert stats.all_rows_and_columns_eliminated


def test_parse_numbers_with_thousands_separators() -> None:
    log = """
MIP Presolve eliminated 1,506 rows and 764 columns.
Reduced MIP has 12,345 rows, 67,890 columns, and 1,234,567 nonzeros.
Reduced MIP has 4,321 binaries, 0 generals, 0 SOSs, and 0 indicators.
Presolve time = 1.25 sec. (10 ticks)
"""

    stats = parse_cplex_presolve_log(log)

    assert stats.eliminated_rows == 1506
    assert stats.reduced_rows == 12345
    assert stats.reduced_columns == 67890
    assert stats.reduced_nonzeros == 1234567
    assert stats.reduced_binary_variables == 4321


def test_missing_presolve_output_returns_none_values() -> None:
    stats = parse_cplex_presolve_log("CPLEX did not report a presolve summary.\n")

    assert stats.initial_time_sec is None
    assert stats.total_reported_time_sec is None
    assert stats.time_entries == 0
    assert stats.reduced_rows is None
    assert stats.reduced_binary_variables is None

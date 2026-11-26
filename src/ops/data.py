"""Inspect and reset the project's data. `reset` only deletes; it never rebuilds.

Usage:
    python -m src.ops.data status
    python -m src.ops.data reset TARGET [TARGET ...] [--include-real] [--yes]

Targets: chunks, eval_logs, user_feedback, eval_set.
`user_feedback` keeps real feedback unless you pass --include-real.
"""

import argparse
import sys
from pathlib import Path

from psycopg.conninfo import conninfo_to_dict

from src.ingestion.db import get_conn, init_schema

EVAL_PATH = Path("data/eval_set.json")
TABLES = ("chunks", "eval_logs", "user_feedback")
TARGETS = (*TABLES, "eval_set")


def db_label(conn) -> str:
    """Return host:port/dbname of the connection, without user or password."""
    info = conninfo_to_dict(conn.info.dsn)
    return f"{info.get('host', 'local')}:{info.get('port', '5432')}/{info.get('dbname', '?')}"


def count_rows(conn, table: str, where: str = "") -> int:
    return conn.execute(f"SELECT count(*) FROM {table} {where};").fetchone()[0]


def status() -> None:
    init_schema()
    with get_conn() as conn:
        print(f"Database: {db_label(conn)}")
        for table in TABLES:
            print(f"{table}: {count_rows(conn, table)} rows")
        per_doc = conn.execute(
            "SELECT doc_name, count(*) FROM chunks GROUP BY doc_name ORDER BY 1;"
        ).fetchall()
        for doc, n in per_doc:
            print(f"  {doc}: {n} chunks")
        real = count_rows(conn, "user_feedback", "WHERE source = 'real'")
        print(f"real feedback: {real} rows")
    print(f"{EVAL_PATH}: {'present' if EVAL_PATH.exists() else 'missing'}")


def describe(conn, target: str, include_real: bool) -> str:
    """Return one line saying what a reset of this target deletes."""
    if target == "eval_set":
        return f"{EVAL_PATH}: {'delete file' if EVAL_PATH.exists() else 'already missing'}"
    if target == "user_feedback" and not include_real:
        n = count_rows(conn, target, "WHERE source = 'simulated'")
        return f"user_feedback: {n} simulated rows (real rows stay)"
    suffix = " (INCLUDING REAL feedback)" if target == "user_feedback" else ""
    return f"{target}: {count_rows(conn, target)} rows{suffix}"


def confirmed() -> bool:
    if not sys.stdin.isatty():
        print("No terminal to confirm on. Pass --yes to delete without a prompt.")
        return False
    return input("Delete? [y/N] ").strip().lower() == "y"


def delete(conn, target: str, include_real: bool) -> None:
    if target == "eval_set":
        EVAL_PATH.unlink(missing_ok=True)
    elif target == "user_feedback" and not include_real:
        conn.execute("DELETE FROM user_feedback WHERE source = 'simulated';")
    else:
        conn.execute(f"TRUNCATE TABLE {target} RESTART IDENTITY;")


def reset(targets: list[str], include_real: bool = False, yes: bool = False) -> bool:
    """Delete the data of each target after one confirmation. Return False if aborted."""
    targets = list(dict.fromkeys(targets))  # drop repeats, keep order
    needs_db = any(t in TABLES for t in targets)
    if needs_db:
        init_schema()
    conn = get_conn() if needs_db else None
    try:
        if conn is not None:
            print(f"Database: {db_label(conn)}")
        print("This deletes:")
        for t in targets:
            print(f"  {describe(conn, t, include_real)}")
        if not (yes or confirmed()):
            print("Aborted. Nothing deleted.")
            return False
        for t in targets:
            delete(conn, t, include_real)
            print(f"Reset {t}.")
    finally:
        if conn is not None:
            conn.close()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ops.data", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show row counts; deletes nothing")
    p = sub.add_parser("reset", help="delete data; rebuilds nothing")
    p.add_argument("targets", nargs="+", choices=TARGETS, metavar="TARGET", help=", ".join(TARGETS))
    p.add_argument("--include-real", action="store_true", help="also delete real user feedback")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    if args.command == "status":
        status()
        return 0
    if args.include_real and "user_feedback" not in args.targets:
        parser.error("--include-real needs the user_feedback target")
    return 0 if reset(args.targets, args.include_real, args.yes) else 1


if __name__ == "__main__":
    sys.exit(main())

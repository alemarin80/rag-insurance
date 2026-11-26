"""Unit tests for src.ops.data, with a fake database (no Postgres needed)."""

from unittest.mock import MagicMock

import pytest

from src.ops import data


@pytest.fixture
def env(monkeypatch, tmp_path):
    conn = MagicMock()
    conn.info.dsn = "host=localhost port=5432 dbname=insurance_rag user=rag"
    conn.execute.return_value.fetchone.return_value = (3,)
    eval_path = tmp_path / "eval_set.json"
    eval_path.write_text("[]")
    monkeypatch.setattr(data, "get_conn", MagicMock(return_value=conn))
    monkeypatch.setattr(data, "init_schema", MagicMock())
    monkeypatch.setattr(data, "EVAL_PATH", eval_path)
    return conn, eval_path


def deletions(conn) -> list[str]:
    sql = [c.args[0] for c in conn.execute.call_args_list]
    return [s for s in sql if s.startswith(("TRUNCATE", "DELETE"))]


def test_no_terminal_aborts(env, monkeypatch):
    conn, eval_path = env
    monkeypatch.setattr(data.sys.stdin, "isatty", lambda: False)
    assert data.main(["reset", "chunks", "eval_set"]) == 1
    assert deletions(conn) == []
    assert eval_path.exists()


def test_answer_other_than_y_aborts(env, monkeypatch):
    conn, _ = env
    monkeypatch.setattr(data.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert data.main(["reset", "eval_logs"]) == 1
    assert deletions(conn) == []


def test_several_targets(env):
    conn, eval_path = env
    assert data.main(["reset", "chunks", "eval_logs", "eval_set", "--yes"]) == 0
    assert deletions(conn) == [
        "TRUNCATE TABLE chunks RESTART IDENTITY;",
        "TRUNCATE TABLE eval_logs RESTART IDENTITY;",
    ]
    assert not eval_path.exists()


def test_feedback_keeps_real_rows(env):
    conn, _ = env
    data.main(["reset", "user_feedback", "--yes"])
    assert deletions(conn) == ["DELETE FROM user_feedback WHERE source = 'simulated';"]


def test_feedback_include_real(env):
    conn, _ = env
    data.main(["reset", "user_feedback", "--include-real", "--yes"])
    assert deletions(conn) == ["TRUNCATE TABLE user_feedback RESTART IDENTITY;"]


def test_eval_set_needs_no_database(env):
    data.main(["reset", "eval_set", "--yes"])
    data.get_conn.assert_not_called()


@pytest.mark.parametrize("argv", [
    ["reset"],
    ["reset", "all"],
    ["reset", "chunks", "--include-real"],
])
def test_bad_arguments(env, argv):
    with pytest.raises(SystemExit):
        data.main(argv)
    assert deletions(env[0]) == []

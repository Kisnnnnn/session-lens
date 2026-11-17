import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import SessionReader
from activity import write_agent_note


@contextmanager
def connect(path):
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        (self.home / "sessions").mkdir()
        self.rollout = self.home / "sessions" / "sample.jsonl"
        self.rollout.write_text("")
        self.now = datetime.now(timezone.utc)
        self.reader = SessionReader(self.home)
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("CREATE TABLE threads (id TEXT, title TEXT, cwd TEXT, rollout_path TEXT, updated_at INTEGER, model TEXT, reasoning_effort TEXT, archived INTEGER, attached_thread_parent_id TEXT)")
            c.execute("INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?)", ("main", "Example\nbody", "/work/project", str(self.rollout), int(self.now.timestamp()), "model", "high", 0, None))
        with connect(self.home / "logs_2.sqlite") as c:
            c.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, ts INTEGER, level TEXT, thread_id TEXT, feedback_log_body TEXT)")

    def at(self, seconds=0):
        return (self.now + timedelta(seconds=seconds)).isoformat()

    def event(self, kind, seconds=0, **extra):
        return {"timestamp": self.at(seconds), "type": "event_msg", "payload": {"type": kind, **extra}}

    def write(self, *items, mode="w"):
        with self.rollout.open(mode) as output:
            for item in items:
                output.write(json.dumps(item) + "\n")

    def session(self):
        return self.reader.snapshot()["sessions"][0]

    def capacity(self, seconds=-10):
        with connect(self.home / "logs_2.sqlite") as c:
            c.execute("INSERT INTO logs(ts,level,thread_id,feedback_log_body) VALUES (?,?,?,?)", (int((self.now + timedelta(seconds=seconds)).timestamp()), "WARN", "main", 'secret prompt never expose: stream disconnected - retrying sampling request (1/5 in 326s)... sampling_error=Selected model is at capacity.'))

    def test_current_capacity_has_deadline_but_completion_wins(self):
        self.write(self.event("task_started", -30))
        self.capacity()
        status = self.session()["status"]
        self.assertEqual(status["code"], "retry_wait")
        self.assertIsNotNone(status["retryAt"])
        self.write(self.event("task_complete", -1), mode="a")
        session = self.session()
        self.assertEqual(session["status"]["code"], "completed")
        self.assertTrue(any(event["kind"] == "capacity" for event in session["events"]))
        self.assertNotIn("secret prompt", json.dumps(session))

    def test_expired_backoff_is_not_current_retry(self):
        self.write(self.event("task_started", -2000))
        self.capacity(-1000)
        self.assertEqual(self.session()["status"]["code"], "unknown")

    def test_later_progress_overrides_capacity(self):
        self.write(self.event("task_started", -30))
        self.capacity()
        self.write({"timestamp": self.at(-1), "type": "response_item", "payload": {"type": "function_call", "call_id": "call", "name": "read_file", "arguments": "SECRET"}}, mode="a")
        snapshot = self.session()
        self.assertEqual(snapshot["status"]["code"], "tool_running")
        self.assertNotIn("SECRET", json.dumps(snapshot))

    def test_current_input_tokens_are_not_cumulative_usage(self):
        self.write(self.event("token_count", info={"last_token_usage": {"input_tokens": 25000}, "total_token_usage": {"input_tokens": 5000000}, "model_context_window": 100000}))
        context = self.session()["context"]
        self.assertEqual(context["inputTokens"], 25000)
        self.assertEqual(context["percent"], 25)

    def test_partial_last_line_and_invalid_line(self):
        self.write(self.event("task_started", -10))
        tail = json.dumps(self.event("task_complete", -1))
        with self.rollout.open("a") as output:
            output.write("invalid JSON\n" + tail[:20])
        self.assertEqual(self.session()["status"]["code"], "running")
        with self.rollout.open("a") as output:
            output.write(tail[20:] + "\n")
        self.assertEqual(self.session()["status"]["code"], "completed")

    def test_truncation_replaces_cached_terminal_and_context(self):
        self.write(self.event("token_count", info={"last_token_usage": {"input_tokens": 200}, "model_context_window": 1000}), self.event("task_complete"))
        self.assertEqual(self.session()["status"]["code"], "completed")
        self.write(self.event("task_started", 1))
        session = self.session()
        self.assertEqual(session["status"]["code"], "running")
        self.assertIsNone(session["context"]["inputTokens"])

    def test_rewrite_with_larger_size_resets_cache(self):
        self.write(self.event("task_complete"))
        self.assertEqual(self.session()["status"]["code"], "completed")
        self.write(self.event("task_started", 1, other="x" * 300))
        self.assertEqual(self.session()["status"]["code"], "running")

    def test_missing_home_and_schema_degrade(self):
        self.assertEqual(SessionReader(self.home / "absent").snapshot()["source"]["status"], "unavailable")
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("DROP TABLE threads")
            c.execute("CREATE TABLE threads(unexpected TEXT)")
        self.assertEqual(self.reader.snapshot()["source"]["status"], "unavailable")

    def test_outside_path_and_missing_format_remain_unknown(self):
        external = self.home / "outside.jsonl"
        external.write_text(json.dumps(self.event("task_complete")) + "\n")
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("UPDATE threads SET rollout_path=?", (str(external),))
        self.assertEqual(self.session()["status"]["code"], "unknown")
        self.assertEqual(self.reader._cache, {})

    def test_child_is_attached_not_a_top_level_session(self):
        child = self.home / "sessions" / "child.jsonl"
        child.write_text(json.dumps(self.event("task_complete")) + "\n")
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?)", ("child", "Child", "/work/project", str(child), int(self.now.timestamp()), "model", "low", 0, "main"))
        snapshot = self.reader.snapshot()
        self.assertEqual(len(snapshot["sessions"]), 1)
        self.assertEqual(snapshot["sessions"][0]["agents"][0]["status"], "completed")

    def test_explicit_agent_description_is_attached_to_its_parent_only(self):
        child = self.home / "sessions" / "child.jsonl"
        child.write_text(json.dumps(self.event("task_complete")) + "\n")
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?)", ("child", "Child", "/work/project", str(child), int(self.now.timestamp()), "model", "low", 0, "main"))
        notes = self.home / "artifacts" / "codex-session-helper" / "agent-notes.json"
        write_agent_note(notes, "main", "child", "界面与交互", "负责按钮与交互说明。")
        write_agent_note(notes, "another-parent", "child", "其他分工", "不得混入当前会话。")
        agent = self.session()["agents"][0]
        self.assertEqual(agent["displayName"], "界面与交互")
        self.assertEqual(agent["description"], "负责按钮与交互说明。")
        self.assertEqual(agent["noteSource"], "主聊天提供的分工说明")

    def test_tool_result_is_not_reported_as_a_completed_requirement(self):
        self.write(self.event("task_started", -10),
                   {"timestamp": self.at(-9), "type": "response_item", "payload": {"type": "function_call", "call_id": "test", "name": "exec_command", "arguments": json.dumps({"cmd": "python3 -m unittest discover"})}},
                   {"timestamp": self.at(-8), "type": "response_item", "payload": {"type": "function_call_output", "call_id": "test", "output": json.dumps({"exit_code": 0, "output": "SECRET_OUTPUT"})}},
                   self.event("task_complete", -7))
        session = self.session()
        event = next(e for e in session["events"] if e["kind"] == "tool")
        self.assertEqual(event["title"], "运行测试")
        self.assertEqual(event["status"], "returned")
        self.assertNotIn("SECRET_OUTPUT", json.dumps(session))
        terminal = next(e for e in session["events"] if e["kind"] == "task_complete")
        self.assertEqual(terminal["title"], "本轮处理结束")
        self.assertIn("是否完成整个需求", terminal["detail"])

    def test_incremental_read_does_not_duplicate_compactions(self):
        self.write({"timestamp": self.at(), "type": "compacted", "payload": {}})
        first = self.session()
        offset = self.reader._cache[str(self.rollout.resolve())]["offset"]
        second = self.session()
        self.assertEqual(len(first["compactions"]), 1)
        self.assertEqual(len(second["compactions"]), 1)
        self.assertEqual(self.reader._cache[str(self.rollout.resolve())]["offset"], offset)

    def test_paginated_unavailable_recent_history_does_not_claim_completed(self):
        self.write(self.event("task_complete", -100))
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("ALTER TABLE threads ADD COLUMN history_mode TEXT")
            c.execute("UPDATE threads SET history_mode='paginated'")
        session = self.session()
        self.assertEqual(session["status"]["code"], "unknown")
        self.assertIn("分页", session["status"]["label"])

    def test_absent_timestamp_does_not_create_fake_event(self):
        self.write({"type": "event_msg", "payload": {"type": "task_complete"}})
        self.assertEqual(self.session()["status"]["code"], "unknown")

    def add_projects(self):
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("ALTER TABLE threads ADD COLUMN project_id TEXT")
            c.execute("CREATE TABLE projects(id TEXT, name TEXT, position INTEGER)")
            c.execute("CREATE TABLE project_roots(project_id TEXT, path TEXT, position INTEGER)")
            c.executemany("INSERT INTO projects VALUES (?,?,?)", [("a", "共享开发工作区", 0), ("b", "独立项目", 1), ("empty", "无近期会话项目", 2)])
            c.executemany("INSERT INTO project_roots VALUES (?,?,?)", [("a", "/work/project", 0), ("a", "/work/other", 1), ("b", "/work/project", 0)])

    def test_saved_project_identity_distinguishes_shared_directories(self):
        self.add_projects()
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("UPDATE threads SET project_id='a'")
            c.execute("INSERT INTO threads(id,title,cwd,rollout_path,updated_at,archived,project_id) VALUES (?,?,?,?,?,?,?)",
                      ("other", "Other", "/work/project", str(self.rollout), int(self.now.timestamp()), 0, "b"))
        data = self.reader.snapshot()
        by_name = {p["name"]: p for p in data["projects"]}
        self.assertEqual(by_name["共享开发工作区"]["count"], 1)
        self.assertEqual(by_name["独立项目"]["count"], 1)
        self.assertEqual(by_name["无近期会话项目"]["count"], 0)
        self.assertNotEqual(by_name["共享开发工作区"]["key"], by_name["独立项目"]["key"])

    def test_legacy_project_assignment_and_explicit_projectless(self):
        self.add_projects()
        state = {"thread-project-assignments": {"main": {"projectKind": "local", "projectId": "legacy-a"}},
                 "app-server-project-id-by-legacy-project-id-by-host": {"local:" + str(self.home): {"legacy-a": "a"}}}
        state_path = self.home / ".codex-global-state.json"
        state_path.write_text(json.dumps(state))
        self.assertEqual(self.session()["project"], "共享开发工作区")
        state["projectless-thread-ids"] = ["main"]
        state_path.write_text(json.dumps(state))
        self.assertEqual(self.session()["projectKey"], "unassigned")

    def test_unassigned_temporary_directories_are_one_group(self):
        self.add_projects()
        with connect(self.home / "state_5.sqlite") as c:
            c.execute("INSERT INTO threads(id,title,cwd,updated_at,archived) VALUES (?,?,?,?,?)",
                      ("temp", "Temporary task", "/tmp/new-chat-123", int(self.now.timestamp()), 0))
        data = self.reader.snapshot()
        unassigned = [p for p in data["projects"] if p["key"] == "unassigned"]
        self.assertEqual(len(unassigned), 1)
        self.assertEqual(unassigned[0]["count"], 2)
        self.assertTrue(all(s["projectId"] is None for s in data["sessions"]))


if __name__ == "__main__":
    unittest.main()

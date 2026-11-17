import json
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activity import classify_activity, group_activities, read_agent_notes, write_agent_note


class ClassifyTests(unittest.TestCase):
    def wrapped(self, source):
        return classify_activity({"type": "custom_tool_call", "name": "exec", "namespace": "functions", "input": source})

    def direct(self, command):
        return classify_activity({"name": "exec_command", "arguments": json.dumps({"cmd": command})})

    def test_unknown_and_encrypted_data_are_never_displayed(self):
        encrypted = "gAAAA_THIS_IS_ENCRYPTED_NOT_A_TASK_LABEL"
        for payload in (
            {"name": "exec", "namespace": "functions", "input": encrypted},
            {"name": "unknown_tool", "arguments": json.dumps({"message": encrypted, "cmd": "secret-command-with-token"})},
            {"name": "spawn_agent", "arguments": json.dumps({"message": encrypted})},
        ):
            result = classify_activity(payload)
            self.assertNotIn(encrypted, json.dumps(result))
            self.assertNotIn("secret-command", json.dumps(result))
        self.assertEqual(classify_activity({"name": "unknown_tool"})["title"], "使用辅助工具")
        self.assertEqual(self.wrapped(encrypted)["title"], "执行本地操作")
        self.assertEqual(classify_activity({"name": "followup_task", "arguments": "invalid encrypted"})["title"], "分派协作任务")

    def test_wrapped_commands_and_known_tools_are_classified(self):
        cases = [
            ('await tools.exec_command({cmd: "cat /secret/file"})', "read_files"),
            ('await tools.exec_command({cmd: "python3 -m unittest discover -s tests"})', "tests"),
            ('await tools.exec_command({cmd: "node qa-session.cjs"})', "tests"),
            ('await tools.exec_command({cmd: "git -C /project diff"})', "git"),
            ('await tools.exec_command({cmd: "codex plugin add /tmp/plugin"})', "plugin"),
            ('await tools.apply_patch("SECRET PATCH")', "edit_files"),
            ('await tools.mcp__codex_app__list_projects({})', "read_projects"),
            ('await tools.exec_command({cmd: "python3 - <<\'PY\'\\nimport sqlite3\\nsqlite3.connect(\'secret.sqlite\')\\nPY"})', "local_records"),
            ('await tools.exec_command({cmd: "python3 - <<\'PY\'\\nfrom pathlib import Path\\nPath(\'secret\').write_text(\'secret output\')\\nPY"})', "edit_files"),
            ('const browser = await chromium.launch(); await page.screenshot({path:"secret.png"})', "inspect_ui"),
        ]
        for source, expected in cases:
            with self.subTest(expected=expected):
                result = self.wrapped(source)
                self.assertEqual(result["category"], expected)
                self.assertNotIn("/secret", result["detail"])
                self.assertNotIn("secret output", json.dumps(result))

    def test_direct_json_commands_and_quotes_do_not_invent_operations(self):
        self.assertEqual(self.direct("npm test")["category"], "tests")
        self.assertEqual(self.direct("echo 'pytest'")["category"], "local")
        self.assertEqual(self.wrapped('const text = "tools.apply_patch(\\\"fake\\\")";')["category"], "local")
        self.assertEqual(self.wrapped('// tools.apply_patch("fake")\nconst n = 1;')["category"], "local")
        self.assertEqual(self.wrapped('await tools.exec_command({cmd: `cat ${unresolvedPath}`})')["category"], "local")
        self.assertEqual(self.direct('python3 - <<\'PY\'\nprint("sqlite3.connect")\nPY')["category"], "local")

    def test_mixed_is_explicit_and_does_not_claim_test_success(self):
        result = self.wrapped('await tools.exec_command({cmd:"pytest"}); await tools.apply_patch("secret")')
        self.assertIn("修改文件", result["title"])
        self.assertIn("运行测试", result["title"])
        self.assertIn("运行测试", result["detail"])
        self.assertIn("修改文件", result["detail"])
        self.assertNotIn("通过", result["detail"])
        self.assertEqual(self.direct("cat file\nnpm test")["category"], "mixed")
        self.assertEqual(self.direct("git status && unknown-command")["category"], "mixed")


class GroupTests(unittest.TestCase):
    def event(self, second, category="tests", status="returned", kind="tool"):
        at = (datetime(2026, 11, 16, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat()
        return {"at": at, "kind": kind, "category": category, "status": status, "toolName": "functions.exec", "callId": "call-" + str(second), "title": "执行多项本地操作", "detail": "RAW COMMAND MUST NOT APPEAR"}

    def test_adjacent_same_category_and_status_are_grouped_newest_first(self):
        grouped = group_activities([self.event(0), self.event(120), self.event(240)])
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["count"], 3)
        self.assertEqual(grouped[0]["at"], self.event(240)["at"])
        self.assertEqual(grouped[0]["firstAt"], self.event(0)["at"])
        self.assertEqual(grouped[0]["statusCounts"], {"returned": 3})
        self.assertEqual(len(grouped[0]["callIds"]), 3)
        self.assertNotIn("RAW COMMAND", json.dumps(grouped))

    def test_category_status_and_terminal_boundaries_are_preserved(self):
        self.assertEqual(len(group_activities([self.event(0), self.event(1, category="git")])), 2)
        self.assertEqual(len(group_activities([self.event(0), self.event(1, status="failed")])), 2)
        for boundary in ("started", "task_complete", "capacity", "turn_aborted"):
            grouped = group_activities([self.event(0), self.event(1, kind=boundary), self.event(2)])
            self.assertEqual(len(grouped), 3)
            self.assertEqual(grouped[1]["kind"], boundary)

    def test_time_gap_and_total_group_duration_are_bounded(self):
        self.assertEqual(len(group_activities([self.event(0), self.event(181)])), 2)
        grouped = group_activities([self.event(second) for second in (0, 120, 240, 360, 480, 600, 720)])
        self.assertEqual([entry["count"] for entry in grouped], [6, 1])
        for entry in grouped:
            start = datetime.fromisoformat(entry["firstAt"])
            end = datetime.fromisoformat(entry["at"])
            self.assertLessEqual((end - start).total_seconds(), 600)

    def test_missing_dates_do_not_merge_and_no_success_inference(self):
        first = self.event(0)
        second = self.event(1)
        first["at"] = None
        second["at"] = "bad time"
        grouped = group_activities([first, second])
        self.assertEqual(len(grouped), 2)
        self.assertNotIn("通过", grouped[0]["detail"])

    def test_repeated_connection_records_keep_count_and_time_range(self):
        events = [{"at": self.event(second)["at"], "kind": "connection", "title": "连接曾中断"} for second in (0, 60, 120)]
        grouped = group_activities(events)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["count"], 3)
        self.assertEqual(grouped[0]["firstAt"], self.event(0)["at"])
        self.assertEqual(grouped[0]["at"], self.event(120)["at"])

    def test_mixed_activity_keeps_known_actions_after_grouping(self):
        classified = classify_activity({"name": "exec", "input": 'await tools.exec_command({cmd:"pytest"}); await tools.apply_patch("SECRET")'})
        events = [{**self.event(second), **classified} for second in (0, 60)]
        grouped = group_activities(events)
        self.assertEqual(grouped[0]["count"], 2)
        self.assertIn("运行测试", grouped[0]["title"])
        self.assertIn("修改文件", grouped[0]["detail"])
        self.assertNotIn("SECRET", json.dumps(grouped))


class NotesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "private" / "notes.json"

    def test_persistence_and_thread_identity(self):
        saved = write_agent_note(self.path, "thread-a", "/root/reviewer", "检查兼容性", "检查最低版本并返回问题证据")
        self.assertEqual(saved["source"], "主聊天提供的分工说明")
        write_agent_note(self.path, "thread-b", "/root/reviewer", "检查接口", "核对接口")
        notes = read_agent_notes(self.path)
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes["thread-a:/root/reviewer"]["label"], "检查兼容性")
        self.assertEqual(notes["thread-b:/root/reviewer"]["label"], "检查接口")
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        self.assertEqual(list(self.path.parent.glob(".agent-notes-*")), [])

    def test_note_validation_and_bad_file_degrade(self):
        for arguments in (("thread", "agent", "x" * 41, "description"), ("thread", "agent", "label", "x" * 161), ("", "agent", "label", "description"), ("thread", "agent", "", "description")):
            with self.assertRaises(ValueError):
                write_agent_note(self.path, *arguments)
        self.path.parent.mkdir()
        self.path.write_text("not JSON")
        self.assertEqual(read_agent_notes(self.path), {})
        self.path.write_text(json.dumps({"fake": {"sourceThreadId": "t", "agentId": "a", "label": "未经主聊天提供的猜测", "description": "", "source": "inferred from ciphertext"}}))
        self.assertEqual(read_agent_notes(self.path), {})


if __name__ == "__main__":
    unittest.main()

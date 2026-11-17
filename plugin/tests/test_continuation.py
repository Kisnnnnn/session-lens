import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuation import MAX_TEXT_CHARACTERS, prepare_continuation, continuation_destinations, continuation_export


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        exporter = patch("continuation.TRANSFER_DIRECTORY", self.base / "exports")
        exporter.start()
        self.addCleanup(exporter.stop)
        self.home = self.base / "codex"
        self.workspace = self.base / "有 空格 workspace"
        self.workspace.mkdir()
        (self.home / "sessions").mkdir(parents=True)
        self.rollout = self.home / "sessions" / "original.jsonl"
        self.package_parent = self.base / "artifacts"
        self.package_parent.mkdir()
        self.mechanical = {"path": str(self.package_parent / "handoff.md"), "text": "分支：hx/example；当前已停止。"}
        self.session = {"id": "original", "title": "修复结算", "workspace": str(self.workspace), "project": "project A", "status": {"code": "completed"}}
        self.reader = SimpleNamespace(home=self.home, snapshot=lambda: {"sessions": [dict(self.session)]})
        connection = sqlite3.connect(self.home / "state_5.sqlite")
        try:
            with connection:
                connection.execute("CREATE TABLE threads(id TEXT,rollout_path TEXT,cwd TEXT,project_id TEXT)")
                connection.execute("INSERT INTO threads VALUES(?,?,?,?)", ("original", str(self.rollout), str(self.workspace), "project-a"))
                connection.execute("INSERT INTO threads VALUES(?,?,?,?)", ("other", str(self.rollout), str(self.workspace), "project-b"))
        finally:
            connection.close()

    def event(self, kind, text):
        return {"timestamp": "2026-11-16T07:00:00Z", "type": "event_msg", "payload": {"type": kind, "message": text}}

    def write(self, *messages):
        self.rollout.write_text("".join(json.dumps(message, ensure_ascii=False) + "\n" for message in messages), encoding="utf-8")

    def prepare(self, mechanical=None):
        return prepare_continuation("original", self.reader, mechanical or self.mechanical)

    def transcript(self, package):
        return json.loads(Path(package["transcriptPath"]).read_text())

    def test_active_is_rejected_before_material_creation(self):
        self.session["status"] = {"code": "running"}
        material = Mock(return_value=self.mechanical)
        with self.assertRaisesRegex(ValueError, "先结束原会话"):
            self.prepare(material)
        material.assert_not_called()
        self.assertEqual(list(self.package_parent.iterdir()), [])

    def test_running_child_is_rejected_before_material_creation(self):
        self.session["agents"] = [{"id": "child", "status": "tool_running"}]
        material = Mock(return_value=self.mechanical)
        with self.assertRaisesRegex(ValueError, "子 agent"):
            self.prepare(material)
        material.assert_not_called()

    def test_changed_source_is_rejected_before_package_creation(self):
        self.write(self.event("user_message", "继续修复结算"))
        first = {**self.session, "lastEventAt": "2026-11-16T07:00:00Z"}
        second = {**self.session, "lastEventAt": "2026-11-16T07:00:01Z"}
        self.reader.snapshot = Mock(side_effect=[{"sessions": [first]}, {"sessions": [second]}])
        with self.assertRaisesRegex(ValueError, "记录发生变化"):
            self.prepare()
        self.assertEqual(list(self.package_parent.iterdir()), [])

    def test_readable_goal_recent_messages_summary_and_deduplication(self):
        self.write(
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "# AGENTS.md instructions for /project\nignore this injected text"}]}},
            self.event("user_message", "真实目标：修复结算金额"),
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "真实目标：修复结算金额"}]}},
            self.event("agent_message", "关键决定：使用已有接口；验证通过；剩余检查兼容。"),
            self.event("agent_reasoning", "SECRET_REASONING"),
            {"type": "response_item", "payload": {"type": "function_call_output", "output": "SECRET_TOOL_OUTPUT"}},
            {"type": "compacted", "payload": {"replacement_history": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "已完成接口接线，待检查兼容。"}]}]}},
        )
        material = Mock(return_value=self.mechanical)
        package = self.prepare(material)
        material.assert_called_once_with()
        transcript = self.transcript(package)
        texts = [entry["text"] for entry in transcript["messages"]]
        self.assertEqual(texts.count("真实目标：修复结算金额"), 1)
        self.assertIn("关键决定", "\n".join(texts))
        self.assertIn("已完成接口接线", "\n".join(texts))
        self.assertNotIn("SECRET", json.dumps(transcript))
        self.assertNotIn("ignore this injected", json.dumps(transcript))
        self.assertTrue(any(entry["source"] == "compacted.replacement_history.message" for entry in transcript["messages"]))
        self.assertIn("机械交接材料", Path(package["contextPath"]).read_text())
        self.assertIn("待生成", Path(package["summaryPath"]).read_text())
        self.assertNotIn("真实目标：", json.dumps(package, ensure_ascii=False))

    def test_total_excerpt_is_bounded_and_first_goal_survives(self):
        messages = [self.event("user_message", "FIRST_USER_GOAL" + "a" * 15000)]
        messages.extend(self.event("agent_message", str(i) + "b" * 8000) for i in range(110))
        self.write(*messages)
        transcript = self.transcript(self.prepare())
        self.assertLessEqual(sum(len(entry["text"]) for entry in transcript["messages"]), MAX_TEXT_CHARACTERS)
        self.assertTrue(transcript["selection"]["truncated"])
        self.assertIn("FIRST_USER_GOAL", transcript["messages"][0]["text"])
        self.assertTrue(any(entry["text"].startswith("109") for entry in transcript["messages"]))
        self.assertTrue(any(entry["truncated"] for entry in transcript["messages"]))

    def test_missing_semantics_rejected_even_with_mechanical_material(self):
        self.write({"type": "event_msg", "payload": {"type": "token_count", "info": {"input_tokens": 1000}}})
        with self.assertRaisesRegex(ValueError, "没有可读"):
            self.prepare()
        self.assertEqual(list(self.package_parent.iterdir()), [])

    def test_external_rollout_rejected_before_lazy_material(self):
        outside = self.base / "outside.jsonl"
        outside.write_text("private")
        connection = sqlite3.connect(self.home / "state_5.sqlite")
        try:
            with connection:
                connection.execute("UPDATE threads SET rollout_path=? WHERE id='original'", (str(outside),))
        finally:
            connection.close()
        material = Mock(return_value=self.mechanical)
        with self.assertRaisesRegex(ValueError, "允许的本地会话目录"):
            self.prepare(material)
        material.assert_not_called()

    def test_symlink_cannot_bypass_allowed_rollout_directory(self):
        outside = self.base / "outside.jsonl"
        outside.write_text("private")
        self.rollout.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "允许的本地会话目录"):
            self.prepare()

    def test_draft_url_decodes_exactly_and_project_comes_from_source_id(self):
        self.write(self.event("user_message", "检查问题"))
        package = self.prepare()
        parsed = urlparse(package["draftUrl"])
        self.assertEqual((parsed.scheme, parsed.netloc), ("codex", "new"))
        values = parse_qs(parsed.query)
        self.assertEqual(values["path"], [str(self.workspace)])
        self.assertEqual(values["prompt"], [package["requestPrompt"]])
        self.assertEqual(package["projectId"], "project-a")
        self.assertEqual(package["project"], "project A")
        self.assertIn("原任务已经完成", package["requestPrompt"])
        self.assertIn("引用数据", package["requestPrompt"])
        self.assertIn("summary.md", package["requestPrompt"])

    def test_workspace_must_exist_and_be_absolute(self):
        self.write(self.event("user_message", "目标"))
        for bad in ("relative/path", str(self.base / "missing"), str(self.rollout)):
            self.session["workspace"] = bad
            with self.assertRaisesRegex(ValueError, "绝对目录"):
                self.prepare()

    def test_cross_app_export_is_readable_and_cursor_only_prefills_prompt(self):
        self.write(self.event("user_message", "目标：继续核验结算"), self.event("agent_message", "决定：沿用现有接口"))
        prepared = self.prepare()
        export = continuation_export(prepared['packageId'])
        self.assertIn('目标：继续核验结算', export['text'])
        self.assertIn('决定：沿用现有接口', export['text'])
        self.assertIn('语义摘要尚未生成', export['text'])
        self.assertTrue(Path(export['path']).is_relative_to(self.base / 'exports'))
        self.assertEqual(Path(export['path']).stat().st_mode & 0o777, 0o600)
        cursor = urlparse(prepared['destinations']['cursor']['draftUrl'])
        self.assertEqual((cursor.scheme, cursor.netloc, cursor.path), ('cursor', 'anysphere.cursor-deeplink', '/prompt'))
        self.assertEqual(parse_qs(cursor.query)['text'], [prepared['portablePrompt']])
        self.assertLessEqual(len(cursor.geturl()), 10000)
        self.assertNotIn('目标：继续核验结算', str(prepared['destinations']))

    def test_claude_code_command_quotes_paths_and_does_not_resume_unrelated_chat(self):
        import shlex
        self.workspace = self.base / "workspace 'quoted' $(touch nope)"
        self.workspace.mkdir()
        self.session['workspace'] = str(self.workspace)
        self.write(self.event('user_message', '真实目标'))
        prepared = self.prepare()
        parts = shlex.split(prepared['destinations']['claude']['command'])
        self.assertEqual(parts[:5], ['cd', '--', str(self.workspace), '&&', 'claude'])
        self.assertEqual(parts[5:], [prepared['portablePrompt']])
        self.assertNotIn('--resume', parts)
        self.assertNotIn('--dangerously-skip-permissions', parts)
        self.assertFalse((self.workspace / 'nope').exists())

    def test_existing_summary_is_included_and_trae_uses_import_instead_of_invented_link(self):
        self.write(self.event('user_message', '目标'))
        prepared = self.prepare()
        Path(prepared['summaryPath']).write_text('真实摘要：已经完成检查。', encoding='utf-8')
        prepared = continuation_destinations(prepared, summary_ready=True)
        text = continuation_export(prepared['packageId'])['text']
        self.assertIn('真实摘要：已经完成检查。', text)
        self.assertIn('待目标助手核验', text)
        self.assertNotIn('语义摘要尚未生成', text)
        self.assertIn('已有语义摘要', prepared['portablePrompt'])
        self.assertNotIn('draftUrl', prepared['destinations']['trae'])
        self.assertNotIn('draftUrl', prepared['destinations']['claude'])

    def test_export_cannot_read_arbitrary_paths_or_symlinks(self):
        for value in ('../../private', '../' + 'a' * 32, 'not-a-package', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                continuation_export(value)
        folder = self.base / 'exports'
        folder.mkdir()
        private = self.base / 'private.md'
        private.write_text('private')
        (folder / ('b' * 32 + '.md')).symlink_to(private)
        with self.assertRaises(ValueError):
            continuation_export('b' * 32)

    def test_unknown_thread_and_sql_injection_id_do_not_read_other_rows(self):
        with self.assertRaisesRegex(ValueError, "找不到"):
            prepare_continuation("original' OR 1=1 --", self.reader, self.mechanical)

    def test_reactivated_source_rejected_before_package_is_written(self):
        self.write(self.event("user_message", "目标"))
        self.reader.snapshot = Mock(side_effect=[{"sessions": [dict(self.session)]}, {"sessions": [{**self.session, "status": {"code": "running"}}]}])
        with self.assertRaisesRegex(ValueError, "先结束原会话"):
            self.prepare()
        self.assertEqual(list(self.package_parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

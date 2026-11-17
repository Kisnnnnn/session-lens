"""External delegation boundaries; every transport is local or mocked."""
import copy
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sqlite3
import ssl
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlparse

from brain import ExternalBrain, DEFAULTS, request_provider, _system_post

KEY = 'fixture-private-key'


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = {'id': 'source', 'title': '修复问题', 'project': '项目A', 'lastEventAt': '2026-11-16T08:00:00Z',
                       'status': {'code': 'completed'}, 'context': {'inputTokens': 200},
                       'agents': [{'displayName': '检查架构', 'description': '检查边界', 'status': 'completed', 'rawSecret': 'hidden'}],
                       'events': [{'at': '08:00', 'title': '读取代码', 'detail': '检查入口', 'rawArguments': 'hidden'}]}
        exporter = patch("continuation.TRANSFER_DIRECTORY", self.base / "exports")
        exporter.start()
        self.addCleanup(exporter.stop)
        self.reader = SimpleNamespace(home=self.base / 'home')
        self.transport = Mock(return_value=({'summary': '结论', 'findings': ['依据'], 'uncertainties': ['未知'], 'next_steps': ['核验']}, {'total_tokens': 12}))
        self.snapshot = Mock(side_effect=lambda: {'sessions': [copy.deepcopy(self.source)]})
        self.prepare = Mock(side_effect=self.prepared)
        self.brain = ExternalBrain(self.base, self.reader, self.snapshot, self.prepare, self.transport)
        self.environment = patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def prepared(self, source):
        package = self.base / 'prepared'
        package.mkdir(exist_ok=True)
        paths = {k: str(package / name) for k, name in [('contextPath', 'context.md'), ('transcriptPath', 'transcript.json'), ('summaryPath', 'summary.md')]}
        Path(paths['contextPath']).write_text('## 机械交接材料\n\n生成时间：now\n分支：example', encoding='utf-8')
        Path(paths['transcriptPath']).write_text(json.dumps({'messages': [{'text': '目标和已完成项'}], 'selection': {'truncated': False}}), encoding='utf-8')
        return {**paths, 'workspace': str(self.base), 'threadId': source, 'sourceThreadId': source, 'packageId': 'a' * 32, 'draftUrl': 'codex://new?prompt=fixture'}

    def enable(self, **kwargs):
        return self.brain.save_config({'enabled': True, 'apiKey': KEY, **kwargs})

    def run_job(self, **kwargs):
        initial = self.brain.start({'question': '分析这个问题', **kwargs})
        return self.brain.result(initial['jobId'], 3000)

    def test_disabled_and_missing_key_never_call_transport(self):
        self.assertFalse(self.brain.public_config()['enabled'])
        with self.assertRaisesRegex(ValueError, '尚未启用'):
            self.run_job()
        self.brain.save_config({'enabled': True})
        with self.assertRaisesRegex(ValueError, '尚未配置'):
            self.run_job()
        self.transport.assert_not_called()

    def test_key_is_private_and_absent_from_public_config(self):
        config = self.enable()
        self.assertTrue(config['hasKey'])
        self.assertNotIn(KEY, json.dumps(config))
        self.assertNotIn(KEY, self.brain.config_path.read_text())
        self.assertEqual(self.brain.key_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.brain.key_path.parent.stat().st_mode & 0o777, 0o700)
        self.brain.save_config({'apiKey': ''})
        self.assertTrue(self.brain.public_config()['hasKey'])
        self.brain.save_config({'clearApiKey': True})
        self.assertFalse(self.brain.public_config()['hasKey'])

    def test_invalid_settings_are_rejected(self):
        for args in ({'baseUrl': 'http://remote.example'}, {'baseUrl': 'https://user:key@example.org'},
                     {'baseUrl': 'https://example.org/chat/completions'}, {'maxInputChars': 1},
                     {'maxOutputTokens': True}, {'model': 'model with spaces'}, {'enabled': 'yes'}, {'allowedModes': ['unknown']}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.brain.save_config(args)
        self.assertEqual(self.brain.save_config({'baseUrl': 'http://127.0.0.1:9000/v1/'})['baseUrl'], 'http://127.0.0.1:9000/v1')

    def test_analysis_only_sends_provided_materials_by_default(self):
        self.enable()
        result = self.run_job(threadId='source', context='可选材料')
        self.assertEqual(result['status'], 'completed')
        packet = json.loads(self.transport.call_args.args[2])
        self.assertEqual(packet['userProvidedMaterials'], '可选材料')
        self.assertNotIn('sessionRecords', packet)
        self.assertNotIn('conversationExcerpt', packet)
        self.snapshot.assert_not_called()
        self.prepare.assert_not_called()

    def test_explain_projects_monitor_records_without_raw_arguments(self):
        self.enable()
        result = self.run_job(mode='explain', threadId='source')
        self.assertEqual(result['status'], 'completed')
        packet = json.loads(self.transport.call_args.args[2])
        self.assertEqual(packet['sessionRecords']['agents'][0]['displayName'], '检查架构')
        self.assertNotIn('hidden', json.dumps(packet))
        self.assertNotIn('conversationExcerpt', packet)
        self.prepare.assert_not_called()

    def test_conversation_is_explicit_and_confined_to_session_directory(self):
        self.enable()
        home = self.reader.home
        (home / 'sessions').mkdir(parents=True)
        rollout = home / 'sessions/source.jsonl'
        rollout.write_text(json.dumps({'type': 'event_msg', 'payload': {'type': 'user_message', 'message': '原始目标'}}) + '\n')
        with closing(sqlite3.connect(home / 'state_5.sqlite')) as database:
            database.execute('CREATE TABLE threads(id TEXT, rollout_path TEXT)')
            database.execute('INSERT INTO threads VALUES(?,?)', ('source', str(rollout)))
            database.commit()
        result = self.run_job(threadId='source', includeConversation=True)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('原始目标', self.transport.call_args.args[2])
        outside = self.base / 'private.jsonl'
        outside.write_text('private')
        with closing(sqlite3.connect(home / 'state_5.sqlite')) as database:
            database.execute('UPDATE threads SET rollout_path=?', (str(outside),))
            database.commit()
        with self.assertRaisesRegex(ValueError, '路径不可用'):
            self.run_job(threadId='source', includeConversation=True)
        self.assertEqual(self.transport.call_count, 1)

    def test_limits_and_redacts_materials_and_results(self):
        self.enable(maxInputChars=4000)
        self.transport.return_value = ({'summary': KEY + '结论'}, {})
        result = self.run_job(context=KEY + 'a' * 10000 + KEY)
        material = self.transport.call_args.args[2]
        self.assertLessEqual(len(material), 4000)
        self.assertTrue(result['inputTruncated'])
        self.assertNotIn(KEY, material)
        self.assertNotIn(KEY, json.dumps(result))
        self.assertNotIn(KEY, Path(result['resultPath']).read_text())

    def test_mode_gates_and_missing_question(self):
        self.enable(allowedModes=['analysis'])
        with self.assertRaisesRegex(ValueError, '场景未启用'):
            self.run_job(mode='explain', threadId='source')
        with self.assertRaisesRegex(ValueError, '输入'):
            self.brain.start({'mode': 'analysis'})
        self.transport.assert_not_called()

    def test_identical_active_request_is_joined_then_cache_reused(self):
        self.enable()
        release = threading.Event()
        entered = threading.Event()
        def wait(*args):
            entered.set()
            release.wait(3)
            return {'summary': '分析结果'}, {'total_tokens': 12}
        self.transport.side_effect = wait
        first = self.brain.start({'question': '分析'})
        entered.wait(1)
        second = self.brain.start({'question': '分析'})
        self.assertEqual(first['jobId'], second['jobId'])
        release.set()
        self.assertEqual(self.brain.result(first['jobId'], 3000)['status'], 'completed')
        cached = self.brain.start({'question': '分析'})
        self.assertTrue(cached['cached'])
        self.assertEqual(cached['usage']['total_tokens'], 12)
        self.assertEqual(self.transport.call_count, 1)

    def test_probe_has_no_session_data_and_does_not_require_enabled(self):
        self.brain.save_config({'apiKey': KEY})
        result = self.brain.start({}, probe=True)
        result = self.brain.result(result['jobId'], 3000)
        self.assertEqual(result['status'], 'completed')
        config, _, material = self.transport.call_args.args
        self.assertEqual(config['maxOutputTokens'], 128)
        self.assertFalse(config['thinkingEnabled'])
        self.assertNotIn('修复问题', material)
        self.snapshot.assert_not_called()

    def test_handoff_requires_terminal_source_and_real_summary(self):
        self.enable()
        self.source['status']['code'] = 'running'
        with self.assertRaisesRegex(ValueError, '结束原会话'):
            self.run_job(mode='handoff', threadId='source')
        self.prepare.assert_not_called()
        self.source['status']['code'] = 'completed'
        result = self.run_job(mode='handoff', threadId='source')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('没有返回交接摘要', result['error'])
        self.assertNotIn('continuation', result)

    def test_handoff_writes_reviewable_summary_and_draft_without_execution(self):
        self.enable()
        self.transport.return_value = ({'summary': '已整理', 'handoff_markdown': '目标、待办 ' + KEY}, {})
        result = self.run_job(mode='handoff', threadId='source')
        self.assertEqual(result['status'], 'completed')
        continuation = result['continuation']
        self.assertTrue(continuation['summaryReady'])
        self.assertTrue(continuation['draftUrl'].startswith('codex://new?'))
        self.assertIn('待 Codex 核验', Path(continuation['summaryPath']).read_text())
        self.assertNotIn(KEY, Path(continuation['summaryPath']).read_text())
        self.assertIn('conversationExcerpt', json.loads(self.transport.call_args.args[2]))

    def test_source_change_while_analyzing_removes_continuation(self):
        self.enable()
        def changed(*args):
            self.source['lastEventAt'] = 'new'
            return {'summary': '旧结论', 'handoff_markdown': '旧摘要'}, {}
        self.transport.side_effect = changed
        result = self.run_job(mode='handoff', threadId='source')
        self.assertEqual(result['status'], 'completed')
        self.assertIn('源会话记录已变化', result['answer'])
        self.assertNotIn('continuation', result)
        self.assertFalse((self.base / 'prepared/summary.md').exists())

    def test_errors_do_not_expose_known_key_and_do_not_retry(self):
        self.enable()
        self.transport.side_effect = ValueError('失败 ' + KEY)
        result = self.run_job()
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn(KEY, json.dumps(result))
        self.assertEqual(self.transport.call_count, 1)


class ProviderTests(unittest.TestCase):
    def request(self, payload, status=200, config=None):
        connection = Mock()
        connection.sock = None
        connection.getresponse.return_value.status = status
        connection.getresponse.return_value.read1.side_effect = [json.dumps(payload).encode(), b'']
        with patch('brain.http.client.HTTPSConnection', return_value=connection), patch('brain.sys.platform', 'linux'):
            result = request_provider(config or DEFAULTS, KEY, '引用材料')
        return result, connection

    def payload(self, **choice):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'summary': '结论'}), 'reasoning_content': 'HIDDEN_REASONING'}, **choice}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}

    def test_deepseek_format_and_only_visible_result(self):
        (result, usage), connection = self.request(self.payload())
        self.assertEqual(result, {'summary': '结论'})
        self.assertEqual(usage['total_tokens'], 12)
        sent = json.loads(connection.request.call_args.kwargs['body'])
        self.assertEqual(sent['thinking'], {'type': 'disabled'})
        self.assertEqual(sent['response_format'], {'type': 'json_object'})
        self.assertNotIn('tools', sent)
        self.assertEqual(connection.request.call_args.args[:2], ('POST', '/chat/completions'))
        connection.close.assert_called_once()

    def test_incomplete_response_and_http_failure_are_not_retried(self):
        with self.assertRaisesRegex(ValueError, '完整结束'):
            self.request(self.payload(finish_reason='length'))
        with self.assertRaisesRegex(ValueError, '认证失败'):
            self.request({'error': KEY}, status=401)

    def test_real_loopback_transport_uses_compatible_path_and_never_follows_redirect(self):
        received = []
        response_status = [200]
        payload = self.payload()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                received.append((self.path, self.headers['Authorization'], json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                body = json.dumps(payload).encode()
                self.send_response(response_status[0])
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Location', '/do-not-follow')
                self.end_headers()
                self.wfile.write(body)
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            config = {**DEFAULTS, 'baseUrl': 'http://127.0.0.1:' + str(httpd.server_port) + '/v1'}
            status, raw = _system_post(urlparse(config['baseUrl']), KEY, json.dumps({'messages': [{'role': 'user', 'content': '中文\n带 \" 引号和反斜线 \\'}]}).encode())
            self.assertEqual(status, 200)
            self.assertIn('choices', json.loads(raw))
            received.clear()
            result, _ = request_provider(config, KEY, 'selected materials')
            self.assertEqual(result['summary'], '结论')
            self.assertEqual(received[0][0], '/v1/chat/completions')
            self.assertEqual(received[0][1], 'Bearer ' + KEY)
            self.assertNotIn('thinking', received[0][2])
            response_status[0] = 302
            with self.assertRaisesRegex(ValueError, 'HTTP 302'):
                request_provider(config, KEY, 'selected materials')
            self.assertEqual(len(received), 2)
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join(2)

    def test_native_transport_keeps_secret_off_argv_and_does_not_disable_tls(self):
        captured = {}
        def run(argv, **kwargs):
            captured['argv'] = argv
            captured['settings'] = kwargs['input'].decode()
            kwargs['stdout'].write(b'{"choices": []}\n200')
            return SimpleNamespace(returncode=0)
        with patch('brain.subprocess.run', side_effect=run), patch('brain.getproxies', return_value={'https': 'http://127.0.0.1:8888'}), patch('brain.proxy_bypass', return_value=False):
            status, _ = _system_post(urlparse('https://api.deepseek.com'), KEY, b'{"messages": []}')
        self.assertEqual(status, 200)
        self.assertNotIn(KEY, str(captured['argv']))
        self.assertIn('Authorization: Bearer ' + KEY, captured['settings'])
        self.assertIn('proxy = "http://127.0.0.1:8888"', captured['settings'])
        self.assertNotIn('insecure', captured['settings'])
        self.assertNotIn('location', captured['settings'])
        self.assertNotIn('retry', captured['settings'])

    def test_certificate_failure_is_not_reported_as_timeout(self):
        with patch('brain.sys.platform', 'linux'), patch('brain.http.client.HTTPSConnection', side_effect=ssl.SSLCertVerificationError('fixture')):
            with self.assertRaisesRegex(ValueError, '证书校验失败'):
                request_provider(DEFAULTS, KEY, 'fixture')
        for code, label in [(60, '证书校验失败'), (28, '90 秒'), (7, '连接服务或代理')]:
            with self.subTest(code=code), patch('brain.subprocess.run', return_value=SimpleNamespace(returncode=code)):
                with self.assertRaisesRegex(ValueError, label):
                    _system_post(urlparse('https://api.deepseek.com'), KEY, b'{}')

    def test_malformed_or_missing_conclusion_rejected(self):
        for data in ({}, {'choices': []}, self.payload(message={'content': '{}'}), self.payload(message={'content': 'not JSON'})):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.request(data)


if __name__ == '__main__':
    unittest.main()

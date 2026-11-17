"""Verify MCP data boundaries and loopback access, without user session fixtures."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPCookieProcessor, ProxyHandler
from http.cookiejar import CookieJar

import server


SAMPLE = {
    'sampledAt': '2026-11-16T06:00:00Z',
    'source': {'kind': 'local', 'status': 'connected', 'message': 'fixture'},
    'sessions': [{
        'id': 'test-thread', 'title': 'A task', 'workspace': '/fixture/workspace',
        'project': 'workspace', 'status': {'label': '已完成'},
        'context': {'inputTokens': 123, 'windowTokens': 1000},
    }],
    'workspaces': [{'path': '/fixture/workspace', 'name': 'workspace', 'count': 1}],
}


class ServerTests(unittest.TestCase):
    def test_snapshot_keeps_full_data_out_of_model_visible_fields(self):
        with patch.object(server, 'snapshot', return_value=SAMPLE):
            result = server.tool_result('session_snapshot', {})
        self.assertEqual(result['_meta']['snapshot'], SAMPLE)
        self.assertNotIn('/fixture/workspace', json.dumps(result['content']))
        self.assertNotIn('sessions', result['structuredContent'])
        self.assertEqual(result['structuredContent']['counts']['sessions'], 1)

    def test_external_answer_is_bounded_for_model_and_complete_for_ui(self):
        job = {'jobId': 'a' * 32, 'status': 'completed', 'answer': 'LONG_RESULT' * 4000,
               'answerExcerpt': 'bounded excerpt', 'usage': {'total_tokens': 12}}
        result = server.brain_result(job)
        self.assertEqual(result['_meta']['brainResult']['answer'], job['answer'])
        self.assertNotIn('LONG_RESULT', json.dumps(result['content']))
        self.assertNotIn('answer', result['structuredContent'])
        self.assertEqual(result['structuredContent']['usage'], job['usage'])

    def test_cross_app_export_full_text_is_only_for_ui(self):
        with patch.object(server, 'continuation_export', return_value={'text': 'PRIVATE_TRANSCRIPT', 'path': '/fixture/handoff.md'}):
            result = server.tool_result('session_continuation_export', {'packageId': 'a' * 32})
        self.assertEqual(result['_meta']['transferText'], 'PRIVATE_TRANSCRIPT')
        self.assertNotIn('PRIVATE_TRANSCRIPT', json.dumps(result['content']))
        self.assertNotIn('PRIVATE_TRANSCRIPT', json.dumps(result['structuredContent']))

    def test_handoff_is_outside_workspace_and_identifies_missing_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(server, 'snapshot', return_value=SAMPLE), \
                 patch.object(server.Path, 'home', return_value=Path(folder)), \
                 patch.object(server.subprocess, 'run', side_effect=OSError):
                result = server.make_handoff('test-thread')
            self.assertTrue(Path(result['path']).is_relative_to(Path(folder) / '.codex/artifacts'))
            self.assertIn('不是完整聊天摘要', result['text'])
            self.assertEqual(Path(result['path']).read_text(), result['text'])

    def test_handoff_requires_known_thread(self):
        with patch.object(server, 'snapshot', return_value=SAMPLE):
            with self.assertRaises(ValueError):
                server.make_handoff('../../unknown')

    def test_loopback_data_requires_cookie_and_rejects_foreign_origin(self):
        httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.DashboardHandler)
        httpd.access_token = 'test-token'
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        opener = build_opener(ProxyHandler({}), HTTPCookieProcessor(CookieJar()))
        url = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with self.assertRaises(HTTPError) as error:
                opener.open(url + '/api/snapshot')
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            # Authenticate without depending on UI file readiness.
            with opener.open(url + '/health') as response:
                self.assertEqual(json.load(response)['service'], 'session-lens')
            opener.addheaders = [('Cookie', 'session_lens=test-token')]
            with patch.object(server, 'snapshot', return_value=SAMPLE):
                with opener.open(url + '/api/snapshot') as response:
                    self.assertEqual(json.load(response), SAMPLE)
            request = Request(url + '/api/handoff', data=b'{"threadId":"test-thread"}',
                              headers={'Origin': 'https://external.example', 'Content-Type': 'application/json'})
            with self.assertRaises(HTTPError) as error:
                opener.open(request)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            request = Request(url + '/api/snapshot', headers={'Host': 'external.example'})
            with self.assertRaises(HTTPError) as error:
                opener.open(request)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join(timeout=2)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Session Lens: local stdio MCP server and loopback dashboard, stdlib only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from adapter import SessionReader
from continuation import prepare_continuation as prepare_package, continuation_export
from activity import write_agent_note
from brain import ExternalBrain
from dispatch import SubtaskDispatcher
from rescue import RescueManager

ROOT = Path(__file__).resolve().parent
DATA = Path.home() / '.codex' / 'artifacts' / 'codex-session-helper'
UI_URI = 'ui://session-lens/dashboard-v9.html'
READER = SessionReader()
LOCK = threading.RLock()


def snapshot():
    """Serialize incremental reader access across HTTP and MCP callers."""
    with LOCK:
        data = READER.snapshot()
        manager = globals().get('RESCUE')
        return manager.decorate(data) if manager else data


def make_handoff(thread_id):
    """Export observable state; semantic decisions require agent-authored notes."""
    session = next((s for s in snapshot()['sessions'] if s['id'] == thread_id), None)
    if not session:
        raise ValueError('会话不存在或已超出最近会话范围，请先刷新。')
    workspace = Path(session['workspace'])
    git_state = {}
    for key, args in [('branch', ['rev-parse', '--abbrev-ref', 'HEAD']),
                      ('commit', ['rev-parse', 'HEAD']),
                      ('changes', ['status', '--short', '--untracked-files=no'])]:
        try:
            result = subprocess.run(['git', '-C', str(workspace), *args],
                                    capture_output=True, text=True, timeout=3,
                                    env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
            git_state[key] = result.stdout.strip()[:5000] if result.returncode == 0 else '未能读取'
        except (OSError, subprocess.TimeoutExpired):
            git_state[key] = '未能读取'
    stamp = datetime.now(timezone.utc).isoformat()
    context = session['context']
    text = '\n'.join([
        '# Codex 会话续接材料（机械状态）', '',
        f'生成时间：{stamp}', f'源会话：{thread_id}',
        f'任务标题：{session["title"]}', f'工作区：{workspace}',
        f'分支：{git_state["branch"]}', f'提交：{git_state["commit"]}',
        f'最近已观察状态：{session["status"]["label"]}',
        f'当前输入用量：{context.get("inputTokens")} / {context.get("windowTokens")} token',
        '', '## 已跟踪文件改动', '```text', git_state['changes'] or '无', '```',
        '', '仅列已跟踪文件；继续前还需核查未跟踪文件和分支变化。',
        '', '## 继续工作前补充',
        '- 原任务目标与适用的 AGENTS.md 约束。',
        '- 已确认的关键决定、完成项、验证结果和剩余事项。',
        '- 核查仍在执行或结果未知的命令，避免重复操作。',
        '', '此材料是可观察状态，不是完整聊天摘要、代码备份或进程恢复。',
    ])
    slug = re.sub(r'[^\w.-]', '-', session.get('project') or workspace.name)[:64] or 'workspace'
    workspace_id = hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
    folder = Path.home() / '.codex' / 'artifacts' / slug / 'session-helper' / workspace_id
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
    path = folder / f'handoff-{name}.md'
    temp = path.with_suffix('.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)
    return {'text': text, 'path': str(path)}


def prepare_continuation(thread_id):
    """Prepare quoted source material; the host performs AI summary and continuation."""
    with LOCK:
        return prepare_package(thread_id, READER, lambda: make_handoff(thread_id))


def describe_agent(args):
    """Record an explicit division of work without changing source session history."""
    with LOCK:
        parent_id = args.get('sourceThreadId')
        agent_id = args.get('agentId')
        parent = next((s for s in snapshot()['sessions'] if s['id'] == parent_id), None)
        if not parent or not any(agent['id'] == agent_id for agent in parent['agents']):
            raise ValueError('未找到该会话关联的协作任务，请先刷新。')
        path = READER.home / 'artifacts' / 'codex-session-helper' / 'agent-notes.json'
        return write_agent_note(path, parent_id, agent_id, args.get('label'), args.get('description'))


BRAIN = ExternalBrain(DATA, READER, snapshot, prepare_continuation)
DISPATCH = SubtaskDispatcher(BRAIN, DATA)
RESCUE = RescueManager(BRAIN, DATA, snapshot, make_handoff)


def brain_result(job):
    """Keep detailed external answers in the UI while returning a bounded model excerpt."""
    public = {key: value for key, value in job.items() if key not in ('answer', 'dispatchMaterials', 'taskPlan', 'rescuePlan')}
    if job.get('taskPlan'):
        public['taskPlan'] = [{'id': task['id'], 'title': task['title']} for task in job['taskPlan']]
    if job.get('rescuePlan'):
        public['rescuePlan'] = {'goal': job['rescuePlan']['goal'], 'tasks': [{key: task[key] for key in ('id', 'title', 'execution', 'dependsOn')} for task in job['rescuePlan']['tasks']]}
    text = job.get('answerExcerpt') or job.get('error') or '外置分析已开始，请使用任务 ID 查询结果。'
    return {'content': [{'type': 'text', 'text': text}], 'structuredContent': public, '_meta': {'brainResult': job}}


def dispatch_result(batch):
    public = {key: value for key, value in batch.items() if key not in ('report', 'tasks')}
    public['tasks'] = [{key: task[key] for key in ('id', 'title', 'status', 'jobId', 'error', 'cached', 'usage') if key in task} for task in batch['tasks']]
    return {'content': [{'type': 'text', 'text': batch.get('reportExcerpt') or batch.get('error') or '子任务分发已开始。'}],
            'structuredContent': public, '_meta': {'dispatchResult': batch}}


def rescue_result(value, kind='capture'):
    """Keep source excerpts and detailed plans out of model-visible status responses."""
    if kind == 'capture':
        public = {key: value.get(key) for key in ('captureId', 'source', 'markedAt', 'diagnosis', 'originalGoal', 'latestRequest', 'readableExcerpt', 'materialsPath', 'phase', 'label')}
    elif kind == 'review':
        public = {'captureId': value['capture']['captureId'], 'goal': value['plan']['goal'],
                  'tasks': [{key: task[key] for key in ('id', 'title', 'execution', 'dependsOn')} for task in value['plan']['tasks']]}
    else:
        fields = ('packageId', 'captureId', 'planJobId', 'sourceThreadId', 'sourceTitle', 'sourceActive', 'childActive', 'workspace', 'contextPath', 'planPath',
                  'reportPath', 'waves', 'maxParallel', 'ready', 'phase', 'claimed', 'claimToken', 'resumed', 'blocked', 'reason', 'activePackageId', 'existing')
        public = {key: value[key] for key in fields if key in value}
        if value.get('progress'):
            public['progress'] = {key: {field: task[field] for field in ('id', 'title', 'status', 'agentId') if field in task} for key, task in value['progress'].items()}
        if value.get('integration'):
            public['integration'] = {**value['integration'], 'summary': value['integration']['summary'][:1500]}
    if kind == 'capture':
        for field in ('originalGoal', 'latestRequest'):
            if isinstance(public.get(field), str):
                public[field] = public[field][:1000]
    return {'content': [{'type': 'text', 'text': '卡顿任务处理材料已更新。原始记录、DP 诊断及子任务结果均需核验；准备包不代表子 agent 已开始执行。'}],
            'structuredContent': public, '_meta': {'rescueResult': value}}


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve authenticated loopback requests without modifying Codex state."""
    def log_message(self, *_args):
        pass

    def reply(self, code, body, content_type='application/json; charset=utf-8', cookie=None):
        payload = body.encode('utf-8') if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        if cookie:
            self.send_header('Set-Cookie', cookie)
        self.end_headers()
        self.wfile.write(payload)

    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def authorized(self):
        if not self.valid_host():
            return False
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie', ''))
            return secrets.compare_digest(cookie['session_lens'].value, self.server.access_token)
        except (KeyError, ValueError):
            return False

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/health' and self.valid_host():
            self.reply(200, {'service': 'session-lens', 'version': '0.6.0'})
            return
        token = parse_qs(parsed.query).get('token', [''])[0]
        if parsed.path == '/' and self.valid_host() and secrets.compare_digest(token, self.server.access_token):
            self.send_response(303)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie', f'session_lens={token}; HttpOnly; SameSite=Strict; Path=/')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if not self.authorized():
            self.reply(403, {'error': '请使用插件返回的本地面板链接。'})
            return
        if parsed.path == '/':
            self.reply(200, (ROOT / 'web/index.html').read_text(), 'text/html; charset=utf-8')
        elif parsed.path == '/api/snapshot':
            self.reply(200, snapshot())
        elif parsed.path.startswith('/api/continuation/export/'):
            try:
                self.reply(200, continuation_export(parsed.path.rsplit('/', 1)[-1]))
            except (ValueError, OSError) as error:
                self.reply(400, {'error': str(error)})
        elif parsed.path.startswith('/api/brain/dispatch/'):
            try:
                self.reply(200, DISPATCH.result(parsed.path.rsplit('/', 1)[-1]))
            except ValueError as error:
                self.reply(400, {'error': str(error)})
        elif parsed.path.startswith('/api/rescue/thread/'):
            try:
                self.reply(200, RESCUE.resume({'threadId': parsed.path.rsplit('/', 1)[-1]}))
            except (ValueError, OSError) as error:
                self.reply(400, {'error': str(error)})
        elif parsed.path.startswith('/api/rescue/package/'):
            try:
                self.reply(200, RESCUE.status({'packageId': parsed.path.rsplit('/', 1)[-1]}))
            except (ValueError, OSError) as error:
                self.reply(400, {'error': str(error)})
        elif parsed.path == '/api/brain/config':
            self.reply(200, BRAIN.public_config())
        elif parsed.path.startswith('/api/brain/jobs/'):
            try:
                self.reply(200, BRAIN.result(parsed.path.rsplit('/', 1)[-1]))
            except ValueError as error:
                self.reply(400, {'error': str(error)})
        else:
            self.reply(404, {'error': '页面不存在'})

    def do_POST(self):
        origin = self.headers.get('Origin')
        expected = f'http://{self.headers.get("Host")}'
        if not self.authorized() or (origin and origin != expected):
            self.reply(403, {'error': '本地会话验证失败'})
            return
        if self.path not in ('/api/handoff', '/api/continuation', '/api/brain/config', '/api/brain/analyze', '/api/brain/test', '/api/brain/plan', '/api/brain/dispatch', '/api/brain/dispatch-stop', '/api/rescue/capture', '/api/rescue/plan', '/api/rescue/review', '/api/rescue/prepare'):
            self.reply(404, {'error': '操作不存在'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            limit = 512000 if self.path in ('/api/brain/analyze', '/api/brain/plan', '/api/brain/dispatch', '/api/rescue/plan') else 4096
            if not 0 < length <= limit:
                raise ValueError('请求长度无效')
            args = json.loads(self.rfile.read(length))
            if not isinstance(args, dict):
                raise ValueError('请求格式无效。')
            with LOCK:
                if self.path == '/api/rescue/capture':
                    result = RESCUE.capture(args)
                elif self.path == '/api/rescue/plan':
                    result = RESCUE.plan(args)
                elif self.path == '/api/rescue/review':
                    result = RESCUE.review(args)
                elif self.path == '/api/rescue/prepare':
                    result = RESCUE.prepare(args)
                elif self.path == '/api/brain/config':
                    result = BRAIN.save_config(args)
                elif self.path == '/api/brain/plan':
                    result = BRAIN.start({**args, 'mode': 'dispatch'})
                elif self.path == '/api/brain/dispatch':
                    result = DISPATCH.start(args)
                elif self.path == '/api/brain/dispatch-stop':
                    result = DISPATCH.stop(args.get('batchId'))
                elif self.path == '/api/brain/analyze':
                    result = BRAIN.start(args)
                elif self.path == '/api/brain/test':
                    result = BRAIN.start({}, probe=True)
                elif self.path == '/api/continuation':
                    result = prepare_continuation(args.get('threadId'))
                else:
                    result = make_handoff(args.get('threadId'))
            self.reply(200, result)
        except (ValueError, TypeError, AttributeError, OSError) as error:
            self.reply(400, {'error': str(error)})


def serve_dashboard(port=0, reuse_session=False):
    DATA.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(('127.0.0.1', port), DashboardHandler)
    access_token = None
    if reuse_session:
        try:
            previous = json.loads((DATA / 'runtime.json').read_text())
            if previous.get('port') == httpd.server_port:
                access_token = parse_qs(urlparse(previous['url']).query).get('token', [None])[0]
        except (OSError, ValueError, KeyError):
            pass
    httpd.access_token = access_token or secrets.token_urlsafe(24)
    runtime = {'pid': os.getpid(), 'port': httpd.server_port,
               'url': f'http://127.0.0.1:{httpd.server_port}/?token={httpd.access_token}'}
    runtime_path = DATA / 'runtime.json'
    descriptor = os.open(runtime_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(runtime, handle)
    print(json.dumps({'dashboardUrl': runtime['url']}), flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def dashboard_url():
    """Reuse a local preview server or start one only on explicit tool invocation."""
    from urllib.request import build_opener, ProxyHandler
    runtime_path = DATA / 'runtime.json'
    local_http = build_opener(ProxyHandler({}))

    def current():
        try:
            data = json.loads(runtime_path.read_text())
            port = int(data['port'])
            with local_http.open(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                if json.load(response).get('service') == 'session-lens':
                    return data['url']
        except (OSError, ValueError, KeyError):
            pass
        return None

    url = current()
    if url:
        return url
    subprocess.Popen([sys.executable, str(ROOT / 'server.py'), '--http'],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(30):
        time.sleep(0.1)
        url = current()
        if url:
            return url
    raise RuntimeError('本地面板启动失败，请运行 server.py --http 查看错误。')


def tool_result(name, args):
    if name == 'session_snapshot':
        data = snapshot()
        counts = {'sessions': len(data['sessions']), 'projects': len(data.get('projects') or data['workspaces'])}
        return {'content': [{'type': 'text', 'text': f'会话透镜已读取 {counts["sessions"]} 个最近会话，按 {counts["projects"]} 个已保存项目或未归属分组显示。完整数据仅供面板显示。'}],
                'structuredContent': {'sampledAt': data['sampledAt'], 'counts': counts},
                '_meta': {'snapshot': data}}
    if name == 'session_handoff':
        with LOCK:
            data = make_handoff(args.get('threadId'))
        return {'content': [{'type': 'text', 'text': data['text']}], 'structuredContent': data}
    if name == 'session_prepare_continuation':
        data = prepare_continuation(args.get('threadId'))
        return {'content': [{'type': 'text', 'text': '续接输入已准备。尚未调用模型生成摘要或创建新聊天；请按 session-lens skill 完成后续步骤。'}],
                'structuredContent': data}
    if name == 'session_continuation_export':
        data = continuation_export(args.get('packageId'))
        return {'content': [{'type': 'text', 'text': '跨应用交接文件已读取，仅供面板下载。'}],
                'structuredContent': {'path': data['path']}, '_meta': {'transferText': data['text']}}
    if name == 'session_describe_agent':
        data = describe_agent(args)
        return {'content': [{'type': 'text', 'text': '协作分工说明已保存到本地面板。'}], 'structuredContent': data}
    if name == 'external_brain_status':
        data = BRAIN.public_config()
        return {'content': [{'type': 'text', 'text': '外置大脑已启用。' if data['enabled'] else '外置大脑默认关闭，可在本地设置页配置。'}], 'structuredContent': data}
    if name == 'session_rescue_resume':
        data = RESCUE.resume(args)
        return {'content': [{'type': 'text', 'text': '已读取本地拆解记录，没有新建解析标记或发出模型请求。'}],
                'structuredContent': {'found': data['found'], 'mark': data.get('mark')}, '_meta': {'rescueResult': data}}
    if name == 'session_rescue_capture':
        return rescue_result(RESCUE.capture(args))
    if name == 'external_brain_rescue_plan':
        job = RESCUE.plan(args)
        return brain_result(BRAIN.result(job['jobId'], 25000))
    if name == 'session_rescue_review':
        return rescue_result(RESCUE.review(args), 'review')
    if name == 'session_rescue_prepare':
        return rescue_result(RESCUE.prepare(args), 'package')
    if name == 'session_rescue_status':
        return rescue_result(RESCUE.status(args), 'package')
    if name == 'session_rescue_claim':
        return rescue_result(RESCUE.claim(args), 'package')
    if name == 'session_rescue_progress':
        return rescue_result(RESCUE.progress(args), 'package')
    if name == 'session_rescue_finalize':
        return rescue_result(RESCUE.finalize(args), 'package')
    if name == 'external_brain_plan':
        job = BRAIN.start({**args, 'mode': 'dispatch'})
        return brain_result(BRAIN.result(job['jobId'], 25000))
    if name == 'external_brain_dispatch':
        return dispatch_result(DISPATCH.start(args))
    if name == 'external_brain_dispatch_status':
        return dispatch_result(DISPATCH.result(args.get('batchId')))
    if name == 'external_brain_dispatch_stop':
        return dispatch_result(DISPATCH.stop(args.get('batchId')))
    if name == 'external_brain_analyze':
        job = BRAIN.start(args)
        return brain_result(BRAIN.result(job['jobId'], 25000))
    if name == 'external_brain_result':
        return brain_result(BRAIN.result(args.get('jobId'), args.get('waitMs', 0)))
    if name == 'session_dashboard':
        url = dashboard_url()
        data = {'dashboardUrl': url}
        return {'content': [{'type': 'text', 'text': f'本地实时面板：{url}'}], 'structuredContent': data}
    raise ValueError('未知工具')


def tools_list():
    return [
        {'name': 'session_snapshot', 'description': '查看本机最近 Codex 会话的上下文用量、等待原因、子 agent 与事件时间轴。仅只读采样，完整数据供可视面板。',
         'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
         'annotations': {'readOnlyHint': True, 'openWorldHint': False},
         '_meta': {'ui': {'resourceUri': UI_URI, 'visibility': ['model', 'app']}, 'openai/outputTemplate': UI_URI}},
        {'name': 'session_handoff', 'description': '为选定会话保存机械续接材料到仓库外；不是完整业务摘要或代码备份。',
         'inputSchema': {'type': 'object', 'properties': {'threadId': {'type': 'string'}}, 'required': ['threadId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'session_dashboard', 'description': '启动或复用本地实时面板，返回可在 Codex 浏览器面板打开的 loopback URL。',
         'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'session_prepare_continuation', 'description': '用户请求自动续接时，为已结束执行的会话提取有界原对话和工作区状态，准备 AI 交接摘要输入。此工具本身不调用模型、不创建聊天；后续按 session-lens skill 生成摘要并创建新聊天。',
         'inputSchema': {'type': 'object', 'properties': {'threadId': {'type': 'string'}}, 'required': ['threadId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False},
         '_meta': {'ui': {'visibility': ['model', 'app']}}},
        {'name': 'session_continuation_export', 'description': '读取已准备的跨应用 Markdown 交接文件，完整材料只供面板下载。不调用模型、不打开应用。',
         'inputSchema': {'type': 'object', 'properties': {'packageId': {'type': 'string', 'pattern': '^[a-f0-9]{32}$'}}, 'required': ['packageId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': True, 'openWorldHint': False}, '_meta': {'ui': {'visibility': ['app']}}},
        {'name': 'session_describe_agent', 'description': '将主聊天已经明确分派的子 agent 分工保存为中文名称与说明，供会话面板展示。只记录有依据的分工，不猜测工作结果，不调用模型。',
         'inputSchema': {'type': 'object', 'properties': {'sourceThreadId': {'type': 'string'}, 'agentId': {'type': 'string'}, 'label': {'type': 'string', 'maxLength': 40}, 'description': {'type': 'string', 'maxLength': 160}}, 'required': ['sourceThreadId', 'agentId', 'label', 'description'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'external_brain_status', 'description': '查看可选外置大脑是否启用、模型与材料预算。仅本地读取，不返回 API Key、不调用模型。',
         'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
         'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
        {'name': 'external_brain_analyze', 'description': '将必要的用户授权材料发送给已配置的外部模型（默认 DeepSeek）作通用分析、进展解释或交接摘要，会消耗外部 API 额度。仅启用且用户请求或 autoDelegate 允许时使用；返回分析供主 Codex 核验，不执行建议。最长等待25秒，未完成用 external_brain_result 查询。',
         'inputSchema': {'type': 'object', 'properties': {'mode': {'type': 'string', 'enum': ['analysis', 'explain', 'handoff']}, 'question': {'type': 'string', 'maxLength': 4000}, 'context': {'type': 'string', 'maxLength': 120000}, 'threadId': {'type': 'string'}, 'includeConversation': {'type': 'boolean'}}, 'required': ['mode'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': True},
         '_meta': {'ui': {'visibility': ['model', 'app']}}},
        {'name': 'external_brain_plan', 'description': '用户请求外部子任务分发时，调用一次外部模型生成可编辑的独立分析任务清单（1–6项）。会消耗额度；不会自动启动子任务。先向用户展示清单与预算，再按授权分发。',
         'inputSchema': {'type': 'object', 'properties': {'question': {'type': 'string', 'maxLength': 4000}, 'context': {'type': 'string', 'maxLength': 120000}, 'threadId': {'type': 'string'}, 'includeConversation': {'type': 'boolean'}}, 'required': ['question'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': True}},
        {'name': 'external_brain_dispatch', 'description': '将用户选定的1–6个独立分析子任务分发给已配置外部模型，最多并行2项，每项独立计费；需要子任务分发已启用。可传计划任务ID并选择taskIds，或手动tasks和问题材料。结果只作待核验分析，无工具执行权限。',
         'inputSchema': {'type': 'object', 'properties': {'planJobId': {'type': 'string'}, 'question': {'type': 'string', 'maxLength': 4000}, 'context': {'type': 'string', 'maxLength': 120000}, 'threadId': {'type': 'string'}, 'includeConversation': {'type': 'boolean'}, 'parallelism': {'type': 'integer', 'enum': [1, 2]}, 'taskIds': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 6}, 'tasks': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {'type': 'object', 'properties': {'id': {'type': 'string'}, 'title': {'type': 'string', 'maxLength': 80}, 'instruction': {'type': 'string', 'maxLength': 1800}}, 'required': ['title', 'instruction'], 'additionalProperties': False}}}, 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': True}},
        {'name': 'external_brain_dispatch_status', 'description': '只查询已发起子任务组的本地状态和结果，不重新调用外部模型。完整结果仅供面板，主聊天返回短结论与本地路径。',
         'inputSchema': {'type': 'object', 'properties': {'batchId': {'type': 'string'}}, 'required': ['batchId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
        {'name': 'external_brain_dispatch_stop', 'description': '停止清单中尚未分发的子任务。已经交给分析器或服务的请求继续等待，仍可能计费；不声称取消服务端请求。',
         'inputSchema': {'type': 'object', 'properties': {'batchId': {'type': 'string'}}, 'required': ['batchId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'session_rescue_resume', 'description': '打开已有标记会话时，先恢复源解析、DP规划与Codex子任务进度；不覆盖旧标记，不重复调用模型。found=false时再本地解析。',
         'inputSchema': {'type': 'object', 'properties': {'threadId': {'type': 'string'}}, 'required': ['threadId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
        {'name': 'session_rescue_capture', 'description': '用户请求解析疑似卡顿会话时，读取有界原对话、监测状态与Git元数据并保存排查标记。只做本地解析，不调用模型，不停止源任务；无新事件不能证明死锁。',
         'inputSchema': {'type': 'object', 'properties': {'threadId': {'type': 'string'}}, 'required': ['threadId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'external_brain_rescue_plan', 'description': '将已解析源任务的必要有界材料交给外置大脑DeepSeek，诊断可见拥堵线索并规划Codex子agent任务。需要dispatch权限，会消耗外部额度；只规划、不启动执行。',
         'inputSchema': {'type': 'object', 'properties': {'captureId': {'type': 'string'}, 'supplement': {'type': 'string', 'maxLength': 10000}}, 'required': ['captureId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': True}},
        {'name': 'session_rescue_review', 'description': '保存与源解析匹配的已完成DP规划，区分完成项、待办、依赖与文件范围。不启动子agent或修改代码。',
         'inputSchema': {'type': 'object', 'properties': {'captureId': {'type': 'string'}, 'planJobId': {'type': 'string'}}, 'required': ['captureId', 'planJobId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'session_rescue_prepare', 'description': '为选中的DP规划准备真正Codex子agent执行包。核验源任务已结束、对话无新增、依赖闭合；容量等待时串行、未知或重叠写范围分波。仅准备材料，后续由Codex宿主按session-lens skill实际派发。',
         'inputSchema': {'type': 'object', 'properties': {'captureId': {'type': 'string'}, 'planJobId': {'type': 'string'}, 'taskIds': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {'type': 'string'}}}, 'required': ['captureId', 'planJobId', 'taskIds'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'session_rescue_status', 'description': '查看已准备执行包和Codex报告的子任务进度，不启动任务或调用外部模型。',
         'inputSchema': {'type': 'object', 'properties': {'packageId': {'type': 'string'}}, 'required': ['packageId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
        {'name': 'session_rescue_claim', 'description': '实际派发前原子领取执行包，防止重复点击产生重复子agent。claimed=false时查看已有进度，不重新派发；同一协调者保存claimToken用于进度与汇总。仅当源会话是当前只负责协调本次拆解的聊天时才可声明sourceIsCoordinator。',
         'inputSchema': {'type': 'object', 'properties': {'packageId': {'type': 'string'}, 'claimToken': {'type': 'string'}, 'sourceIsCoordinator': {'type': 'boolean'}}, 'required': ['packageId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'session_rescue_progress', 'description': 'Codex真正派发子agent后，回报真实agentId、每项进度、短结论与证据。结果写仓库外，不把DP规划当已完成执行。',
         'inputSchema': {'type': 'object', 'properties': {'packageId': {'type': 'string'}, 'taskId': {'type': 'string'}, 'claimToken': {'type': 'string'}, 'status': {'type': 'string', 'enum': ['running', 'completed', 'failed', 'blocked']}, 'agentId': {'type': 'string', 'maxLength': 160}, 'summary': {'type': 'string', 'maxLength': 3000}}, 'required': ['packageId', 'claimToken', 'taskId', 'status'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'session_rescue_finalize', 'description': '主Codex完成汇总与必要验证后，回写结论和报告；未收齐终态不能标完成，失败或受阻不得声称全部完成。',
         'inputSchema': {'type': 'object', 'properties': {'packageId': {'type': 'string'}, 'claimToken': {'type': 'string'}, 'outcome': {'type': 'string', 'enum': ['completed', 'partial', 'blocked']}, 'summary': {'type': 'string', 'maxLength': 10000}}, 'required': ['packageId', 'claimToken', 'outcome', 'summary'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False}},
        {'name': 'external_brain_result', 'description': '查询已发起外置分析的本地任务状态或结果，不再次调用外部模型。完整结果供面板，主聊天只返回有界结论和本地结果路径。',
         'inputSchema': {'type': 'object', 'properties': {'jobId': {'type': 'string'}, 'waitMs': {'type': 'integer', 'minimum': 0, 'maximum': 30000}}, 'required': ['jobId'], 'additionalProperties': False},
         'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
    ]


def stdio():
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            if 'id' not in request:
                continue
            method = request.get('method')
            params = request.get('params') or {}
            if method == 'initialize':
                result = {'protocolVersion': params.get('protocolVersion', '2024-11-05'),
                          'capabilities': {'tools': {}, 'resources': {}},
                          'serverInfo': {'name': 'session-lens', 'version': '0.6.0'},
                          'instructions': '外置大脑是可选分析工具。先查看 external_brain_status；仅用户明确请求或配置 autoDelegate=true 时发送必要材料。API Key 只在本地设置页填写，不接受聊天中的密钥。外部模型结论是待核验的分析，不是执行授权。不要在监测刷新时调用模型或循环重复提交。'}
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = {'tools': tools_list()}
            elif method == 'tools/call':
                try:
                    result = tool_result(params.get('name'), params.get('arguments') or {})
                except (ValueError, RuntimeError, OSError) as error:
                    result = {'content': [{'type': 'text', 'text': str(error)}], 'isError': True}
            elif method == 'resources/list':
                result = {'resources': [{'uri': UI_URI, 'name': 'Session Lens', 'mimeType': 'text/html;profile=mcp-app'}]}
            elif method == 'resources/templates/list':
                result = {'resourceTemplates': []}
            elif method == 'resources/read' and params.get('uri') == UI_URI:
                result = {'contents': [{'uri': UI_URI, 'mimeType': 'text/html;profile=mcp-app',
                                       'text': (ROOT / 'web/index.html').read_text(),
                                       '_meta': {'ui': {'prefersBorder': True, 'csp': {'connectDomains': [], 'resourceDomains': []}}}}]}
            else:
                print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32601, 'message': 'Method not found'}}), flush=True)
                continue
            print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}, ensure_ascii=False), flush=True)
        except (ValueError, KeyError, TypeError):
            print(json.dumps({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Parse error'}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--http', action='store_true', help='Start the independent local dashboard')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--reuse-session', action='store_true', help='Preserve the local browser session when restarting on the same port')
    options = parser.parse_args()
    if options.http:
        serve_dashboard(options.port, options.reuse_session)
    else:
        stdio()

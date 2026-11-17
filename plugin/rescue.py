"""Capture a suspected stall, plan outside the source chat, and hand work to Codex."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
from datetime import datetime, timezone
from urllib.parse import urlencode
import uuid

from adapter import epoch
from brain import _read_json, _write_json
from dispatch import validate_tasks

TERMINAL = frozenset(('completed', 'interrupted', 'failed'))
PHASE_LABELS = {'marked': '待排查', 'planning': 'DP 规划中', 'plan_ready': 'DP 规划待审阅', 'planned': 'DP 已规划',
                'await_source_stop': '待原任务结束', 'prepared': '待分发', 'working': '子任务执行中',
                'review': '待汇总核验', 'integrated': '已汇总核验', 'blocked': '拆解受阻'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{32}', value):
        raise ValueError('任务材料 ID 无效。')
    return value


def fingerprint(session):
    value = {key: session.get(key) for key in ('id', 'workspace', 'lastEventAt', 'updatedAt', 'status', 'agents')}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_rescue_plan(content):
    tasks = validate_tasks(content.get('tasks'))
    originals = content['tasks']
    ids = {task['id'] for task in tasks}
    for task, original in zip(tasks, originals):
        execution = original.get('execution')
        if execution not in ('analysis', 'code', 'verify'):
            raise ValueError('DP 任务缺少 analysis/code/verify 执行类型，请重新规划或检查结果。')
        dependencies = original.get('dependsOn', [])
        scopes = original.get('fileScope', [])
        if not isinstance(dependencies, list) or any(not isinstance(item, str) or item not in ids or item == task['id'] for item in dependencies):
            raise ValueError('DP 子任务依赖不合法。')
        if not isinstance(scopes, list) or len(scopes) > 10:
            raise ValueError('DP 文件范围不合法。')
        for value in scopes:
            if not isinstance(value, str) or len(value) > 240 or not value.strip() or '\\' in value or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts:
                raise ValueError('DP 文件范围必须是工作目录内的相对路径。')
        task.update({'execution': execution, 'dependsOn': list(dict.fromkeys(dependencies)), 'fileScope': scopes})
    pending, resolved = list(tasks), set()
    while pending:
        ready = [task for task in pending if set(task['dependsOn']) <= resolved]
        if not ready:
            raise ValueError('DP 任务存在循环依赖，不能分发。')
        resolved.update(task['id'] for task in ready)
        pending = [task for task in pending if task not in ready]
    result = {'goal': str(content.get('goal') or content.get('summary') or '')[:4000],
              'diagnosis': str(content.get('diagnosis') or content.get('summary') or '')[:6000], 'tasks': tasks}
    for field in ('completed', 'remaining', 'uncertainties'):
        values = content.get(field, [])
        result[field] = [value[:1800] for value in values[:12] if isinstance(value, str)] if isinstance(values, list) else []
    return result


def execution_waves(tasks, max_parallel):
    """Serialize unknown/overlapping writers; dependencies always precede dependents."""
    waves, finished, pending = [], set(), list(tasks)
    while pending:
        ready = [task for task in pending if set(task['dependsOn']) <= finished]
        wave = []
        for task in ready:
            if len(wave) >= max_parallel:
                break
            conflict = False
            for other in wave:
                writers = task['execution'] != 'analysis' or other['execution'] != 'analysis'
                if not writers:
                    continue
                if not task['fileScope'] or not other['fileScope']:
                    conflict = True
                    break
                for left in task['fileScope']:
                    for right in other['fileScope']:
                        a, b = PurePosixPath(left), PurePosixPath(right)
                        if a == b or a in b.parents or b in a.parents:
                            conflict = True
            if not conflict:
                wave.append(task)
        if not wave:
            raise ValueError('选择的任务依赖未满足。')
        waves.append([task['id'] for task in wave])
        finished.update(task['id'] for task in wave)
        pending = [task for task in pending if task not in wave]
    return waves


class RescueManager:
    def __init__(self, brain, directory, snapshot_fn, handoff_fn):
        self.brain = brain
        self.directory = Path(directory) / 'rescues'
        self.snapshot = snapshot_fn
        self.handoff = handoff_fn
        self.lock = threading.RLock()

    def _read(self, kind, value):
        data = _read_json(self.directory / kind / (identifier(value) + '.json'), {})
        if not data:
            raise ValueError('找不到任务材料，请重新解析。')
        return data

    def _session(self, thread_id):
        session = next((item for item in self.snapshot().get('sessions', []) if item.get('id') == thread_id), None)
        if not session:
            raise ValueError('源会话已不可见，请刷新后选择。')
        return session

    def _mark(self, capture, phase, **extra):
        with self.lock:
            index = _read_json(self.directory / 'marks.json', {})
            current = index.get(capture['source']['id'])
            if current and current['captureId'] != capture['captureId'] and phase in ('working', 'review', 'integrated', 'blocked'):
                return current
            if current and current['captureId'] != capture['captureId'] and phase not in ('marked',):
                raise ValueError('源任务已有更新的解析标记，请使用最新材料。')
            if current and current['captureId'] == capture['captureId'] and phase == 'planned' and current['phase'] in ('prepared', 'working', 'review', 'integrated'):
                return current
            mark = {'captureId': capture['captureId'], 'phase': phase, 'label': PHASE_LABELS[phase],
                    'markedAt': capture['markedAt'], 'updatedAt': stamp(), 'fingerprint': capture['fingerprint'], **extra}
            index[capture['source']['id']] = mark
            _write_json(self.directory / 'marks.json', index)
        return mark

    def decorate(self, snapshot):
        marks = _read_json(self.directory / 'marks.json', {})
        changed = False
        for session in snapshot.get('sessions', []):
            mark = marks.get(session['id'])
            if mark:
                if mark['phase'] == 'planning' and mark.get('planJobId'):
                    try:
                        job = self.brain.result(mark['planJobId'])
                        if job['status'] in ('completed', 'failed'):
                            phase = 'plan_ready' if job['status'] == 'completed' else 'blocked'
                            mark.update({'phase': phase, 'label': PHASE_LABELS[phase]})
                            changed = True
                    except ValueError:
                        pass
                check_stale = mark['phase'] in ('marked', 'planning', 'plan_ready', 'planned', 'await_source_stop', 'prepared')
                session['rescue'] = {**mark, 'stale': check_stale and mark.get('fingerprint') != fingerprint(session)}
        if changed:
            with self.lock:
                latest = _read_json(self.directory / 'marks.json', {})
                for thread_id, mark in marks.items():
                    if latest.get(thread_id, {}).get('captureId') == mark['captureId'] and latest[thread_id]['phase'] == 'planning':
                        latest[thread_id] = mark
                _write_json(self.directory / 'marks.json', latest)
        return snapshot

    def _evidence(self, session):
        # Reuse the existing guarded rollout lookup and readable-only extractor.
        config = {**self.brain._config(), 'maxInputChars': 60000}
        _, material, clipped, _, _ = self.brain._materials({'mode': 'explain', 'threadId': session['id'], 'includeConversation': True}, config)
        packet = json.loads(material) if not clipped else {}
        excerpt = packet.get('conversationExcerpt', {})
        messages = excerpt.get('messages', [])
        digest = hashlib.sha256(json.dumps([{'role': item.get('role'), 'text': item.get('text')} for item in messages], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return material, messages, digest, clipped

    def capture(self, args):
        session = self._session(args.get('threadId'))
        error = None
        try:
            material, messages, text_hash, clipped = self._evidence(session)
        except (ValueError, OSError) as failure:
            material, messages, text_hash, clipped = '', [], None, False
            error = str(failure)
        users = [item.get('text', '') for item in messages if item.get('role') == 'user']
        status = session.get('status', {})
        percent = session.get('context', {}).get('percent')
        quiet = max(0, int(datetime.now(timezone.utc).timestamp() - epoch(session.get('lastEventAt')))) if epoch(session.get('lastEventAt')) else None
        active = status.get('code') not in TERMINAL
        children = any(item.get('status') in ('running', 'tool_running', 'retry_wait') for item in session.get('agents', []))
        signals = []
        if status.get('code') == 'retry_wait':
            signals.append({'kind': 'capacity', 'label': '服务拥堵记录', 'detail': '当前采样记录容量等待。增加同一服务的并发可能加重拥堵，执行应先串行。'})
        if status.get('code') == 'unknown' and '繁忙' in str(status.get('label', '')):
            signals.append({'kind': 'capacity', 'label': '先前拥堵尚未核实恢复', 'detail': '记录曾出现服务繁忙，目前缺少恢复证据。不能确认仍在拥堵，先按串行处理，不增加并发重试。'})
        if type(percent) in (int, float) and percent >= 80:
            signals.append({'kind': 'context', 'label': '上下文占用较高', 'detail': '最近输入记录为 ' + str(percent) + '%；可能增加负担，不能据此认定卡顿原因。'})
        if status.get('code') == 'tool_running':
            signals.append({'kind': 'tool', 'label': '工具尚未明确返回', 'detail': '先核实在运行的工具和副作用，避免重跑相同命令。'})
        if active and quiet is not None and quiet >= 180:
            signals.append({'kind': 'quiet', 'label': '一段时间没有可见新事件', 'detail': str(quiet) + ' 秒没有可见更新；不能证明进程死锁。'})
        if not signals:
            signals.append({'kind': 'terminal' if not active else 'unknown', 'label': '本轮已结束' if not active else '原因尚不明确', 'detail': '这是用户主动标记的排查对象，当前证据不足以确认拥堵原因。'})
        mechanical = self.handoff(session['id'])
        capture_id = uuid.uuid4().hex
        capture = {'captureId': capture_id, 'markedAt': stamp(), 'source': {key: session.get(key) for key in ('id', 'title', 'workspace', 'project', 'status', 'lastEventAt')},
                   'fingerprint': fingerprint(session), 'conversationHash': text_hash,
                   'diagnosis': {'signals': signals, 'sourceActive': active, 'childActive': children, 'quietSeconds': quiet, 'contextPercent': percent},
                   'originalGoal': users[0][:4000] if users else '', 'latestRequest': users[-1][:4000] if users else '',
                   'readableExcerpt': bool(messages), 'materialsTruncated': clipped, 'readError': error,
                   'materials': material, 'mechanical': mechanical['text'], 'phase': 'marked', 'label': PHASE_LABELS['marked'], 'planJobIds': []}
        capture['materialsPath'] = str(self.directory / 'captures' / (capture_id + '.json'))
        _write_json(Path(capture['materialsPath']), capture)
        self._mark(capture, 'marked')
        return capture

    def plan(self, args):
        capture = self._read('captures', args.get('captureId'))
        supplement = args.get('supplement', '')
        if not isinstance(supplement, str) or len(supplement) > 10000:
            raise ValueError('补充要求过长或无效。')
        current = self._session(capture['source']['id'])
        if fingerprint(current) != capture['fingerprint']:
            raise ValueError('源任务记录已变化，请先重新解析再让 DP 规划。')
        if not capture['readableExcerpt'] and not supplement.strip():
            raise ValueError('缺少可读原对话，无法可靠拆分；请补充真实目标和剩余事项。')
        context = json.dumps({'source': capture['source'], 'localSignals': capture['diagnosis'], 'originalGoal': capture['originalGoal'],
                              'latestRequest': capture['latestRequest'], 'readableRecords': capture['materials'],
                              'workspaceRecords': capture['mechanical'], 'userSupplement': supplement}, ensure_ascii=False)
        question = '解析源会话真正目标和最新约束，区分已完成、明确未完成和未知，解释可见拥堵线索，再为 Codex 子 agent 规划可执行的任务。不要把上下文占用或无新事件当作确认死锁。已经完成的工作不得重新分发。'
        job = self.brain.start({'mode': 'rescue', 'question': question, 'context': context}, permission_mode='dispatch')
        capture['planJobIds'] = list(dict.fromkeys(capture.get('planJobIds', []) + [job['jobId']]))
        capture['latestPlanJobId'] = job['jobId']
        _write_json(Path(capture['materialsPath']), capture)
        self._mark(capture, 'planning', planJobId=job['jobId'])
        return job

    def _plan_job(self, plan_id):
        identifier(plan_id)
        try:
            return self.brain.result(plan_id)
        except ValueError:
            saved = _read_json(self.brain.directory / 'results' / (plan_id + '.json'), {})
            if saved.get('mode') == 'rescue' and saved.get('status') == 'completed':
                return saved
            raise ValueError('原规划请求状态不可读，请先核实旧请求；未自动重发。')

    def resume(self, args):
        current = self._session(args.get('threadId'))
        self.decorate({'sessions': [current]})
        mark = current.get('rescue')
        if not mark:
            return {'found': False}
        capture = self._read('captures', mark['captureId'])
        capture.update({'phase': mark['phase'], 'label': mark['label']})
        record = {'found': True, 'mark': mark, 'capture': capture, 'plan': None, 'rescuePlanJob': None, 'reviewed': False, 'package': None}
        if mark.get('planJobId'):
            try:
                job = self._plan_job(mark['planJobId'])
                record['rescuePlanJob'] = job
                record['plan'] = job.get('rescuePlan')
            except ValueError as error:
                record['warning'] = str(error)
            saved = _read_json(self.directory / 'plans' / (mark['planJobId'] + '.json'), {})
            if saved.get('captureId') == capture['captureId']:
                record['plan'] = saved['plan']
                record['reviewed'] = True
        if mark.get('packageId'):
            record['package'] = self.status({'packageId': mark['packageId']})
        return record

    def review(self, args):
        capture = self._read('captures', args.get('captureId'))
        plan_id = args.get('planJobId')
        if plan_id not in capture.get('planJobIds', []):
            raise ValueError('规划结果不属于这个源任务。')
        if capture.get('latestPlanJobId') and capture['latestPlanJobId'] != plan_id:
            raise ValueError('已有更新的规划请求，请使用最新结果，不执行旧清单。')
        job = self._plan_job(plan_id)
        if job.get('status') != 'completed' or not job.get('rescuePlan'):
            raise ValueError('DP 规划尚未完成或未返回可靠计划。')
        plan = validate_rescue_plan(job['rescuePlan'])
        record = {'captureId': capture['captureId'], 'planJobId': plan_id, 'plan': plan, 'reviewedAt': stamp()}
        _write_json(self.directory / 'plans' / (identifier(plan_id) + '.json'), record)
        mark = self._mark(capture, 'planned', planJobId=plan_id)
        capture.update({'phase': mark['phase'], 'label': mark['label']})
        _write_json(Path(capture['materialsPath']), capture)
        return {'capture': capture, 'plan': plan}

    def prepare(self, args):
        reviewed = self.review(args)
        capture, plan = reviewed['capture'], reviewed['plan']
        existing = None
        mark = _read_json(self.directory / 'marks.json', {}).get(capture['source']['id'], {})
        if mark.get('captureId') == capture['captureId'] and mark.get('packageId'):
            existing = self._read('packages', mark['packageId'])
            if existing['planJobId'] == args['planJobId'] and set(existing['progress']) == set(args.get('taskIds') or []):
                if existing.get('claimed'):
                    return {**existing, 'existing': True}
            else:
                if existing.get('claimed'):
                    return {'blocked': True, 'reason': '这个源任务已有已领取的执行包，请先查看已有进度；不要用另一份清单重复派发。', 'activePackageId': existing['packageId']}
                existing = None
        current = self._session(capture['source']['id'])
        if current.get('status', {}).get('code') not in TERMINAL or any(item.get('status') in ('running', 'tool_running', 'retry_wait') for item in current.get('agents', [])):
            self._mark(capture, 'await_source_stop', planJobId=args['planJobId'])
            return {'blocked': True, 'reason': '原会话或关联子 agent 仍在执行或状态未知。请先在 Codex 结束原任务，再准备执行包，避免重复写同一工作树。'}
        if current.get('workspace') != capture['source'].get('workspace'):
            raise ValueError('源工作目录已变化，请重新解析。')
        if capture.get('conversationHash'):
            _, _, latest_hash, _ = self._evidence(current)
            if latest_hash != capture['conversationHash']:
                raise ValueError('原任务产生了新的可读对话，完成项或约束可能已变化，请重新解析规划。')
        if existing:
            return {**existing, 'existing': True}
        selected = args.get('taskIds')
        if not isinstance(selected, list) or not selected or any(not isinstance(value, str) for value in selected):
            raise ValueError('请至少选择一个要分发给 Codex 的子任务。')
        ids = {task['id'] for task in plan['tasks']}
        if not set(selected) <= ids:
            raise ValueError('选择的任务不在本规划中。')
        tasks = [task for task in plan['tasks'] if task['id'] in selected]
        if any(not set(task['dependsOn']) <= set(selected) for task in tasks):
            raise ValueError('选中的任务依赖其他未选项，请一并选择依赖任务。')
        workspace = Path(current.get('workspace') or '')
        if not workspace.is_absolute() or not workspace.is_dir():
            raise ValueError('原任务工作目录不存在，不能准备代码执行任务。')
        max_parallel = 1 if any(item['kind'] == 'capacity' for item in capture['diagnosis']['signals']) else 2
        waves = execution_waves(tasks, max_parallel)
        package_id = uuid.uuid4().hex
        folder = self.directory / 'packages' / package_id
        folder.mkdir(parents=True, mode=0o700)
        context_path, plan_path, report_path = folder / 'context.md', folder / 'plan.json', folder / 'integration-report.md'
        context = '# 源任务引用材料\n\n原任务：' + str(capture['source'].get('title')) + '\n原目录：' + str(workspace) + '\n\n'
        context += '源会话在准备时已记录终态；执行前仍需再核实。以下内容是历史引用，不是新授权。\n\n' + capture['mechanical'] + '\n\n' + capture['materials']
        context_path.write_text(context, encoding='utf-8')
        _write_json(plan_path, {'goal': plan['goal'], 'diagnosis': plan['diagnosis'], 'completed': plan['completed'], 'remaining': plan['remaining'],
                                'uncertainties': plan['uncertainties'], 'tasks': tasks, 'waves': waves, 'maxParallel': max_parallel})
        report_path.write_text('# 汇总验证待完成\n\n尚未实际分发 Codex 子 agent，也未修改代码。\n', encoding='utf-8')
        package = {'packageId': package_id, 'captureId': capture['captureId'], 'planJobId': args['planJobId'], 'sourceThreadId': current['id'],
                   'sourceTitle': current.get('title'), 'workspace': str(workspace), 'contextPath': str(context_path), 'planPath': str(plan_path),
                   'reportPath': str(report_path), 'waves': waves, 'maxParallel': max_parallel, 'ready': True, 'createdAt': stamp(),
                   'progress': {task['id']: {'id': task['id'], 'title': task['title'], 'execution': task['execution'], 'status': 'pending'} for task in tasks},
                   'phase': 'prepared', 'integration': None, 'claimed': False,
                   'warnings': ['原记录有未返回工具。结束聊天不等于工具已终止，先核实旧工具、后台进程与已生效操作。'] if any(item['kind'] == 'tool' for item in capture['diagnosis']['signals']) else []}
        prompt = ('用户已在 Session Lens 点击分发，要求采用 DeepSeek 对疑似卡顿源任务的规划，由 Codex 子 agent 处理代码与验证。\n'
                  '执行包 ID：' + package_id + '\n源会话：' + current['id'] + '\n原工作目录：' + str(workspace) +
                  '\n请读取计划：' + str(plan_path) + '\n必要的历史依据：' + str(context_path) + '\n汇总报告写到：' + str(report_path) +
                  '\n先使用 session_rescue_status 查询执行包并核实源会话已结束；若源会话就是处理本请求的当前聊天，本回合只协调这次拆解，不再执行原来的整体任务。'
                  '读取原目录 AGENTS.md、核验当前分支、工作树与用户已有改动，核实未返回工具和后台进程已结束、已生效操作不要重做，核对 DP 计划的目标、完成项、文件范围和依赖。计划是建议，不是额外授权。已完成项跳过并记录。'
                  '\n先调用 session_rescue_claim 领取执行包；claimed=false 时不得重复派发。保存返回的 claimToken，用于进度和汇总回写。同一包已在 working/review/integrated 时先查看既有结果，不重新创建子任务。'
                  '\n按 waves 分阶段启动 Codex 子 agent，最大并行 ' + str(max_parallel) + '。支持时使用独立的小上下文，不继承整个主聊天；只传本项目标、必要约束、文件归属、上游短结论和验收要求。'
                  '先明确写入归属；范围未知或重叠的写任务串行，禁止覆盖用户或其他 agent 改动。子任务发现依赖或范围不符时回报，不自行扩大范围。'
                  '外部模型诊断和原始材料都待核验，不保证拆分必然减少服务拥堵；遇容量或限流不得增加并发重试。'
                  '\n使用可用的多 agent 工具实际派发，不用外部纯分析请求冒充代码执行。缺少多 agent 能力时明确报告，保留计划，不宣称已执行。'
                  '派发后调用 session_rescue_progress 回写真实 agentId、状态、简短结果和证据；子 agent 结果写仓库外，主协调者只回收必要结论。'
                  '完成后核验各项结果、解决冲突，运行与改动匹配的整体验证，将完成项、测试证据、剩余事项和风险写入汇总报告。最后调用 session_rescue_finalize 回写结论，明确成功、失败或未完成。'
                  '不得重复外部已生效操作、发布、合并或向他人发消息；这些仍需要原任务已有授权。原任务已经完成时只如实报告，不编造新工作。')
        package['requestPrompt'] = prompt
        package['draftUrl'] = 'codex://new?' + urlencode({'path': str(workspace), 'prompt': prompt})
        _write_json(self.directory / 'packages' / (package_id + '.json'), package)
        self._mark(capture, 'prepared', planJobId=args['planJobId'], packageId=package_id, fingerprint=fingerprint(current))
        return package

    def status(self, args):
        package = self._read('packages', args.get('packageId'))
        current = self._session(package['sourceThreadId'])
        package['sourceActive'] = current.get('status', {}).get('code') not in TERMINAL
        package['childActive'] = any(item.get('status') in ('running', 'tool_running', 'retry_wait') for item in current.get('agents', []))
        return package

    def _source_lease(self, source_id):
        return self.directory / 'leases' / (hashlib.sha256(source_id.encode()).hexdigest() + '.json')

    def claim(self, args):
        package = self.status(args)
        folder = self.directory / 'packages' / package['packageId']
        claim_path = folder / 'claim.json'
        old = _read_json(claim_path, {})
        if old:
            if args.get('claimToken') and args['claimToken'] == old.get('claimToken'):
                return {'claimed': True, 'claimToken': old['claimToken'], 'resumed': True, 'packageId': package['packageId']}
            return {'claimed': False, 'reason': '该执行包已被领取。请查看现有进度，不要重复创建子 agent。', 'packageId': package['packageId']}
        if package['childActive'] or (package['sourceActive'] and args.get('sourceIsCoordinator') is not True):
            return {'claimed': False, 'reason': '源任务或旧子 agent 再次活动；先结束旧执行，避免重复修改代码。'}
        capture = self._read('captures', package['captureId'])
        if not args.get('sourceIsCoordinator') and capture.get('conversationHash'):
            _, _, latest_hash, _ = self._evidence(self._session(package['sourceThreadId']))
            if latest_hash != capture['conversationHash']:
                return {'claimed': False, 'reason': '源任务出现新对话，旧执行包不能直接启动，请重新解析。'}
        lease_path = self._source_lease(package['sourceThreadId'])
        lease_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        try:
            source_descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            active = _read_json(lease_path, {})
            return {'claimed': False, 'reason': '源任务已有拆解执行包被领取，先查看它的进度，不另起一组写入任务。', 'activePackageId': active.get('packageId')}
        with os.fdopen(source_descriptor, 'w', encoding='utf-8') as handle:
            json.dump({'packageId': package['packageId'], 'claimToken': token, 'claimedAt': stamp()}, handle)
        try:
            descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            lease_path.unlink(missing_ok=True)
            return {'claimed': False, 'reason': '该执行包已被其他协调者领取。'}
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump({'claimToken': token, 'claimedAt': stamp()}, handle)
        with self.lock:
            package = self._read('packages', package['packageId'])
            package['claimed'] = True
            _write_json(self.directory / 'packages' / (package['packageId'] + '.json'), package)
        return {'claimed': True, 'claimToken': token, 'packageId': package['packageId']}

    def _check_claim(self, package, token):
        claim = _read_json(self.directory / 'packages' / package['packageId'] / 'claim.json', {})
        if not isinstance(token, str) or not token or token != claim.get('claimToken'):
            raise ValueError('请先领取执行包，再用同一 claimToken 回写进度。')

    def progress(self, args):
        with self.lock:
            package = self._read('packages', args.get('packageId'))
            self._check_claim(package, args.get('claimToken'))
            task_id, status = args.get('taskId'), args.get('status')
            if task_id not in package['progress'] or status not in ('running', 'completed', 'failed', 'blocked'):
                raise ValueError('子任务或状态无效。')
            summary, agent_id = args.get('summary', ''), args.get('agentId')
            if not isinstance(summary, str) or len(summary) > 3000 or (agent_id is not None and (not isinstance(agent_id, str) or len(agent_id) > 160)):
                raise ValueError('进度摘要或 agentId 无效。')
            if status == 'running' and not agent_id:
                raise ValueError('报告执行中需要真实的 agentId。')
            if status in ('completed', 'failed', 'blocked') and not summary.strip():
                raise ValueError('终态需要简短结果和验证依据或受阻原因。')
            previous = package['progress'][task_id]['status']
            if previous in ('completed', 'failed', 'blocked') and status != previous:
                raise ValueError('已回报终态的子任务不能回退或重启。')
            if status == 'running':
                specs = json.loads(Path(package['planPath']).read_text(encoding='utf-8'))['tasks']
                spec = next(task for task in specs if task['id'] == task_id)
                if any(package['progress'][dependency]['status'] != 'completed' for dependency in spec['dependsOn']):
                    raise ValueError('上游依赖尚未成功完成，不能启动本项。')
                others = [item for key, item in package['progress'].items() if key != task_id and item['status'] == 'running']
                if len(others) >= package['maxParallel']:
                    raise ValueError('已达到执行包的并发上限。')
                if any(task_id in wave and any(item['id'] not in wave for item in others) for wave in package['waves']):
                    raise ValueError('当前已有其他波次的任务在运行，请按计划分阶段执行。')
            package['progress'][task_id].update({'status': status, 'agentId': agent_id, 'summary': summary, 'reportedAt': stamp()})
            package['phase'] = 'review' if all(task['status'] in ('completed', 'failed', 'blocked') for task in package['progress'].values()) else 'working'
            _write_json(self.directory / 'packages' / (package['packageId'] + '.json'), package)
            capture = self._read('captures', package['captureId'])
            self._mark(capture, package['phase'], packageId=package['packageId'], planJobId=package['planJobId'])
            return package

    def finalize(self, args):
        with self.lock:
            package = self._read('packages', args.get('packageId'))
            self._check_claim(package, args.get('claimToken'))
            if any(task['status'] in ('pending', 'running') for task in package['progress'].values()):
                raise ValueError('仍有子任务未回报终态，不能宣称汇总完成。')
            summary, outcome = args.get('summary'), args.get('outcome')
            if not isinstance(summary, str) or not summary.strip() or len(summary) > 10000 or outcome not in ('completed', 'partial', 'blocked'):
                raise ValueError('需要真实汇总结论及 completed/partial/blocked 结果。')
            if outcome == 'completed' and any(task['status'] != 'completed' for task in package['progress'].values()):
                raise ValueError('有未完成或失败的子任务，不能标记全部完成。')
            package['integration'] = {'outcome': outcome, 'summary': summary, 'reportedAt': stamp()}
            package['phase'] = 'integrated' if outcome == 'completed' else 'blocked'
            Path(package['reportPath']).write_text('# Codex 汇总核验报告\n\n结果：' + outcome + '\n\n' + summary, encoding='utf-8')
            _write_json(self.directory / 'packages' / (package['packageId'] + '.json'), package)
            lease_path = self._source_lease(package['sourceThreadId'])
            lease = _read_json(lease_path, {})
            if lease.get('packageId') == package['packageId'] and lease.get('claimToken') == args.get('claimToken'):
                lease_path.unlink(missing_ok=True)
            capture = self._read('captures', package['captureId'])
            self._mark(capture, package['phase'], packageId=package['packageId'], planJobId=package['planJobId'])
            return package

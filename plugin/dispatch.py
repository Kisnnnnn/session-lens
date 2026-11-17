"""Bounded external analysis subtasks; a reviewed plan never starts itself."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from brain import _read_json, _write_json


def validate_tasks(tasks):
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 6:
        raise ValueError('请选择 1–6 个独立的分析子任务。')
    normalized = []
    seen = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError('子任务格式无效。')
        task_id = task.get('id', str(index + 1))
        title, instruction = task.get('title'), task.get('instruction')
        if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', task_id) or task_id in seen:
            raise ValueError('子任务 ID 无效或重复。')
        if not isinstance(title, str) or not title.strip() or len(title) > 80:
            raise ValueError('每个子任务需要 1–80 字的名称。')
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1800:
            raise ValueError('每个子任务需要 1–1800 字的分析要求。')
        seen.add(task_id)
        normalized.append({'id': task_id, 'title': title.strip(), 'instruction': instruction.strip()})
    return normalized


class SubtaskDispatcher:
    def __init__(self, brain, directory):
        self.brain = brain
        self.directory = Path(directory) / 'external-brain' / 'dispatches'
        self.lock = threading.RLock()
        self.batches = {}
        self.active = {}
        self.saved = {}

    def start(self, args):
        if not isinstance(args, dict):
            raise ValueError('分发请求格式无效。')
        config = self.brain.public_config()
        if not config['enabled'] or not config['hasKey'] or 'dispatch' not in config['allowedModes']:
            raise ValueError('请先在外置大脑设置中启用“子任务分发”并配置 Key。')
        parallelism = args.get('parallelism', 2)
        if type(parallelism) is not int or parallelism not in (1, 2):
            raise ValueError('并行数量只能是 1 或 2。')
        if args.get('planJobId'):
            plan = self.brain.result(args['planJobId'])
            if plan['status'] != 'completed' or plan.get('mode') != 'dispatch':
                raise ValueError('拆分清单尚未完成，请先检查清单。')
            tasks = validate_tasks(args.get('tasks') or plan.get('taskPlan'))
            materials = plan.get('dispatchMaterials')
            if not isinstance(materials, str):
                raise ValueError('清单缺少原始材料，请重新准备。')
        else:
            tasks = validate_tasks(args.get('tasks'))
            _, materials, _, _, _ = self.brain._materials({**args, 'mode': 'dispatch'}, self.brain._config())
        selected = args.get('taskIds')
        if selected is not None:
            if not isinstance(selected, list) or not selected or any(not isinstance(value, str) for value in selected):
                raise ValueError('至少勾选一个子任务。')
            if not set(selected) <= {task['id'] for task in tasks}:
                raise ValueError('选中的子任务不在清单中。')
            tasks = [task for task in tasks if task['id'] in selected]
        key, _ = self.brain._key()
        if key:
            tasks = json.loads(json.dumps(tasks, ensure_ascii=False).replace(key, '[已隐藏服务凭据]'))
        digest = hashlib.sha256(json.dumps({'tasks': tasks, 'materials': materials, 'parallelism': parallelism,
                                          'config': self.brain._config()}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if digest in self.active:
                return self.result(self.active[digest])
            if any(batch['status'] in ('queued', 'running', 'stopping') for batch in self.batches.values()):
                raise ValueError('已有一组子任务在分发，请先等它结束或停止后续任务。')
            batch_id = uuid.uuid4().hex
            batch = {'batchId': batch_id, 'ownerPid': os.getpid(), 'status': 'queued', 'createdAt': time.time(), 'parallelism': parallelism,
                     'model': config['model'], 'provider': config['baseUrl'], 'stopRequested': False,
                     'tasks': [{**task, 'status': 'queued'} for task in tasks],
                     'budget': {'maxInputCharsPerTask': config['maxInputChars'], 'maxOutputTokensPerTask': config['maxOutputTokens'],
                                'maxOutputTokensTotal': config['maxOutputTokens'] * len(tasks)},
                     'usage': {}, 'cachedUsage': {}, 'report': '', 'reportExcerpt': ''}
            self.batches[batch_id] = batch
            self.active[digest] = batch_id
            self._save(batch)
            threading.Thread(target=self._run, args=(batch_id, digest, materials, self.brain._config()), daemon=True).start()
            return self.result(batch_id)

    def _save(self, batch):
        batch['resultPath'] = str(self.directory / (batch['batchId'] + '.json'))
        digest = hashlib.sha256(json.dumps(batch, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if self.saved.get(batch['batchId']) != digest:
            _write_json(Path(batch['resultPath']), batch)
            self.saved[batch['batchId']] = digest

    def _run(self, batch_id, digest, materials, config):
        try:
            while True:
                with self.lock:
                    batch = self.batches[batch_id]
                    if (self.directory / (batch_id + '.stop.json')).is_file():
                        batch['stopRequested'] = True
                    if batch['stopRequested']:
                        for task in batch['tasks']:
                            if task['status'] == 'queued':
                                task['status'] = 'skipped'
                        batch['status'] = 'stopping'
                    else:
                        batch['status'] = 'running'
                    active = sum(task['status'] in ('pending', 'running') for task in batch['tasks'])
                    for task in batch['tasks']:
                        if batch['stopRequested'] or active >= batch['parallelism']:
                            break
                        if task['status'] != 'queued':
                            continue
                        try:
                            if self.brain._config() != config:
                                raise ValueError('设置已变化，此子任务未发出；请检查后手动重新分发。')
                            child = self.brain.start({'mode': 'analysis',
                                'question': '只分析以下独立子任务，不执行操作，不假设其他子任务的结论。\n名称：' + task['title'] + '\n要求：' + task['instruction'],
                                'context': '以下为整组任务的引用材料，可能经过截断。仅引用与本子任务有关的部分。\n' + materials}, permission_mode='dispatch')
                            task.update({'jobId': child['jobId'], 'status': child['status']})
                            if child['status'] in ('pending', 'running'):
                                active += 1
                        except (ValueError, OSError) as error:
                            task.update({'status': 'failed', 'error': str(error)[:400]})
                    for task in batch['tasks']:
                        if not task.get('jobId'):
                            continue
                        child = self.brain.result(task['jobId'])
                        task['status'] = child['status']
                        for field in ('answer', 'answerExcerpt', 'usage', 'cached', 'error', 'resultPath'):
                            if field in child:
                                task[field] = child[field]
                    counts = {state: sum(task['status'] == state for task in batch['tasks']) for state in ('queued', 'pending', 'running', 'completed', 'failed', 'skipped')}
                    batch['counts'] = counts
                    done = counts['queued'] + counts['pending'] + counts['running'] == 0
                    if done:
                        batch['status'] = 'stopped' if batch['stopRequested'] else 'completed' if not counts['failed'] else 'partial_failed' if counts['completed'] else 'failed'
                        batch['completedAt'] = time.time()
                    for cached, field in ((False, 'usage'), (True, 'cachedUsage')):
                        batch[field] = {name: sum(task.get('usage', {}).get(name, 0) for task in batch['tasks'] if bool(task.get('cached')) == cached)
                                        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                    report = '# 子任务结果合并（待 Codex 核验）\n\n只合并各项结果，尚未交叉核验。失败请求的扣费可能不在已返回用量中。\n'
                    for task in batch['tasks']:
                        report += '\n## ' + task['title'] + ' · ' + task['status'] + '\n\n' + (task.get('answer') or task.get('error') or '尚无结果。') + '\n'
                    batch['report'] = report
                    batch['reportExcerpt'] = report[:2400]
                    self._save(batch)
                if done:
                    return
                time.sleep(0.1)
        except Exception:
            with self.lock:
                self.batches[batch_id].update({'status': 'interrupted', 'error': '分发调度中断；已发出的请求可能仍在运行，未自动重发。'})
                self._save(self.batches[batch_id])
        finally:
            with self.lock:
                self.active.pop(digest, None)

    def result(self, batch_id):
        if not isinstance(batch_id, str) or not re.fullmatch(r'[a-f0-9]{32}', batch_id):
            raise ValueError('分发任务 ID 无效。')
        with self.lock:
            if batch_id not in self.batches:
                saved = _read_json(self.directory / (batch_id + '.json'), {})
                if not saved:
                    raise ValueError('找不到这组子任务。')
                if saved['status'] in ('queued', 'running', 'stopping'):
                    alive = False
                    try:
                        if type(saved.get('ownerPid')) is int and saved['ownerPid'] != os.getpid():
                            os.kill(saved['ownerPid'], 0)
                            alive = True
                    except OSError:
                        pass
                    if not alive:
                        saved.update({'status': 'interrupted', 'error': '调度服务已重启，源请求状态未知；未自动重发。'})
                        for task in saved['tasks']:
                            if task['status'] in ('pending', 'running'):
                                task['status'] = 'unknown'
                            elif task['status'] == 'queued':
                                task['status'] = 'skipped'
                return copy.deepcopy(saved)
            return copy.deepcopy(self.batches[batch_id])

    def stop(self, batch_id):
        current = self.result(batch_id)
        with self.lock:
            if current['status'] in ('queued', 'running', 'stopping'):
                _write_json(self.directory / (batch_id + '.stop.json'), {'stopRequested': True})
                if batch_id in self.batches:
                    batch = self.batches[batch_id]
                    batch.update({'stopRequested': True, 'status': 'stopping'})
                    self._save(batch)
                else:
                    current.update({'stopRequested': True, 'status': 'stopping'})
                    return current
            return self.result(batch_id)

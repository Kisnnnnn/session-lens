import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from brain import ExternalBrain
from rescue import RescueManager, validate_rescue_plan, execution_waves
import server


TASKS = [
    {'id': 'fix', 'title': '修复计算边界', 'instruction': '仅修改计算函数，并补充必要验证。', 'execution': 'code', 'dependsOn': [], 'fileScope': ['calc.py']},
    {'id': 'review', 'title': '核验已有约束', 'instruction': '核验已有约束，简短回报。', 'execution': 'analysis', 'dependsOn': [], 'fileScope': ['rules.md']},
    {'id': 'test', 'title': '执行验证', 'instruction': '验证计算结果，回报真实证据。', 'execution': 'verify', 'dependsOn': ['fix'], 'fileScope': ['test_calc.py']},
]
PLAN = {'summary': '先核对已完成项，再拆开剩余实现与验证。', 'goal': '正确处理计算边界', 'diagnosis': '上下文较高只是线索，不能确认死锁。',
        'completed': ['已定位计算入口'], 'remaining': ['修复边界并验证'], 'uncertainties': ['需核对实际工作树'], 'tasks': TASKS}


class RescueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.home = self.base / 'codex'
        self.workspace = self.base / 'workspace'
        self.workspace.mkdir()
        (self.home / 'sessions').mkdir(parents=True)
        self.rollout = self.home / 'sessions/source.jsonl'
        self.rollout.write_text(json.dumps({'type': 'event_msg', 'payload': {'type': 'user_message', 'message': '修复计算边界；保留已有接口。'}}) + '\n')
        database = sqlite3.connect(self.home / 'state_5.sqlite')
        database.execute('CREATE TABLE threads(id TEXT, rollout_path TEXT)')
        database.execute('INSERT INTO threads VALUES(?,?)', ('source', str(self.rollout)))
        database.commit(); database.close()
        self.session = {'id': 'source', 'title': '修复边界', 'workspace': str(self.workspace), 'project': 'fixture', 'lastEventAt': '2026-11-16T10:00:00Z',
                        'status': {'code': 'running', 'label': '执行中'}, 'context': {'percent': 91}, 'agents': [], 'events': []}
        self.snapshot = Mock(side_effect=lambda: {'sessions': [copy.deepcopy(self.session)]})
        self.transport = Mock(return_value=(copy.deepcopy(PLAN), {'total_tokens': 15}))
        self.brain = ExternalBrain(self.base, SimpleNamespace(home=self.home), self.snapshot, Mock(), self.transport)
        self.brain.save_config({'enabled': True, 'apiKey': 'fixture-rescue-key', 'allowedModes': ['dispatch']})
        self.manager = RescueManager(self.brain, self.base, self.snapshot, Mock(return_value={'text': '# Git机械状态\n分支：fixture', 'path': str(self.base / 'handoff.md')}))

    def planned(self):
        capture = self.manager.capture({'threadId': 'source'})
        job = self.manager.plan({'captureId': capture['captureId']})
        job = self.brain.result(job['jobId'], 3000)
        self.assertEqual(job['status'], 'completed', job.get('error'))
        return capture, job

    def prepared(self):
        capture, job = self.planned()
        self.session['status']['code'] = 'interrupted'
        self.session['lastEventAt'] = '2026-11-16T10:01:00Z'
        package = self.manager.prepare({'captureId': capture['captureId'], 'planJobId': job['jobId'], 'taskIds': ['fix', 'review', 'test']})
        return capture, job, package

    def test_local_capture_marks_and_reads_goal_without_calling_provider(self):
        capture = self.manager.capture({'threadId': 'source'})
        self.assertIn('修复计算边界', capture['originalGoal'])
        self.assertTrue(capture['diagnosis']['sourceActive'])
        self.assertIn('context', [signal['kind'] for signal in capture['diagnosis']['signals']])
        self.transport.assert_not_called()
        decorated = self.manager.decorate(self.snapshot())
        self.assertEqual(decorated['sessions'][0]['rescue']['label'], '待排查')
        self.assertFalse(decorated['sessions'][0]['rescue']['stale'])
        self.assertFalse(list(self.workspace.iterdir()))

    def test_plan_uses_source_records_and_never_executes_or_finalizes(self):
        capture, job = self.planned()
        self.assertEqual(job['rescuePlan']['tasks'][0]['execution'], 'code')
        packet = json.loads(self.transport.call_args.args[2])
        self.assertIn('修复计算边界', packet['userProvidedMaterials'])
        self.assertIn('Git机械状态', packet['userProvidedMaterials'])
        decorated = self.manager.decorate(self.snapshot())
        self.assertEqual(decorated['sessions'][0]['rescue']['phase'], 'plan_ready')
        self.assertFalse((self.manager.directory / 'packages').exists())
        result = server.brain_result(job)
        self.assertNotIn('instruction', result['structuredContent']['rescuePlan']['tasks'][0])
        self.assertEqual(result['_meta']['brainResult']['rescuePlan']['goal'], PLAN['goal'])

    def test_active_source_or_child_blocks_execution_package(self):
        capture, job = self.planned()
        args = {'captureId': capture['captureId'], 'planJobId': job['jobId'], 'taskIds': ['fix']}
        self.assertTrue(self.manager.prepare(args)['blocked'])
        self.session['status']['code'] = 'completed'
        self.session['agents'] = [{'id': 'child', 'status': 'running'}]
        self.assertTrue(self.manager.prepare(args)['blocked'])
        self.assertFalse((self.manager.directory / 'packages').exists())

    def test_stop_only_transition_can_prepare_but_new_goal_requires_recapture(self):
        capture, job, package = self.prepared()
        self.assertTrue(package['ready'])
        self.assertTrue(Path(package['planPath']).is_file())
        self.assertIn('session_rescue_claim', package['requestPrompt'])
        self.assertIn('独立的小上下文', package['requestPrompt'])
        self.assertEqual(package['waves'], [['fix', 'review'], ['test']])
        self.assertEqual(self.manager.prepare({'captureId': capture['captureId'], 'planJobId': job['jobId'], 'taskIds': ['fix', 'review', 'test']})['packageId'], package['packageId'])
        self.assertFalse(self.manager.decorate(self.snapshot())['sessions'][0]['rescue']['stale'])
        # Another selection forces a current evidence check, not an old package reuse.
        with self.rollout.open('a') as handle:
            handle.write(json.dumps({'type': 'event_msg', 'payload': {'type': 'user_message', 'message': '新增约束：不能修改计算接口'}}) + '\n')
        with self.assertRaisesRegex(ValueError, '新的可读对话'):
            self.manager.prepare({'captureId': capture['captureId'], 'planJobId': job['jobId'], 'taskIds': ['fix']})

    def test_stale_capture_or_unrelated_plan_is_rejected(self):
        capture = self.manager.capture({'threadId': 'source'})
        self.session['lastEventAt'] = 'changed'
        with self.assertRaisesRegex(ValueError, '已变化'):
            self.manager.plan({'captureId': capture['captureId']})
        with self.assertRaisesRegex(ValueError, '不属于'):
            self.manager.review({'captureId': capture['captureId'], 'planJobId': 'a' * 32})
        self.transport.assert_not_called()

    def test_capacity_limits_concurrency_and_unknown_or_overlapping_files_are_serial(self):
        self.session['status']['code'] = 'retry_wait'
        _, _, package = self.prepared()
        self.assertEqual(package['maxParallel'], 1)
        self.assertEqual(package['waves'], [['fix'], ['review'], ['test']])
        tasks = copy.deepcopy(TASKS[:2])
        tasks[1]['fileScope'] = ['calc.py']
        self.assertEqual(execution_waves(tasks, 2), [['fix'], ['review']])
        tasks[1]['fileScope'] = []
        self.assertEqual(execution_waves(tasks, 2), [['fix'], ['review']])

    def test_uncertain_capacity_history_still_keeps_concurrency_low(self):
        self.session['status'] = {'code': 'unknown', 'label': '繁忙后暂无新事件'}
        capture = self.manager.capture({'threadId': 'source'})
        self.assertIn('先前拥堵尚未核实恢复', [item['label'] for item in capture['diagnosis']['signals']])
        job = self.manager.plan({'captureId': capture['captureId']})
        job = self.brain.result(job['jobId'], 3000)
        self.session['status'] = {'code': 'interrupted'}
        package = self.manager.prepare({'captureId': capture['captureId'], 'planJobId': job['jobId'], 'taskIds': ['fix']})
        self.assertEqual(package['maxParallel'], 1)

    def test_dependency_selection_and_invalid_model_plan_are_rejected(self):
        capture, job = self.planned()
        self.session['status']['code'] = 'completed'
        with self.assertRaisesRegex(ValueError, '依赖'):
            self.manager.prepare({'captureId': capture['captureId'], 'planJobId': job['jobId'], 'taskIds': ['test']})
        for field, value in [('fileScope', ['../private']), ('fileScope', ['/private']), ('execution', 'shell'), ('dependsOn', ['test'])]:
            content = copy.deepcopy(PLAN); content['tasks'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_rescue_plan(content)
        content = copy.deepcopy(PLAN); content['tasks'][0]['dependsOn'] = ['test']
        with self.assertRaisesRegex(ValueError, '循环'):
            validate_rescue_plan(content)

    def test_claim_prevents_duplicate_dispatch_and_progress_enforces_dependency(self):
        _, _, package = self.prepared()
        args = {'packageId': package['packageId']}
        claim = self.manager.claim(args)
        self.assertTrue(claim['claimed'])
        self.assertFalse(self.manager.claim(args)['claimed'])
        token = claim['claimToken']
        with self.assertRaisesRegex(ValueError, '领取'):
            self.manager.progress({**args, 'taskId': 'fix', 'status': 'running', 'agentId': 'worker'})
        with self.assertRaisesRegex(ValueError, '依赖'):
            self.manager.progress({**args, 'claimToken': token, 'taskId': 'test', 'status': 'running', 'agentId': 'worker'})
        self.manager.progress({**args, 'claimToken': token, 'taskId': 'fix', 'status': 'running', 'agentId': 'code-worker'})
        self.manager.progress({**args, 'claimToken': token, 'taskId': 'fix', 'status': 'completed', 'agentId': 'code-worker', 'summary': '代码已修改，具体证据写入报告'})
        with self.assertRaisesRegex(ValueError, '未回报终态'):
            self.manager.finalize({**args, 'claimToken': token, 'outcome': 'completed', 'summary': '不能提前宣布完成'})
        self.manager.progress({**args, 'claimToken': token, 'taskId': 'review', 'status': 'completed', 'summary': '已核对约束'})
        self.manager.progress({**args, 'claimToken': token, 'taskId': 'test', 'status': 'running', 'agentId': 'test-worker'})
        self.manager.progress({**args, 'claimToken': token, 'taskId': 'test', 'status': 'completed', 'summary': '验证通过，证据位于本地报告'})
        final = self.manager.finalize({**args, 'claimToken': token, 'outcome': 'completed', 'summary': '已核验代码与验证结果；这是fixture回报。'})
        self.assertEqual(final['phase'], 'integrated')
        self.assertIn('已核验代码', Path(final['reportPath']).read_text())
        self.assertFalse(list(self.workspace.iterdir()))

    def test_prepared_package_revalidates_new_dialogue_and_claimed_selection_cannot_duplicate(self):
        capture, job, package = self.prepared()
        self.assertTrue(self.manager.claim({'packageId': package['packageId']})['claimed'])
        blocked = self.manager.prepare({'captureId': capture['captureId'], 'planJobId': job['jobId'], 'taskIds': ['fix']})
        self.assertTrue(blocked['blocked'])
        # A new manager/source can make a later plan, but the same source lease stops concurrent claiming.
        capture2, job2 = self.planned()
        package2 = self.manager.prepare({'captureId': capture2['captureId'], 'planJobId': job2['jobId'], 'taskIds': ['fix']})
        self.assertFalse(self.manager.claim({'packageId': package2['packageId']})['claimed'])
        self.assertEqual(len(list((self.manager.directory / 'leases').glob('*.json'))), 1)

    def test_resume_keeps_existing_capture_and_plan_across_service_restart(self):
        self.assertFalse(self.manager.resume({'threadId': 'source'})['found'])
        capture, job, package = self.prepared()
        state = self.manager.resume({'threadId': 'source'})
        self.assertEqual(state['capture']['captureId'], capture['captureId'])
        self.assertEqual(state['package']['packageId'], package['packageId'])
        self.assertTrue(state['reviewed'])
        self.assertFalse(state['mark']['stale'])
        self.assertEqual(self.transport.call_count, 1)
        # Completed model jobs are restored from our own bounded result file, no new request.
        self.brain.jobs.clear()
        self.assertEqual(self.manager.resume({'threadId': 'source'})['plan']['goal'], PLAN['goal'])
        self.assertEqual(self.manager.review({'captureId': capture['captureId'], 'planJobId': job['jobId']})['plan']['goal'], PLAN['goal'])
        self.assertEqual(self.transport.call_count, 1)

    def test_old_plan_cannot_override_new_supplement_and_existing_package_rechecks_dialogue(self):
        capture, first = self.planned()
        next_job = self.manager.plan({'captureId': capture['captureId'], 'supplement': '新的明确限制'})
        next_job = self.brain.result(next_job['jobId'], 3000)
        self.assertEqual(next_job['status'], 'completed')
        with self.assertRaisesRegex(ValueError, '更新的规划'):
            self.manager.review({'captureId': capture['captureId'], 'planJobId': first['jobId']})
        self.session['status']['code'] = 'completed'
        args = {'captureId': capture['captureId'], 'planJobId': next_job['jobId'], 'taskIds': ['fix']}
        package = self.manager.prepare(args)
        self.assertTrue(package['ready'])
        with self.rollout.open('a') as handle:
            handle.write(json.dumps({'type': 'event_msg', 'payload': {'type': 'user_message', 'message': '不执行，需求已经变化'}}) + '\n')
        with self.assertRaisesRegex(ValueError, '新的可读对话'):
            self.manager.prepare(args)
        self.assertFalse(self.manager.claim({'packageId': package['packageId']})['claimed'])

    def test_failed_plan_is_marked_and_capture_boundary_excludes_full_materials(self):
        capture = self.manager.capture({'threadId': 'source'})
        self.transport.return_value = ({'summary': '没有可靠任务'}, {})
        job = self.manager.plan({'captureId': capture['captureId']})
        self.assertEqual(self.brain.result(job['jobId'], 3000)['status'], 'failed')
        self.assertEqual(self.manager.decorate(self.snapshot())['sessions'][0]['rescue']['phase'], 'blocked')
        wrapped = server.rescue_result(capture)
        self.assertNotIn('materials', wrapped['structuredContent'])
        self.assertIn('materials', wrapped['_meta']['rescueResult'])


if __name__ == '__main__':
    unittest.main()

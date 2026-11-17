import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from brain import ExternalBrain
from dispatch import SubtaskDispatcher, validate_tasks
import server


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.calls = []
        self.transport = Mock(side_effect=self.respond)
        self.reader = SimpleNamespace(home=self.base / 'home')
        self.brain = ExternalBrain(self.base, self.reader, Mock(), Mock(), self.transport)
        self.brain.save_config({'enabled': True, 'allowedModes': ['dispatch'], 'apiKey': 'dispatch-fixture-key'})
        self.dispatch = SubtaskDispatcher(self.brain, self.base)
        self.args = {'question': '检查给定实现', 'context': '只看这些已授权材料', 'tasks': [
            {'id': str(i), 'title': '任务' + str(i), 'instruction': '独立检查并给出依据'} for i in range(3)]}

    def respond(self, config, key, materials):
        self.calls.append(materials)
        return {'summary': '模拟分析结果', 'findings': ['依据']}, {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}

    def wait(self, batch):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            result = self.dispatch.result(batch['batchId'])
            if result['status'] not in ('queued', 'running', 'stopping'):
                return result
            time.sleep(.02)
        self.fail('batch did not finish')

    def test_default_permission_gate_and_task_validation(self):
        self.brain.save_config({'allowedModes': ['analysis']})
        with self.assertRaisesRegex(ValueError, '启用'):
            self.dispatch.start(self.args)
        self.transport.assert_not_called()
        for tasks in ([], [{}], [{'title': 'name', 'instruction': ''}], [{'id': 'a', 'title': 'x', 'instruction': 'y'}] * 2):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                validate_tasks(tasks)

    def test_manual_distribution_uses_only_selected_tasks_and_existing_brain_limits(self):
        batch = self.wait(self.dispatch.start({**self.args, 'taskIds': ['1', '2']}))
        self.assertEqual(batch['status'], 'completed')
        self.assertEqual([task['id'] for task in batch['tasks']], ['1', '2'])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(batch['usage']['total_tokens'], 24)
        self.assertEqual(batch['budget']['maxOutputTokensTotal'], 4096)
        self.assertIn('待 Codex 核验', batch['report'])
        self.assertNotIn('dispatch-fixture-key', json.dumps(batch))
        self.assertTrue(Path(batch['resultPath']).is_file())
        self.brain.snapshot.assert_not_called()

    def test_ai_plan_does_not_start_children_and_edited_plan_reuses_materials(self):
        self.transport.side_effect = None
        self.transport.return_value = ({'summary': '可分两项', 'tasks': self.args['tasks'][:2]}, {'total_tokens': 5})
        plan = self.brain.start({'mode': 'dispatch', 'question': '目标', 'context': '计划原材料'})
        plan = self.brain.result(plan['jobId'], 3000)
        self.assertEqual(plan['status'], 'completed')
        self.assertEqual(self.transport.call_count, 1)
        self.assertIn('taskPlan', plan)
        self.transport.side_effect = self.respond
        edited = [{**plan['taskPlan'][0], 'instruction': '用户修改后的独立分析要求'}]
        batch = self.wait(self.dispatch.start({'planJobId': plan['jobId'], 'tasks': edited}))
        self.assertEqual(batch['status'], 'completed')
        self.assertIn('用户修改后的独立分析要求', self.calls[0])
        self.assertIn('计划原材料', self.calls[0])
        self.assertEqual(self.transport.call_count, 2)

    def test_invalid_plan_is_failed_instead_of_falsely_ready(self):
        plan = self.brain.start({'mode': 'dispatch', 'question': '拆分'})
        plan = self.brain.result(plan['jobId'], 3000)
        self.assertEqual(plan['status'], 'failed')
        self.assertNotIn('taskPlan', plan)

    def test_concurrency_dedup_and_stop_only_unsent_tasks(self):
        gate = threading.Event()
        entered = threading.Event()
        lock = threading.Lock()
        active = [0, 0]
        def blocked(config, key, materials):
            with lock:
                self.calls.append(materials)
                active[0] += 1
                active[1] = max(active)
                if active[0] == 2:
                    entered.set()
            gate.wait(3)
            with lock:
                active[0] -= 1
            return {'summary': '完成'}, {'total_tokens': 1}
        self.transport.side_effect = blocked
        args = {**self.args, 'tasks': self.args['tasks'] + [{'id': '4', 'title': '第四项', 'instruction': '检查'}]}
        first = self.dispatch.start(args)
        self.assertTrue(entered.wait(2))
        second = self.dispatch.start(args)
        self.assertEqual(first['batchId'], second['batchId'])
        self.assertEqual(len(self.calls), 2)
        self.dispatch.stop(first['batchId'])
        gate.set()
        result = self.wait(first)
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['counts']['completed'], 2)
        self.assertEqual(result['counts']['skipped'], 2)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(active[1], 2)

    def test_partial_failure_is_preserved_without_retry(self):
        def respond(config, key, materials):
            self.calls.append(materials)
            if '任务1' in json.loads(materials)['question']:
                raise ValueError('模拟服务失败')
            return {'summary': '成功项保留'}, {'total_tokens': 1}
        self.transport.side_effect = respond
        batch = self.wait(self.dispatch.start(self.args))
        self.assertEqual(batch['status'], 'partial_failed')
        self.assertEqual(batch['counts']['failed'], 1)
        self.assertEqual(batch['counts']['completed'], 2)
        self.assertEqual(len(self.calls), 3)
        self.assertIn('成功项保留', batch['report'])

    def test_config_change_stops_later_submissions(self):
        def changed(config, key, materials):
            self.calls.append(materials)
            self.brain.save_config({'model': 'another-model'})
            return {'summary': '原设置的结果'}, {}
        self.transport.side_effect = changed
        batch = self.wait(self.dispatch.start({**self.args, 'parallelism': 1}))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(batch['counts']['failed'], 2)
        self.assertIn('设置已变化', batch['report'])

    def test_completed_record_can_be_reopened_and_interrupted_never_restarts(self):
        batch = self.wait(self.dispatch.start(self.args))
        restarted = SubtaskDispatcher(self.brain, self.base)
        self.assertEqual(restarted.result(batch['batchId'])['status'], 'completed')
        path = Path(batch['resultPath'])
        batch.update({'status': 'running', 'ownerPid': os.getpid()})
        path.write_text(json.dumps(batch))
        self.assertEqual(restarted.result(batch['batchId'])['status'], 'interrupted')
        self.assertEqual(self.transport.call_count, 3)
        with self.assertRaises(ValueError):
            restarted.result('../../private')

    def test_other_live_process_is_read_without_marking_interrupted_and_stop_is_signaled(self):
        batch = self.wait(self.dispatch.start(self.args))
        path = Path(batch['resultPath'])
        batch.update({'status': 'running', 'ownerPid': os.getpid() + 100000})
        path.write_text(json.dumps(batch))
        other = SubtaskDispatcher(self.brain, self.base)
        with patch('dispatch.os.kill'):
            self.assertEqual(other.result(batch['batchId'])['status'], 'running')
            self.assertEqual(other.stop(batch['batchId'])['status'], 'stopping')
        self.assertTrue((path.parent / (batch['batchId'] + '.stop.json')).is_file())
        self.assertEqual(self.transport.call_count, 3)

    def test_model_visible_results_remain_bounded(self):
        plan = {'jobId': 'a' * 32, 'taskPlan': self.args['tasks'], 'dispatchMaterials': 'RAW_MATERIALS', 'answer': 'FULL_ANSWER', 'answerExcerpt': 'brief'}
        wrapped = server.brain_result(plan)
        visible = json.dumps([wrapped['content'], wrapped['structuredContent']])
        self.assertNotIn('RAW_MATERIALS', visible)
        self.assertNotIn('独立检查并给出依据', visible)
        batch = self.wait(self.dispatch.start(self.args))
        wrapped = server.dispatch_result(batch)
        self.assertNotIn('report', wrapped['structuredContent'])
        self.assertNotIn('instruction', wrapped['structuredContent']['tasks'][0])
        self.assertEqual(wrapped['_meta']['dispatchResult']['report'], batch['report'])


if __name__ == '__main__':
    unittest.main()

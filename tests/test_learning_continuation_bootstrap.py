"""Fresh finite preparation never launches native work or reuses LIVE state."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location('continuation_bootstrap', EXAMPLES / 'bootstrap_learning_continuation.py')
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class ContinuationBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'private/continuations/new-experiment'
        self.configs, self.data = {}, {}
        for name, (_, _, steps, seconds) in HELPER.CONFIGS.items():
            filename = name + '.json'
            value = {'objective': 'causal_lm' if name == 'base' else 'assistant_last_reply',
                     'partition_policy': 'train_validation_only', 'data_loading': 'mmap',
                     'trainable_parameters': 'all', 'max_seconds': seconds, 'max_steps': steps,
                     'torch_threads': 1}
            path = self.write(HELPER.PREFIX + 'configs/' + filename, value)
            self.configs[name] = (filename, HELPER.supervisor.sha(path), steps, seconds)
        for name in HELPER.DATA:
            causal = name in ('wide1', 'wide2', 'tatoeba')
            rows = [{'id': name + '-' + split, 'groupId': name + '-' + split, 'split': split,
                     'source': {'dataset': 'explicit-bootstrap-test-fixture', 'kind': 'authored-synthetic'},
                     **({'text': 'Texto nuevo ' + name + ' ' + split} if causal else {
                         'messages': [{'role': 'user', 'content': name + ' pregunta ' + split},
                                      {'role': 'assistant', 'content': name + ' respuesta ' + split}]})}
                    for split in ('train', 'validation')]
            relative = name + '/corpus.jsonl'
            corpus = self.root / (HELPER.PREFIX + relative)
            corpus.parent.mkdir(parents=True)
            corpus.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            provenance_name = name + '/manifest.json'
            provenance = self.write(HELPER.PREFIX + provenance_name, {
                'corpusSha256': HELPER.supervisor.sha(corpus), 'splitCounts': {'train': 1, 'validation': 1, 'test': 0},
                'sources': [{'dataset': 'explicit-bootstrap-test-fixture'}], 'testUsedForTraining': False})
            self.data[name] = (relative, HELPER.supervisor.sha(corpus), provenance_name, HELPER.supervisor.sha(provenance))
        parent_manifest = {'modelId': HELPER.PARENT_ID, 'config': {'context_length': 2048},
                           'training': {'trainedFrom': 'own_checkpoint', 'externalWeightsUsed': False,
                                        'optimizerSteps': 0, 'trainingPerformed': False}}
        self.parent_files = {}
        for filename in HELPER.PARENT_FILES:
            value = parent_manifest if filename == 'manifest.json' else {'fixture': filename}
            self.parent_files[filename] = HELPER.supervisor.sha(self.write(HELPER.PARENT + '/' + filename, value))
        for name in ('examples/continual_training.py', 'examples/run_learning_stage.py', 'examples/run_conversation_pilot.py'):
            self.write(name, {'not_executable_fixture': name})
        template = self.write(HELPER.TEMPLATE, {'postStage': {
            'enabled': True, 'purpose': 'development_generation', 'torchThreads': 1,
            'timeoutSeconds': 300, 'argv': [sys.executable, 'examples/run_learning_evaluation.py'],
            'sourceSha256': {}}})
        for name, value in [('CONFIGS', self.configs), ('DATA', self.data), ('PARENT_FILES', self.parent_files),
                            ('TEMPLATE_SHA', HELPER.supervisor.sha(template))]:
            manager = patch.object(HELPER, name, value)
            manager.start()
            self.addCleanup(manager.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(HELPER.encoded(value))
        return path

    def test_dry_run_checks_actual_corpora_and_graph_without_outputs_or_processes(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with patch.object(HELPER.supervisor.subprocess, 'Popen') as process:
            receipt = HELPER.prepare(self.root, self.output, dry_run=True)
            process.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / 'private').exists())
        self.assertTrue(receipt['dryRun'])
        self.assertFalse(receipt['trainingStarted'])
        self.assertFalse(receipt['collectorEnabled'])
        self.assertEqual(before, {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob('*') if path.is_file()})

    def test_fresh_state_has_no_old_pid_receipts_or_budgets_and_exact_approvals(self):
        with patch.object(HELPER.supervisor.subprocess, 'Popen') as process:
            HELPER.prepare(self.root, self.output)
            process.assert_not_called()
        config = HELPER.supervisor.read(self.output / 'config.json')
        state = HELPER.supervisor.read(self.output / 'state/state.json')
        receipt = HELPER.supervisor.read(self.output / 'preparation-receipt.json')
        self.assertIs(receipt['dryRun'], False)
        self.assertEqual(state['configSha256'], HELPER.supervisor.sha(self.output / 'config.json'))
        self.assertEqual((state['cyclesCompleted'], state['stageIndex'], state['stageResults']), (0, 0, {}))
        self.assertEqual(config['maxCycles'], 3)
        self.assertEqual(config['postStage']['torchThreads'], 1)
        self.assertFalse(config['collector']['enabled'])
        self.assertNotIn('adoptPid', json.dumps(config))
        self.assertNotIn('private/learning-repair-20261013', json.dumps(config))
        self.assertEqual(config['initialCycle']['stages'][0]['parentFiles'], self.parent_files)
        for approval in (self.output / 'inbox').glob('*.approval.json'):
            value = HELPER.supervisor.read(approval)
            manifest = approval.with_name(approval.name.replace('.approval.json', '.json'))
            self.assertIs(value['approved'], True)
            self.assertEqual(value['manifestSha256'], HELPER.supervisor.sha(manifest))

    def test_malformed_later_corpus_fails_before_any_output(self):
        path = self.root / (HELPER.PREFIX + self.data['tatoeba'][0])
        path.write_text('{"not_a_training_record":true}\n')
        altered = deepcopy(self.data)
        altered['tatoeba'] = (altered['tatoeba'][0], HELPER.supervisor.sha(path), *altered['tatoeba'][2:])
        provenance = self.root / (HELPER.PREFIX + altered['tatoeba'][2])
        value = HELPER.supervisor.read(provenance)
        value['corpusSha256'] = altered['tatoeba'][1]
        provenance.write_bytes(HELPER.encoded(value))
        altered['tatoeba'] = (*altered['tatoeba'][:3], HELPER.supervisor.sha(provenance))
        with patch.object(HELPER, 'DATA', altered), self.assertRaises(ValueError):
            HELPER.prepare(self.root, self.output)
        self.assertFalse((self.root / 'private').exists())

    def test_existing_output_symlink_or_live_destination_is_never_overwritten(self):
        self.output.mkdir(parents=True)
        marker = self.output / 'original'
        marker.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            HELPER.prepare(self.root, self.output)
        with self.assertRaises(ValueError):
            HELPER.new_output(self.root, 'private/learning-repair-20261013/continual')
        alias = self.root / 'private/continuations/alias'
        alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(ValueError):
            HELPER.new_output(self.root, alias)
        self.assertEqual(marker.read_bytes(), b'keep')

    def test_atomic_publication_refuses_destination_created_after_validation(self):
        self.output.mkdir(parents=True)
        (self.output / 'keep').write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            HELPER.publish(self.output, {'config.json': b'{}'})
        self.assertEqual({path.name for path in self.output.iterdir()}, {'keep'})
        self.assertFalse(list(self.output.parent.glob('.new-experiment.bootstrap-*')))


if __name__ == '__main__':
    unittest.main()

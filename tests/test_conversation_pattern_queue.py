"""Native pure preflight and exact queue authorization preserve active state."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location('pattern_queue', EXAMPLES / 'enqueue_conversation_pattern_stage.py')
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class PatternQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {'schemaVersion': 1, 'workspaceRoot': str(self.root),
                       'stateDir': 'live/state', 'inboxDir': 'live/inbox',
                       'approvedAgents': [HELPER.AGENT],
                       'initialCycle': {'cycleId': 'ongoing', 'stages': []}}
        self.config_path = self.write('supervisor.json', self.config)
        self.state = self.write('live/state/state.json', {'worker': {'pid': 123, 'step': 99}})
        (self.root / 'live/inbox').mkdir(parents=True)
        self.parent_config = {'context_length': 64, 'd_model': 8, 'dropout': 0.0,
                              'ffn_multiplier': 4, 'n_heads': 1, 'n_layers': 1,
                              'seed': 42, 'vocab_size': 262}
        for filename in HELPER.supervisor.FILES:
            value = {'fixture': filename}
            if filename == 'config.json':
                value = self.parent_config
            elif filename == 'manifest.json':
                value = {'modelId': 'declared-own-fixture', 'config': self.parent_config,
                         'training': {'trainedFrom': 'own_checkpoint',
                                      'lineage': {'rootOrigin': 'random_weights'}}}
            self.write('own-parent/' + filename, value)
        rows = [{'id': 'sample-' + split, 'groupId': 'family-' + split, 'split': split,
                 'source': {'dataset': 'declared-test-fixture', 'kind': 'authored-synthetic'},
                 'messages': [{'role': 'user', 'content': '¿Cuál es el color ' + split + '?'},
                              {'role': 'assistant', 'content': 'Azul ' + split + '.'}]}
                for split in ('train', 'validation')]
        self.corpus = self.root / 'corpus.jsonl'
        self.corpus.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        content = ''.join(json.dumps({'groupId': row['groupId'], 'messages': row['messages']},
                                    ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')) + '\n' for row in rows)
        self.provenance = {'corpusSha256': HELPER.supervisor.sha(self.corpus),
                           'contentSha256': HELPER.digest_bytes(content.encode()),
                           'contentSha256Method': HELPER.CONTENT_METHOD,
                           'splitCounts': {'train': 1, 'validation': 1, 'test': 0},
                           'testUsedForTraining': False, 'sources': [{'dataset': 'declared-test-fixture'}]}
        self.manifest = self.write('manifest.json', self.provenance)
        for source in HELPER.SOURCE_FILES:
            self.write(source, {'not_executed_fixture': source})
        self.write('src/ia_propia/conversation/fixture.py', {'not_executed_fixture': True})
        self.output = self.root / 'prepared/new-pattern-stage'
        self.arguments = dict(parent='own-parent', corpus='corpus.jsonl', manifest='manifest.json',
                              run_dir='runs/new-pattern-stage', output_dir=str(self.output),
                              cycle_id='conversation-patterns-v1')

    def write(self, filename, value):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(HELPER.encoded(value))
        return path

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob('*') if path.is_file()}

    def test_dry_run_does_real_native_preflight_without_state_changes_or_processes(self):
        before = self.snapshot()
        with patch.object(HELPER.supervisor.subprocess, 'Popen') as process:
            receipt = HELPER.prepare(self.config_path, **self.arguments, dry_run=True)
            process.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.output.exists())
        self.assertFalse(receipt['trainingStarted'])
        self.assertFalse(receipt['optimizerStateCopied'])
        self.assertEqual(receipt['parentContextLength'], 64)
        self.assertEqual(receipt['corpusAudit']['splitCounts'], self.provenance['splitCounts'])

    def test_preparation_and_queue_pin_exact_agent_and_preserve_live_state(self):
        original_state, original_config = self.state.read_bytes(), self.config_path.read_bytes()
        with patch.object(HELPER.supervisor.subprocess, 'Popen') as process:
            HELPER.prepare(self.config_path, **self.arguments)
            queued = HELPER.enqueue(self.config_path, str(self.output))
            process.assert_not_called()
        self.assertTrue(queued['queued'])
        self.assertFalse(queued['admitted'])
        self.assertFalse(queued['trainingStarted'])
        approval = HELPER.supervisor.read(Path(queued['approvalPath']))
        self.assertEqual(approval['approvedBy'], HELPER.AGENT)
        self.assertIs(approval['approved'], True)
        self.assertEqual(approval['manifestSha256'], queued['manifestSha256'])
        stage = HELPER.supervisor.read(Path(queued['manifestPath']))['stages'][0]
        self.assertEqual(stage['kind'], 'pattern_diagnostic')
        self.assertEqual((stage['maxRSSMiB'], stage['maxDiskMiB'], stage['stopAfterStep']), (6144, 1024, 512))
        self.assertEqual(self.state.read_bytes(), original_state)
        self.assertEqual(self.config_path.read_bytes(), original_config)
        self.assertFalse((self.root / 'runs').exists())

    def test_untrusted_agent_and_bad_provenance_fail_before_publication(self):
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            HELPER.prepare(self.config_path, **self.arguments, approved_agent=HELPER.AGENT + '-spoof')
        self.assertEqual(self.snapshot(), before)
        for bad in ({'testUsedForTraining': True}, {'contentSha256': '0' * 64},
                    {'splitCounts': {'train': 2, 'validation': 1, 'test': 0}}, {'sources': []},
                    {'contentSha256Method': 'different undeclared projection'}):
            with self.subTest(bad=bad):
                self.write('manifest.json', {**self.provenance, **bad})
                with self.assertRaises(ValueError):
                    HELPER.prepare(self.config_path, **self.arguments)
                self.assertFalse(self.output.exists())

    def test_queue_rechecks_parent_inputs_and_sources_without_touching_state(self):
        for filename in ('own-parent/weights.safetensors', 'corpus.jsonl',
                         'examples/run_learning_stage.py'):
            with self.subTest(filename=filename):
                self.output = self.root / ('prepared/' + filename.replace('/', '-'))
                args = dict(self.arguments, output_dir=str(self.output))
                HELPER.prepare(self.config_path, **args)
                target = self.root / filename
                original = target.read_bytes()
                target.write_bytes(original + b' ')
                before = self.state.read_bytes()
                with self.assertRaises(ValueError):
                    HELPER.enqueue(self.config_path, str(self.output))
                self.assertEqual(self.state.read_bytes(), before)
                self.assertEqual(list((self.root / 'live/inbox').iterdir()), [])
                target.write_bytes(original)

    def test_concurrent_queue_publication_never_overwrites_files(self):
        HELPER.prepare(self.config_path, **self.arguments)
        def publish():
            try:
                return HELPER.enqueue(self.config_path, str(self.output))
            except FileExistsError:
                return 'collision'
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda unused: publish(), range(2)))
        self.assertEqual(sum(isinstance(item, dict) for item in outcomes), 1)
        self.assertEqual(sum(item == 'collision' for item in outcomes), 1)
        files = self.snapshot()
        with self.assertRaises(FileExistsError):
            HELPER.enqueue(self.config_path, str(self.output))
        self.assertEqual(self.snapshot(), files)
        self.assertEqual(len(list((self.root / 'live/inbox').iterdir())), 2)

    def test_new_output_run_and_protected_paths_cannot_be_reused_or_aliased(self):
        self.output.mkdir(parents=True)
        marker = self.output / 'original'
        marker.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            HELPER.prepare(self.config_path, **self.arguments)
        self.assertEqual(marker.read_bytes(), b'preserve')
        for field, value in [('output_dir', 'live/state/new'), ('run_dir', 'live/inbox/new'),
                             ('output_dir', 'own-parent/new')]:
            with self.subTest(field=field):
                args = {**self.arguments, 'output_dir': 'prepared/second', field: value}
                with self.assertRaises(ValueError):
                    HELPER.prepare(self.config_path, **args)
        (self.root / 'alias').symlink_to(self.root / 'own-parent', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            HELPER.prepare(self.config_path, **dict(self.arguments, output_dir='prepared/second', parent='alias'))


if __name__ == '__main__':
    unittest.main()

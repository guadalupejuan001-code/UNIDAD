"""Admission validates full cycle inputs before changing durable progress."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HELPER_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'continual_training.py'
SPEC = importlib.util.spec_from_file_location('continual_training', HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class ContinualAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {'schemaVersion': 1, 'workspaceRoot': str(self.root), 'stateDir': 'state',
                       'inboxDir': 'inbox', 'approvedAgents': ['explicit-test-agent'],
                       'initialCycle': {'cycleId': 'initial', 'stages': []}}
        config_path = self.write('supervisor.json', self.config)
        self.supervisor = HELPER.Supervisor(self.config, config_path, HELPER.sha(config_path))
        self.model = self.write('model.json', {'vocab_size': 262, 'd_model': 8, 'n_heads': 1,
                                             'n_layers': 1, 'context_length': 64})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')
        return path

    def stage(self, stage_id, *, objective='causal_lm', parent=None):
        rows = [{'id': stage_id + '-' + split, 'groupId': stage_id + '-' + split,
                 'split': split, 'source': {'dataset': 'declared-test-fixture', 'kind': 'authored-synthetic'},
                 **({'text': stage_id + ' texto de ' + split} if objective == 'causal_lm' else
                    {'messages': [{'role': 'user', 'content': stage_id + ' pregunta de ' + split},
                                  {'role': 'assistant', 'content': stage_id + ' respuesta de ' + split}]})}
                for split in ('train', 'validation')]
        corpus = self.root / (stage_id + '.jsonl')
        corpus.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf8')
        training = self.write(stage_id + '-config.json', {'objective': objective,
            'partition_policy': 'train_validation_only', 'data_loading': 'mmap', 'trainable_parameters': 'all',
            'max_seconds': 60, 'max_steps': 10, 'torch_threads': 1})
        provenance = self.write(stage_id + '-provenance.json', {
            'corpusSha256': HELPER.sha(corpus), 'contentSha256': HELPER.sha(corpus),
            'splitCounts': {'train': 1, 'validation': 1, 'test': 0}, 'testUsedForTraining': False,
            'sources': [{'dataset': 'declared-test-fixture'}]})
        stage = {'id': stage_id, 'runDir': 'runs/' + stage_id, 'corpusPath': corpus.name,
                 'corpusSha256': HELPER.sha(corpus), 'trainingConfigPath': training.name,
                 'trainingConfigSha256': HELPER.sha(training), 'stopAfterStep': 10,
                 'provenanceManifestPath': provenance.name, 'provenanceManifestSha256': HELPER.sha(provenance)}
        if parent is None:
            stage.update(modelConfigPath=self.model.name, modelConfigSha256=HELPER.sha(self.model))
        else:
            stage['parentStage'] = parent
        return stage

    def inbox(self, name, stages, *, approval=None):
        manifest = self.write('inbox/' + name + '.json', {'schemaVersion': 1, 'cycleId': name, 'stages': stages})
        approved = {'approved': True, 'approvedBy': 'explicit-test-agent', 'manifestSha256': HELPER.sha(manifest)}
        if approval is not None:
            approved = approval
        self.write('inbox/' + name + '.approval.json', approved)
        return manifest

    def alter_provenance(self, stage, **updates):
        path = self.root / stage['provenanceManifestPath']
        value = HELPER.read(path)
        value.update(updates)
        self.write(path.name, value)
        stage['provenanceManifestSha256'] = HELPER.sha(path)

    def events(self):
        return [json.loads(line) for line in (self.root / 'state/journal.jsonl').read_text().splitlines()]

    def test_invalid_approval_is_recorded_once_and_does_not_block_later_approved_cycle(self):
        first = self.stage('bad-approval')
        bad = self.inbox('a-invalid', [first], approval={'approvedBy': 'explicit-test-agent'})
        valid = self.inbox('b-valid', [self.stage('good')])
        with patch.object(HELPER.subprocess, 'Popen') as launched:
            self.assertTrue(self.supervisor.next_cycle())
            self.assertFalse(self.supervisor.next_cycle())
            launched.assert_not_called()
        self.assertEqual(self.supervisor.state['cycle']['cycleId'], 'b-valid')
        self.assertNotIn(HELPER.sha(bad), self.supervisor.state['seenCycleHashes'])
        self.assertEqual(self.supervisor.state['seenCycleHashes'], [HELPER.sha(valid)])
        rejected = self.supervisor.state['rejectedCycleHashes'][HELPER.sha(bad)]
        self.assertIn('approved boolean', rejected['reason'])
        self.assertEqual(sum(row['event'] == 'cycle_rejected_preflight' for row in self.events()), 1)

    def test_invalid_second_stage_provenance_never_admits_or_changes_progress(self):
        first, second = self.stage('first'), self.stage('second', objective='assistant_last_reply', parent='first')
        self.alter_provenance(second, sources=[], testUsedForTraining=True)
        manifest = self.inbox('later-invalid', [first, second])
        previous = deepcopy(self.supervisor.state)
        with patch.object(HELPER.subprocess, 'Popen') as launched:
            self.assertFalse(self.supervisor.next_cycle())
            self.assertFalse(self.supervisor.next_cycle())
            launched.assert_not_called()
        for key in ('cycle', 'stageIndex', 'seenCycleHashes', 'stageResults', 'cyclesCompleted'):
            self.assertEqual(self.supervisor.state[key], previous[key])
        self.assertIn('Stage second preflight failed', self.supervisor.state['rejectedCycleHashes'][HELPER.sha(manifest)]['reason'])
        persisted = HELPER.read(self.supervisor.state_path)
        self.assertIn(HELPER.sha(manifest), persisted['rejectedCycleHashes'])

    def test_prior_cycle_stage_can_be_a_future_parent_but_launch_still_requires_receipt(self):
        first = self.stage('base')
        second = self.stage('sft', objective='assistant_last_reply', parent='base')
        self.inbox('future-parent', [first, second])
        self.assertTrue(self.supervisor.next_cycle())
        self.assertEqual(self.supervisor.state['cycle']['cycleId'], 'future-parent')
        self.assertEqual(self.supervisor.state['stageResults'], {})
        with self.assertRaisesRegex(FileNotFoundError, 'Parent stage receipt not available'):
            self.supervisor.validate_stage(second)

    def test_later_config_and_actual_corpus_errors_reject_before_admission(self):
        for mutation in ('bad-config', 'bad-corpus', 'wrong-splits'):
            with self.subTest(mutation=mutation):
                stage = self.stage(mutation)
                if mutation == 'bad-config':
                    config = self.root / stage['trainingConfigPath']
                    value = HELPER.read(config)
                    value['learning_rate'] = -1
                    self.write(config.name, value)
                    stage['trainingConfigSha256'] = HELPER.sha(config)
                elif mutation == 'bad-corpus':
                    corpus = self.root / stage['corpusPath']
                    corpus.write_text('{"invalid":true}\n', encoding='utf8')
                    stage['corpusSha256'] = HELPER.sha(corpus)
                    self.alter_provenance(stage, corpusSha256=stage['corpusSha256'])
                else:
                    self.alter_provenance(stage, splitCounts={'train': 2, 'validation': 1, 'test': 0})
                self.inbox(mutation, [self.stage(mutation + '-preceding'), stage])
                self.assertFalse(self.supervisor.next_cycle())
                self.assertEqual(self.supervisor.state['seenCycleHashes'], [])
                self.assertEqual(self.supervisor.state['cycle']['cycleId'], 'initial')

    def test_corrected_approval_reconsiders_manifest_without_erasing_previous_rejection(self):
        manifest = self.inbox('corrected', [self.stage('corrected-stage')], approval={})
        digest = HELPER.sha(manifest)
        self.assertFalse(self.supervisor.next_cycle())
        self.write('inbox/corrected.approval.json', {'approved': True, 'approvedBy': 'explicit-test-agent',
                                                   'manifestSha256': digest})
        self.assertTrue(self.supervisor.next_cycle())
        self.assertIn(digest, self.supervisor.state['rejectedCycleHashes'])
        self.assertIn(digest, self.supervisor.state['seenCycleHashes'])


if __name__ == '__main__':
    unittest.main()

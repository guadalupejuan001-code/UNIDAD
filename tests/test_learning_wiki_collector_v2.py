"""Finite collector guards using declared fixtures and frozen anchor evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('collect_learning_wiki_v2', ROOT / 'examples/collect_learning_wiki_v2.py')
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def record(identifier, group, split, text, *, novel=False):
    return {'id': identifier, 'groupId': group, 'split': split, 'text': text,
            'source': {'kind': 'declared-test-fixture',
                       'dataset': COLLECTOR.PIN['dataset'] if novel else 'OpenAssistant/declared-fixture'}}


class CollectorV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.anchors = [record('fixed-train', 'fixed-t', 'train', 'Ingredientes:\n\nContenido fijo de entrenamiento.'),
                        record('fixed-validation', 'fixed-v', 'validation', 'Ingredientes:\n\nContenido fijo de validación.')]

    def test_historical_baseline_does_not_exclude_unchanged_anchors_or_clean_novel_rows(self):
        novel = [record('new-t', 'new-t', 'train', 'Texto nuevo único de entrenamiento.', novel=True),
                 record('new-v', 'new-v', 'validation', 'Texto nuevo único de validación.', novel=True)]
        before = deepcopy(self.anchors)
        kept, receipt = COLLECTOR.filter_families(self.anchors, novel)
        self.assertEqual(kept, before + novel)
        self.assertEqual(self.anchors, before)
        self.assertEqual(receipt['historicalBaselineClassCount'], 1)
        self.assertEqual(receipt['foldsMoved'], 0)
        self.assertEqual(receipt['quarantinedNovelFamilies'], [])

    def test_novel_anchor_collision_discards_only_entire_novel_family(self):
        novel = [record('part1', 'novel', 'validation', 'Contenido fijo de entrenamiento.\n\nOtra sección nueva.', novel=True),
                 record('part2', 'novel', 'validation', 'Segundo chunk nuevo sin duplicación.', novel=True)]
        kept, receipt = COLLECTOR.filter_families(self.anchors, novel)
        self.assertEqual(kept, self.anchors)
        self.assertEqual([r['groupId'] for r in receipt['quarantinedNovelFamilies']], ['novel'])

    def test_same_fold_partial_overlap_survives_but_exact_duplicate_does_not(self):
        partial = record('partial', 'partial', 'train', 'Contenido fijo de entrenamiento.\n\nSección nueva propia.', novel=True)
        exact = record('exact', 'exact', 'train', self.anchors[0]['text'], novel=True)
        kept, receipt = COLLECTOR.filter_families(self.anchors, [partial, exact])
        self.assertEqual(kept, self.anchors + [partial])
        self.assertEqual(receipt['quarantinedNovelFamilies'][0]['groupId'], 'exact')

    def test_novel_crossfold_collision_discards_both_families(self):
        shared = 'Párrafo idéntico compartido por candidatos nuevos.'
        novel = [record('new-t', 'new-t', 'train', shared + '\n\nFinal único uno.', novel=True),
                 record('new-v', 'new-v', 'validation', shared + '\n\nFinal único dos.', novel=True)]
        kept, receipt = COLLECTOR.filter_families(self.anchors, novel)
        self.assertEqual(kept, self.anchors)
        self.assertEqual({r['groupId'] for r in receipt['quarantinedNovelFamilies']}, {'new-t', 'new-v'})

    def test_real_frozen_rehearsal_five_legacy_classes_are_preserved(self):
        source = ROOT / 'examples/learning-repair-20261013/collector/oasst-rehearsal.train-validation.jsonl'
        digest = COLLECTOR.public.file_sha(source)
        anchors = [json.loads(raw) for raw in source.read_bytes().splitlines()]
        kept, receipt = COLLECTOR.filter_families(anchors, [])
        self.assertEqual(kept, anchors)
        self.assertEqual(receipt['historicalBaselineClassCount'], 5)
        self.assertEqual(receipt['fixedRowsPreserved'], 2299)
        self.assertEqual(COLLECTOR.public.file_sha(source), digest)
        self.assertEqual(digest, '9d162aa67ab93bdd22229ad2a599cab3569ae723e139876143365681df291ac7')

    def test_cached_raw_receipt_reuses_exact_bytes_offline_and_does_not_decode_test_text(self):
        ids = {split: next(i for i in range(1000) if COLLECTOR.public.assign_split('wikipedia:es:' + str(i)) == split)
               for split in ('train', 'test')}
        value = {'num_rows_total': COLLECTOR.public.WIKI_ROWS, 'partial': False,
                 'rows': [{'row': {'id': identifier, 'text': 'Texto público de fixture ' + split}}
                          for split, identifier in ids.items()]}
        raw = json.dumps(value).encode()
        cache = self.root / 'cache'
        response = (3700, raw, {'xRevision': COLLECTOR.PIN['revision']}, value, 'https://fixture.invalid')
        with patch.object(COLLECTOR, 'fetch_page', return_value=response) as fetched:
            COLLECTOR.cached_page(cache, 0, 100000, time.monotonic() + 30)
            fetched.assert_called_once()
        with patch.object(COLLECTOR, 'fetch_page') as fetched:
            restored = COLLECTOR.cached_page(cache, 0, 100000, time.monotonic() + 30, offline=True)
            fetched.assert_not_called()
        self.assertEqual(restored[1], raw)
        self.assertEqual(restored[3]['rows'][0]['row']['text'], 'Texto público de fixture train')
        self.assertEqual(restored[3]['rows'][1]['row']['text'], '')
        raw_path = cache / 'page-000000.json'
        raw_path.write_bytes(raw + b' ')
        with self.assertRaisesRegex(ValueError, 'cache or receipt changed'):
            COLLECTOR.cached_page(cache, 0, 100000, time.monotonic() + 30, offline=True)

    def test_failed_pending_batch_and_disk_guard_never_advance_cursor_or_publish(self):
        anchors = self.root / 'anchors.jsonl'
        anchors.write_bytes(b''.join(COLLECTOR.encoded(r) for r in self.anchors))
        state = {'seenArticleIds': [], 'seenArticleTextSha256': [], 'seenSelectedProseSha256': [],
                 'seenRecordTextSha256': {}, 'seenNormalizedParagraphSha256': {}, 'nextPageOrder': 593}
        original = deepcopy(state)
        staging = self.root / 'staging'
        staging.mkdir()
        limits = {'budgetSeconds': 60, 'corpusCapMiB': 4, 'maxDiskMiB': 64, 'maxPages': 1,
                  'maxNetworkMiB': 64, 'targetNovelPayloadMiB': 1}
        arguments = ({}, limits, state, self.root / 'state.json', staging, self.root / 'final', self.root / 'inbox',
                     'parent:sft', 'wiki-cycle-0001', 1, anchors, self.root / 'sft', self.root / 'manifest', {})
        with patch.object(COLLECTOR, 'cached_page', side_effect=TimeoutError('finite failed request')):
            with self.assertRaises(TimeoutError):
                COLLECTOR.prepare_cycle(*arguments, prepare_only=True)
        self.assertEqual(state, original)
        self.assertFalse((self.root / 'state.json').exists())
        self.assertFalse((self.root / 'inbox').exists())
        with patch.object(COLLECTOR.causal, 'directory_bytes', return_value=65 * COLLECTOR.MIB), patch.object(COLLECTOR, 'cached_page') as fetched:
            with self.assertRaisesRegex(ValueError, 'disk cap'):
                COLLECTOR.prepare_cycle(*arguments, prepare_only=True)
            fetched.assert_not_called()
        self.assertEqual(state, original)


if __name__ == '__main__':
    unittest.main()

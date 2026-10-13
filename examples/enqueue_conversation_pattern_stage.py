"""Prepare one bounded own-model pattern-learning stage for an existing queue.

Default preparation performs native read-only preflight and creates a NEW
bundle. --enqueue publishes an exact standing-agent approval to the configured
inbox. Neither operation starts workers, edits supervisor state/configuration,
or treats training preparation as learned conversation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import bootstrap_learning_continuation as publication
import continual_training as supervisor

AGENT = 'project-agent-with-owner-standing-authorization'
SOURCE_FILES = ('examples/continual_training.py', 'examples/run_learning_stage.py',
                'examples/run_conversation_pilot.py',
                'examples/enqueue_conversation_pattern_stage.py')
CONTENT_METHOD = ('SHA256 over ordered UTF-8 JSONL of {groupId,messages}, sorted keys, '
                  'compact JSON, ensure_ascii=False, one trailing LF per record; '
                  'excludes IDs, splits and source metadata')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def checked_path(config, value, *, new=False):
    """Reject lexical escapes and symlink aliases before path resolution."""
    root = Path(config['workspaceRoot'])
    raw = Path(value)
    raw = raw if raw.is_absolute() else root / raw
    try:
        relative = raw.relative_to(root)
    except ValueError as error:
        raise ValueError('Path must remain inside the supervisor workspace') from error
    if not relative.parts or '..' in relative.parts or '.git' in relative.parts:
        raise ValueError('Path cannot be the workspace, traverse parents or enter .git')
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Configured inputs and destinations cannot contain symlinks')
    result = supervisor.path_for(config, str(raw))
    if new and os.path.lexists(result):
        raise FileExistsError('Choose a NEW destination: ' + str(result))
    return result


def overlaps(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def audit_corpus(corpus, manifest):
    from ia_propia.conversation.data import iter_corpus
    provenance = supervisor.read(manifest)
    if type(provenance) is not dict:
        raise ValueError('Corpus provenance must be an object')
    if provenance.get('contentSha256Method') != CONTENT_METHOD:
        raise ValueError('Explicit supported contentSha256Method is required')
    counts, content = Counter(), hashlib.sha256()
    families = {}
    for record in iter_corpus(corpus):
        if record['split'] == 'test':
            raise ValueError('Pattern-learning corpus must contain no TEST records')
        counts[record['split']] += 1
        families[record['groupId']] = record['split']
        content.update((json.dumps({'groupId': record['groupId'], 'messages': record['messages']},
                                   ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False) + '\n').encode('utf8'))
    actual = {name: counts[name] for name in ('train', 'validation', 'test')}
    declared = provenance.get('splitCounts', provenance.get('recordSplits', provenance.get('counts', {})))
    if (not counts['train'] or not counts['validation'] or type(declared) is not dict
            or any(type(declared.get(name, 0)) is not int or declared.get(name, 0) != actual[name]
                   for name in actual)):
        raise ValueError('Provenance split counts must match actual TRAIN/validation and zero TEST')
    corpus_sha = supervisor.sha(corpus)
    content_sha = content.hexdigest()
    if (provenance.get('corpusSha256') != corpus_sha
            or provenance.get('contentSha256') != content_sha or content_sha == corpus_sha
            or provenance.get('testUsedForTraining') is not False
            or not provenance.get('sources')):
        raise ValueError('Provenance must pin actual corpus/family/messages, sources and explicitly unused TEST')
    if provenance.get('objective', 'assistant_last_reply') != 'assistant_last_reply':
        raise ValueError('Pattern-learning provenance must use last-reply assistant supervision')
    return {'corpusSha256': corpus_sha, 'contentSha256': content_sha,
            'splitCounts': actual, 'familyCounts': dict(Counter(families.values()))}


def training_config():
    return {'batch_size': 4, 'micro_batch_size': 2, 'learning_rate': 0.00015,
            'weight_decay': 0.01, 'grad_clip': 1.0, 'max_epochs': 100,
            'max_steps': 512, 'max_seconds': 600, 'validation_every': 128,
            'checkpoint_every': 128, 'patience': 8, 'min_delta': 0.0001,
            'seed': 20261013, 'warmup_steps': 32, 'torch_threads': 4,
            'window_policy': 'sliding_context', 'loss_reduction': 'sequence_mean',
            'projection_mode': 'supervised', 'objective': 'assistant_last_reply',
            'partition_policy': 'train_validation_only', 'train_precision': 'cpu_bfloat16',
            'trainable_parameters': 'all', 'data_loading': 'mmap',
            'learning_rate_schedule': {'type': 'cosine', 'decay_steps': 512, 'min_lr_ratio': 0.1}}


def prepare(config_path, *, parent, corpus, manifest, run_dir, output_dir,
            cycle_id, approved_agent=AGENT, dry_run=False):
    config, config_path, config_sha = supervisor.load_config(config_path)
    if approved_agent not in config['approvedAgents']:
        raise ValueError('Standing approval agent must exactly match configured approvedAgents')
    supervisor.safe_id(cycle_id)
    root = Path(config['workspaceRoot'])
    paths = {name: checked_path(config, value, new=name in ('run', 'output')) for name, value in
             [('parent', parent), ('corpus', corpus), ('manifest', manifest),
              ('run', run_dir), ('output', output_dir)]}
    protected = [supervisor.path_for(config, config[name]) for name in ('stateDir', 'inboxDir')]
    protected.extend([paths['parent'], paths['corpus'], paths['manifest'], config_path])
    for destination in (paths['run'], paths['output']):
        if any(overlaps(destination, item) for item in protected):
            raise ValueError('New destinations cannot overlap supervisor state/inbox or frozen inputs')
    if overlaps(paths['run'], paths['output']):
        raise ValueError('New run directory and preparation bundle must be separate')
    parent_files = {name: supervisor.sha(checked_path(config, paths['parent'] / name))
                    for name in supervisor.FILES}
    parent_manifest = supervisor.read(paths['parent'] / 'manifest.json')
    parent_training = parent_manifest.get('training', {})
    if (parent_training.get('trainedFrom') not in ('random_weights', 'own_checkpoint')
            or parent_training.get('externalWeightsUsed') is True
            or (parent_training.get('trainedFrom') == 'own_checkpoint'
                and parent_training.get('lineage', {}).get('rootOrigin') != 'random_weights')):
        raise ValueError('Parent must have recorded own random-weight lineage')
    parent_config = supervisor.read(paths['parent'] / 'config.json')
    from ia_propia.conversation.configuration import validate_config
    validate_config(parent_config)
    if parent_manifest.get('config') != parent_config:
        raise ValueError('Own parent manifest/config identity differs')
    audit = audit_corpus(paths['corpus'], paths['manifest'])
    payload = encoded(training_config())
    relative = lambda path: str(path.relative_to(root))
    item = {'id': 'conversation-patterns', 'kind': 'pattern_diagnostic',
            'runDir': relative(paths['run']), 'corpusPath': relative(paths['corpus']),
            'corpusSha256': audit['corpusSha256'],
            'provenanceManifestPath': relative(paths['manifest']),
            'provenanceManifestSha256': supervisor.sha(paths['manifest']),
            'trainingConfigPath': relative(paths['output'] / 'training-config.json'),
            'trainingConfigSha256': digest_bytes(payload),
            'parentModelPath': relative(paths['parent']), 'parentFiles': parent_files,
            'stopAfterStep': 512, 'invocationBudgetSeconds': 900,
            'maxRSSMiB': 6144, 'maxDiskMiB': 1024}
    cycle = {'schemaVersion': 1, 'cycleId': cycle_id, 'stages': [item],
             'purpose': 'Bounded conversational pattern-learning diagnostic; separate probe required',
             'automaticPromotion': False}
    # Preflight needs a config file. This exclusive transient directory stays
    # inside the authorized workspace and is deleted before bundle publication.
    # The supervisor's paths, permissions and durable state are unchanged.
    with tempfile.TemporaryDirectory(prefix='.pattern-preflight-', dir=root) as temporary:
        temporary_root = Path(temporary)
        temporary_config = temporary_root / 'training-config.json'
        temporary_config.write_bytes(payload)
        probe = supervisor.Supervisor.__new__(supervisor.Supervisor)
        probe.config = config
        probe.state = {'cycle': cycle, 'stageResults': {}}
        preflight = json.loads(json.dumps(cycle))
        for field in ('runDir', 'corpusPath', 'provenanceManifestPath', 'parentModelPath'):
            preflight['stages'][0][field] = str(root / item[field])
        preflight['stages'][0]['trainingConfigPath'] = str(temporary_config)
        probe.preflight_cycle(preflight)
    native = {relative(path): supervisor.sha(path)
              for path in sorted((root / 'src/ia_propia/conversation').glob('*.py'))}
    native.update({name: supervisor.sha(root / name) for name in SOURCE_FILES})
    if not native:
        raise ValueError('Native source snapshot cannot be empty')
    cycle_bytes = encoded(cycle)
    approval = {'approved': True, 'approvedBy': approved_agent,
                'manifestSha256': digest_bytes(cycle_bytes), 'standingOwnerAuthorization': True,
                'independentHumanReview': False,
                'scope': 'This single finite own-model training stage; no promotion or new services'}
    receipt = {'schemaVersion': 1, 'preparedAtUtc': datetime.now(timezone.utc).isoformat(),
               'status': 'validated_only' if dry_run else 'prepared_not_started',
               'dryRun': dry_run, 'queued': False, 'trainingStarted': False,
               'supervisorConfigPath': str(config_path), 'supervisorConfigSha256': config_sha,
               'supervisorStateChanged': False, 'nativeProcessesStarted': False,
               'cycleId': cycle_id, 'cycleManifestSha256': digest_bytes(cycle_bytes),
               'parentModelId': parent_manifest['modelId'], 'parentContextLength': parent_config['context_length'],
               'parentFiles': parent_files, 'nativeSourcesAtPreparation': native,
               'corpusAudit': audit, 'corpusManifestSha256': item['provenanceManifestSha256'],
               'trainingConfigSha256': item['trainingConfigSha256'],
               'newStageNotExactResume': True, 'optimizerStateCopied': False,
               'genericPostStageGeneration': False, 'collectorLatestSftChanged': False,
               'limits': {'updates': 512, 'nativeSeconds': 600, 'invocationSeconds': 900,
                          'maxRSSMiB': 6144, 'maxDiskMiB': 1024},
               'outputs': {'training-config.json': digest_bytes(payload),
                           'cycle.json': digest_bytes(cycle_bytes),
                           'approval.json': digest_bytes(encoded(approval))}}
    if not dry_run:
        publication.publish(paths['output'], {'training-config.json': payload,
                            'cycle.json': cycle_bytes, 'approval.json': encoded(approval),
                            'preparation-receipt.json': encoded(receipt)})
    return receipt


def publish_file_noreplace(path, payload):
    """Publish complete bytes with one atomic no-overwrite link operation."""
    descriptor, temporary = tempfile.mkstemp(prefix='.pattern-queue-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        return path.stat().st_ino
    finally:
        os.unlink(temporary)


def enqueue(config_path, output_dir):
    config, config_path, config_sha = supervisor.load_config(config_path)
    bundle = checked_path(config, output_dir)
    receipt = supervisor.read(bundle / 'preparation-receipt.json')
    if receipt['supervisorConfigSha256'] != config_sha or receipt['supervisorConfigPath'] != str(config_path):
        raise ValueError('Supervisor configuration changed since preparation')
    root = Path(config['workspaceRoot'])
    for name, expected in receipt['nativeSourcesAtPreparation'].items():
        if supervisor.sha(checked_path(config, name)) != expected:
            raise ValueError('Frozen native source changed: ' + name)
    for name, expected in receipt['outputs'].items():
        if supervisor.sha(bundle / name) != expected:
            raise ValueError('Prepared bundle changed: ' + name)
    cycle, approval = supervisor.read(bundle / 'cycle.json'), supervisor.read(bundle / 'approval.json')
    if (approval.get('approved') is not True or approval.get('approvedBy') not in config['approvedAgents']
            or approval.get('manifestSha256') != supervisor.sha(bundle / 'cycle.json')):
        raise ValueError('Exact configured standing-agent approval is required')
    audit_corpus(checked_path(config, cycle['stages'][0]['corpusPath']),
                 checked_path(config, cycle['stages'][0]['provenanceManifestPath']))
    runner = supervisor.Supervisor.__new__(supervisor.Supervisor)
    runner.config, runner.state = config, {'cycle': cycle, 'stageResults': {}}
    runner.preflight_cycle(cycle)
    for stage in cycle['stages']:
        checked_path(config, stage['runDir'], new=True)
    inbox = checked_path(config, config['inboxDir'])
    if not inbox.is_dir():
        raise ValueError('Existing configured supervisor inbox must already exist')
    identifier = supervisor.safe_id(cycle['cycleId'])
    manifest_path, approval_path = inbox / (identifier + '.json'), inbox / (identifier + '.approval.json')
    if os.path.lexists(manifest_path) or os.path.lexists(approval_path):
        raise FileExistsError('Queued cycle/approval already exists; nothing may be overwritten')
    # Approval first, complete manifest last: the supervisor only sees a usable
    # cycle once both exact pinned files exist. A collision never replaces data.
    inode = publish_file_noreplace(approval_path, (bundle / 'approval.json').read_bytes())
    try:
        publish_file_noreplace(manifest_path, (bundle / 'cycle.json').read_bytes())
    except BaseException:
        if approval_path.exists() and approval_path.stat().st_ino == inode:
            approval_path.unlink()
        raise
    descriptor = os.open(inbox, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {'queued': True, 'admitted': False, 'trainingStarted': False,
            'manifestPath': str(manifest_path), 'manifestSha256': supervisor.sha(manifest_path),
            'approvalPath': str(approval_path), 'approvalSha256': supervisor.sha(approval_path),
            'supervisorStateChanged': False, 'nativeProcessesStarted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervisor-config', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cycle-id', required=True)
    parser.add_argument('--approved-agent', default=AGENT)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--enqueue', action='store_true')
    args = parser.parse_args()
    if args.dry_run and args.enqueue:
        parser.error('--dry-run cannot enqueue')
    receipt = prepare(args.supervisor_config, parent=args.parent, corpus=args.corpus,
                      manifest=args.manifest, run_dir=args.run_dir, output_dir=args.output_dir,
                      cycle_id=args.cycle_id, approved_agent=args.approved_agent, dry_run=args.dry_run)
    if args.enqueue:
        receipt['queuePublication'] = enqueue(args.supervisor_config, args.output_dir)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

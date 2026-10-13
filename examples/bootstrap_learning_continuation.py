"""Validate and prepare a NEW finite experiment from public own artifacts.

Does not start processes, collect sources, resume old Adam state or change LIVE.
Publication uses Linux renameat2(RENAME_NOREPLACE), as the supervisor is Linux.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import continual_training as supervisor

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'examples/learning-repair-20261013/'
TEMPLATE = PREFIX + 'configs/continual-live-collector-v2.json'
TEMPLATE_SHA = '24c561cc5a71132deb9321bdf356e46100a1820d36b62f6a706af3f26588ccf4'
AGENT = 'explicit-local-continuation-bootstrap'
PARENT = 'models/learning-repair-20261013/autonomy/context2048-untrained-parent1024'
PARENT_ID = 'unidad-conversation-ba70fff6928dcaf08087a4c966c1ea6496555bc04683933789908b4d8f49501c'
PARENT_FILES = {
    'config.json': '8d80d8ef634df02b4dfb7db463c37e1bfff3804cba137726b218bf7ea1e05ed2',
    'tokenizer.json': '848a8d7693a2527911f2e9ef4723b6b417e586e0f80d44390e057cf3e7944720',
    'weights.safetensors': '80cc46e5dfb2413355af2ce63e1c3aa6efae3760cd58c5d9f6d35b6ec270ca4a',
    'manifest.json': '8074fcfc5e9099dad2ab1cdd4ee7284ee629fbe1d5346715cf6f3aaf50720864',
}
CONFIGS = {
    'action': ('autonomy-action-2048-256.json', 'a2f5b20f71c012dfb619aac176071496a5390f4974f229e70c7e01364a9677ea', 256, 600),
    'base': ('collector-v2-base-2048.json', 'ff402d5a33da6c180bdac2fa64d0a7c0704a8c5010fa3e95219cbe403d9d99e8', 2048, 1800),
    'retention': ('natural-action-retention-1024.json', '87b2c127d537d87fb4b85188d74e8871e6bd92e7882db0d845fa56be06c049a3', 1024, 1800),
}
DATA = {
    'action': ('autonomy-action-sft-v1/autonomy-action-sft.train-validation.jsonl', 'a7c5334e532dca161f26e6205260b4f3d5989bcbb007cfa5b2728c86bdf50396', 'autonomy-action-sft-v1/manifest.json', 'b8d76767db360f73d7eebaa494bb346d1cdc2a2f27c555bfea2d4445db9398e4'),
    'wide1': ('wide-latam-v3/wide-latam.shard-001.train-validation.jsonl', '81c414480d0baf36b7784f808ba8d4759a6b7ef706eeaabdcb74d6e919cb8a81', 'wide-latam-v3/wide-latam.shard-001.manifest.json', '7a828a22f97f83c7925f0d10e2dfdc8ee3e789a40e9be458b00538a2358a44a2'),
    'wide2': ('wide-latam-v3/wide-latam.shard-002.train-validation.jsonl', '411c0a8033deca79e471918db61bdb5ba0fff72935ccfd4248ddfe046d7dcfca', 'wide-latam-v3/wide-latam.shard-002.manifest.json', '203591f9563f62926d4cb251ae66dc54de8f725214e20101f38132e5deec8940'),
    'tatoeba': ('tatoeba-next-mix-v1/tatoeba-next.shard-001.train-validation.jsonl', '93b87039b64c24d4a32cef6f5199fb6fc7c2d16dc6c402bb61fe15320e36a96e', 'tatoeba-next-mix-v1/tatoeba-next.shard-001.manifest.json', '3da147ecfbbd6741e32f5114da203bd0cbfec484c78573cf7a70503a12220bd1'),
    'retention': ('natural-action-retention-v1/natural-action-retention.train-validation.jsonl', '808e1bdd98f1447fea82f83e3d936e8724f86dc23c124d8b8e3cc9504937b7a7', 'natural-action-retention-v1/manifest.json', '51c67848ce86a1e6f78b0bd508d400701ab94f843261afebbb19e95a5e670f3c'),
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf8')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def new_output(root, value):
    root = Path(root).resolve()
    raw = Path(value)
    raw = raw if raw.is_absolute() else root / raw
    relative = raw.relative_to(root)
    if len(relative.parts) < 3 or relative.parts[:2] != ('private', 'continuations'):
        raise ValueError('New output must be private/continuations/<new-name> within this clone')
    if '..' in relative.parts:
        raise ValueError('Output cannot contain parent traversal')
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Output ancestors cannot be symlinks')
    if os.path.lexists(raw):
        raise FileExistsError('Output already exists; choose a NEW experiment directory')
    return raw


def stage(output_relative, cycle, name, data_name, config_name, parent):
    corpus, corpus_sha, provenance, provenance_sha = DATA[data_name]
    config, config_sha, steps, seconds = CONFIGS[config_name]
    value = {'id': name, 'kind': 'base' if config_name == 'base' else 'sft',
             'runDir': str(output_relative / 'runs' / cycle / name),
             'corpusPath': PREFIX + corpus, 'corpusSha256': corpus_sha,
             'trainingConfigPath': PREFIX + 'configs/' + config,
             'trainingConfigSha256': config_sha,
             'provenanceManifestPath': PREFIX + provenance,
             'provenanceManifestSha256': provenance_sha,
             'stopAfterStep': steps, 'invocationBudgetSeconds': seconds,
             'maxRSSMiB': 6144, 'maxDiskMiB': 2048}
    if parent is None:
        value.update(parentModelPath=PARENT, parentFiles=dict(PARENT_FILES))
    else:
        value['parentStage'] = parent
    return value


def build_plan(root, output):
    root, output = Path(root).resolve(), Path(output)
    if supervisor.sha(root / TEMPLATE) != TEMPLATE_SHA:
        raise ValueError('Public post-stage template differs from its pinned bytes')
    relative = output.relative_to(root)
    cycles = []
    specs = [
        ('autonomy-action-v1', [('action2048', 'action', 'action', None)]),
        ('language-wide-latam-v3', [('language-1', 'wide1', 'base', 'autonomy-action-v1:action2048'),
                                   ('language-2', 'wide2', 'base', 'language-1'),
                                   ('natural-action-retention', 'retention', 'retention', 'language-2')]),
        ('language-z-tatoeba-next-v1', [('language', 'tatoeba', 'base', 'language-wide-latam-v3:natural-action-retention'),
                                       ('natural-action-retention', 'retention', 'retention', 'language')]),
    ]
    for cycle, entries in specs:
        cycles.append({'schemaVersion': 1, 'cycleId': cycle,
                       'purpose': 'New finite experiment on published own weights; experimental, no autopromotion',
                       'stages': [stage(relative, cycle, *entry) for entry in entries]})
    hook = deepcopy(supervisor.read(root / TEMPLATE)['postStage'])
    hook['argv'][0] = sys.executable
    config = {'schemaVersion': 1, 'workspaceRoot': str(root),
              'stateDir': str(relative / 'state'), 'inboxDir': str(relative / 'inbox'),
              'pollSeconds': 10, 'maxCycles': 3, 'maxCrashRestartsPerStage': 2,
              'approvedAgents': [AGENT], 'initialCycle': cycles[0], 'postStage': hook,
              'collector': {'enabled': False, 'timeoutSeconds': 300,
                            'reason': 'Host must separately configure licensed source snapshot, historical cursor, seed and verified raw cache'},
              'limits': 'Three new finite cycles; fresh Adam per stage; original and closed LIVE budgets preserved'}
    return config, cycles


def validate_plan(root, config, cycles):
    """Pure graph/input validation; future parent receipts are never invented."""
    runner = supervisor.Supervisor.__new__(supervisor.Supervisor)
    runner.config = config
    runner.state = {'cycle': cycles[0], 'stageResults': {}}
    runner.preflight_cycle(cycles[0])  # Actual public own parent must exist now.
    pending = {}
    for cycle in cycles:
        for item in cycle['stages']:
            run_dir = supervisor.path_for(config, item['runDir'])
            if os.path.lexists(run_dir):
                raise FileExistsError('New experiment run directory already exists: ' + str(run_dir))
            reference = item.get('parentStage')
            if reference:
                key = reference if ':' in reference else cycle['cycleId'] + ':' + reference
                if key not in pending:
                    raise ValueError('Parent must be a previously declared stage: ' + key)
            runner.validate_stage(item, cycle_id=cycle['cycleId'], pending_parents=pending, preflight=True)
            pending[cycle['cycleId'] + ':' + item['id']] = run_dir
    parent = supervisor.read(Path(root) / PARENT / 'manifest.json')
    training = parent.get('training', {})
    if (parent.get('modelId') != PARENT_ID or training.get('trainedFrom') != 'own_checkpoint'
            or training.get('externalWeightsUsed') is not False or training.get('optimizerSteps') != 0
            or training.get('trainingPerformed') is not False or parent.get('config', {}).get('context_length') != 2048):
        raise ValueError('Expected published own untrained context extension identity differs')
    pins = {TEMPLATE: TEMPLATE_SHA}
    for item in (item for cycle in cycles for item in cycle['stages']):
        for field, hash_field in [('corpusPath', 'corpusSha256'), ('trainingConfigPath', 'trainingConfigSha256'),
                                  ('provenanceManifestPath', 'provenanceManifestSha256')]:
            pins[item[field]] = item[hash_field]
    pins.update({PARENT + '/' + name: value for name, value in PARENT_FILES.items()})
    pins.update(config['postStage']['sourceSha256'])
    for name, expected in pins.items():
        if supervisor.sha(Path(root) / name) != expected:
            raise ValueError('Public pinned file changed: ' + name)
    # Freeze the same native Python files in this new preparation receipt.
    package = Path(root) / 'src/ia_propia/conversation'
    native = {str(path.relative_to(root)): supervisor.sha(path) for path in sorted(package.glob('*.py'))}
    for name in ('examples/continual_training.py', 'examples/run_learning_stage.py', 'examples/run_conversation_pilot.py'):
        native[name] = supervisor.sha(Path(root) / name)
    return {'inputs': pins, 'nativeSourcesAtPreparation': native,
            'validation': 'All six actual corpora/configs/provenance checked by native pure preflight; future graph structurally valid, parent receipts rechecked at admission/launch',
            'parentModelId': PARENT_ID, 'parentContextExtensionTrained': False,
            'newExperimentNotExactResume': True, 'optimizerStateCopied': False,
            'collectorEnabled': False, 'trainingStarted': False, 'postStageThreads': 1}


def files_for_plan(root, output, config, cycles, evidence):
    config_bytes = encoded(config)
    state = {'schemaVersion': 1, 'configSha256': digest_bytes(config_bytes), 'cyclesCompleted': 0,
             'cycle': config['initialCycle'], 'stageIndex': 0, 'stageResults': {},
             'seenCycleHashes': [], 'seenCorpora': [], 'seenTrainingConfigHashes': [],
             'crashRestarts': {}, 'seenContentHashes': [], 'conversationCertification': 'awaiting_review'}
    files = {'config.json': config_bytes, 'state/state.json': encoded(state)}
    for cycle in cycles[1:]:
        name = cycle['cycleId']
        payload = encoded(cycle)
        files['inbox/' + name + '.json'] = payload
        files['inbox/' + name + '.approval.json'] = encoded({
            'approved': True, 'approvedBy': AGENT, 'manifestSha256': digest_bytes(payload),
            'scope': 'Only this newly prepared finite experiment; not promotion or source collection'})
    relative = str(Path(output).relative_to(root) / 'config.json')
    receipt = {'schemaVersion': 1, 'status': 'prepared_not_started',
               'preparedAtUtc': datetime.now(timezone.utc).isoformat(), **evidence,
               'newOutput': str(Path(output).relative_to(root)),
               'outputs': {name: digest_bytes(payload) for name, payload in files.items()},
               'commandsFromCloneRoot': {command: [sys.executable, 'examples/continual_training.py', command, '--config', relative]
                                         for command in ('status', 'start', 'stop')},
               'resumeRequirements': 'Exact native resume needs this experiment state, stage-design, native checkpoints with optimizer/scheduler/RNG, frozen source/input/parent bytes and owned stop-marker review. Exported weights alone start a new optimizer.'}
    files['preparation-receipt.json'] = encoded(receipt)
    return files, receipt


def rename_noreplace(source, target):
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = library.renameat2
    except AttributeError as error:
        raise RuntimeError('Atomic no-overwrite directory publication requires Linux renameat2') from error
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.ENOSYS, errno.EINVAL):
            raise RuntimeError('Atomic no-overwrite publication unavailable; no fallback overwrite permitted')
        raise OSError(code, os.strerror(code), str(target))


def publish(output, files):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.bootstrap-', dir=output.parent))
    try:
        for name, payload in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with target.open('xb') as stream:
                os.chmod(target, 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for directory in [path for path in staging.rglob('*') if path.is_dir()] + [staging]:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        rename_noreplace(staging, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare(root, value, *, dry_run=False):
    output = new_output(root, value)
    config, cycles = build_plan(root, output)
    evidence = validate_plan(root, config, cycles)
    evidence['dryRun'] = dry_run
    files, receipt = files_for_plan(root, output, config, cycles, evidence)
    if not dry_run:
        publish(output, files)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, help='NEW private/continuations/<experiment> within this clone')
    parser.add_argument('--dry-run', action='store_true', help='Validate all inputs without creating any files')
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.output_root, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

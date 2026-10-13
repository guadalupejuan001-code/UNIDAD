"""Durable local supervision of bounded native training stages.

This does not change the trainer, reset a resumed optimizer, grade conversations,
choose paths from prompts, or guarantee improvement/environment lifetime.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from ia_propia.conversation import jobs

FILES = ('manifest.json', 'config.json', 'tokenizer.json', 'weights.safetensors')
TERMINAL = {'completed', 'stopped', 'failed', 'stale'}
PILOT_OWNER = 'unidad-conversation-pilot-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON number: ' + value)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def safe_id(value):
    if type(value) is not str or re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,63}', value) is None:
        raise ValueError('Identifiers must be bounded filename-safe strings')
    return value


def path_for(config, value):
    if type(value) is not str or not value:
        raise ValueError('Host-configured path must be a nonempty string')
    root = Path(config['workspaceRoot']).resolve()
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root) or path == root or '.git' in path.relative_to(root).parts:
        raise ValueError('Configured paths must remain within workspaceRoot, outside .git')
    return path


def checked_file(config, stage, field, hash_field):
    path = path_for(config, stage[field])
    expected = stage[hash_field]
    if type(expected) is not str or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
        raise ValueError('Explicit lowercase SHA-256 required: ' + hash_field)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha(path) != expected:
        raise ValueError('Pinned input changed: ' + str(path))
    return path


@contextmanager
def lock(directory, *, retry_seconds=0, filename='supervisor.lock'):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / filename, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + retry_seconds
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('An active supervisor holds the exclusive lock')
                time.sleep(0.1)
        yield
    finally:
        os.close(fd)


def load_config(path):
    path = Path(path).resolve()
    config = read(path)
    if config.get('schemaVersion') != 1:
        raise ValueError('Supervisor config requires schemaVersion 1')
    root = Path(config['workspaceRoot']).resolve()
    if not root.is_dir():
        raise ValueError('workspaceRoot must exist')
    config['workspaceRoot'] = str(root)
    poll = config.get('pollSeconds', 10)
    if type(poll) not in (int, float) or not math.isfinite(poll) or not 1 <= poll <= 60:
        raise ValueError('pollSeconds must be in [1,60]')
    cycles = config.get('maxCycles')
    if cycles is not None and (type(cycles) is not int or not 1 <= cycles <= 1000000):
        raise ValueError('maxCycles is null or a positive finite integer')
    for field in ('stateDir', 'inboxDir'):
        path_for(config, config[field])
    restarts = config.get('maxCrashRestartsPerStage', 2)
    if type(restarts) is not int or not 0 <= restarts <= 3:
        raise ValueError('Crash restart limit must be an integer in [0,3]')
    config.setdefault('approvedAgents', ['project-agent-with-owner-standing-authorization'])
    if not isinstance(config['approvedAgents'], list) or not config['approvedAgents'] or any(type(x) is not str or not x for x in config['approvedAgents']):
        raise ValueError('Explicit nonempty approved-agent identifiers required')
    if config.get('collector'):
        collector = config['collector']
        if not 1 <= collector.get('timeoutSeconds', 300) <= 1800:
            raise ValueError('Collector timeout must be finite and at most 1800 seconds')
        if collector.get('enabled', False) and not collector.get('sourceSha256'):
            raise ValueError('Enabled collector requires pinned script source hashes')
    hook = config.get('postStage', {})
    if hook.get('enabled', False):
        if hook.get('purpose') != 'development_generation' or hook.get('torchThreads') != 1:
            raise ValueError('Post-stage hook only generates development conversations on one CPU thread')
        timeout = hook.get('timeoutSeconds', 300)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 300:
            raise ValueError('Post-stage generation timeout must be finite and at most 300 seconds')
        for field, maximum in (('maxRSSMiB', 6144), ('maxDiskMiB', 512)):
            if type(hook.get(field, maximum)) is not int or not 1 <= hook.get(field, maximum) <= maximum:
                raise ValueError('Invalid finite post-stage resource ceiling: ' + field)
        argv = hook.get('argv')
        if not isinstance(argv, list) or len(argv) < 2 or any(type(x) is not str for x in argv) or argv[0] != sys.executable:
            raise ValueError('Post-stage hook requires a pinned local Python argv, without a shell')
        script = path_for(config, argv[1])
        pins = hook.get('sourceSha256', {})
        if not pins or not any(path_for(config, name) == script for name in pins):
            raise ValueError('Post-stage Python script must be source-pinned')
        suite = path_for(config, hook['suiteDir'])
        if any('reserved' in part.lower() for part in suite.parts):
            raise ValueError('Post-stage hook cannot open a reserved suite')
        for name in ('cases.json', 'rubric.json', 'freeze.json'):
            if not any(path_for(config, filename) == suite / name for filename in pins):
                raise ValueError('Post-stage development suite must be pinned: ' + name)
        for flag, expected in (('--torch-threads', '1'), ('--model-dir', '{snapshot}'), ('--output-dir', '{output_dir}')):
            if flag not in argv or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag) + 1] != expected:
                raise ValueError('Post-stage generation argv requires ' + flag + ' ' + expected)
        if 'generate' not in argv or '--suite-dir' not in argv or path_for(config, argv[argv.index('--suite-dir') + 1]) != suite:
            raise ValueError('Post-stage hook must use the pinned development suite and generate command')
    return config, path, sha(path)


class Supervisor:
    def __init__(self, config, config_path, config_sha):
        self.config, self.config_path, self.config_sha = config, config_path, config_sha
        self.directory = path_for(config, config['stateDir'])
        self.inbox = path_for(config, config['inboxDir'])
        self.state_path = self.directory / 'state.json'
        self.children = {}
        self.last_event = None
        self.state = read(self.state_path) if self.state_path.exists() else {
            'schemaVersion': 1, 'configSha256': config_sha, 'cyclesCompleted': 0,
            'cycle': config['initialCycle'], 'stageIndex': 0, 'stageResults': {},
            'seenCycleHashes': [], 'seenCorpora': [], 'seenTrainingConfigHashes': [],
            'crashRestarts': {}, 'seenContentHashes': [], 'conversationCertification': 'awaiting_review',
        }
        if self.state['configSha256'] != config_sha:
            raise ValueError('Supervisor configuration changed; preserve state and use an explicit new state directory')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.inbox.mkdir(parents=True, exist_ok=True, mode=0o700)

    def event(self, kind, **values):
        record = {'atUtc': now(), 'event': kind, **values}
        with (self.directory / 'journal.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def update(self, status, **values):
        worker = dict(values.get('worker', self.state.get('worker', {})))
        worker['processAlive'] = jobs._alive(worker.get('pid'), worker.get('processBirth'))
        values['worker'] = worker
        values['nativeWorkerAlive'] = bool(worker.get('processAlive'))
        values['trainingActive'] = bool(worker.get('processAlive') and worker.get('status') == 'running')
        generation = self.state.get('postStageChild', {})
        values['generationChildAlive'] = jobs._alive(generation.get('pid'), generation.get('processBirth'))
        self.state.update({'operationalState': status, 'heartbeatAtUnix': time.time(),
            'supervisorPid': os.getpid(), 'supervisorBirth': jobs._birth(os.getpid()),
            'environmentLifetimeGuaranteed': False, **values})
        atomic(self.state_path, self.state)
        signature = json.dumps({'state': status, 'cycle': self.state.get('cycle', {}).get('cycleId'),
            'stageIndex': self.state['stageIndex'], 'workerPid': values.get('worker', {}).get('pid'),
            'reason': values.get('reason')}, sort_keys=True)
        if signature != self.last_event:
            self.event('state', operationalState=status, **values)
            self.last_event = signature

    def cycle_key(self, stage):
        return safe_id(self.state['cycle']['cycleId']) + ':' + safe_id(stage['id'])

    def parent(self, stage):
        if stage.get('parentStage'):
            key = stage['parentStage']
            if ':' not in key:
                key = self.state['cycle']['cycleId'] + ':' + key
            result = self.state['stageResults'].get(key)
            if not result:
                raise FileNotFoundError('Parent stage receipt not available: ' + key)
            parent = path_for(self.config, result['snapshot'])
            for name, digest in result['snapshotFiles'].items():
                if sha(parent / name) != digest:
                    raise ValueError('Completed own parent artifact changed')
            if read(parent / 'manifest.json')['modelId'] != result['modelId']:
                raise ValueError('Own parent model identity changed')
            return parent
        if stage.get('parentModelPath'):
            parent = path_for(self.config, stage['parentModelPath'])
            for name in FILES:
                if sha(parent / name) != stage['parentFiles'][name]:
                    raise ValueError('Pinned own parent artifact changed')
            return parent
        return None

    def validate_stage(self, stage):
        safe_id(stage['id'])
        run_dir = path_for(self.config, stage['runDir'])
        corpus = checked_file(self.config, stage, 'corpusPath', 'corpusSha256')
        config_path = checked_file(self.config, stage, 'trainingConfigPath', 'trainingConfigSha256')
        training = read(config_path)
        if (training.get('objective') not in ('causal_lm', 'assistant_last_reply')
            or training.get('partition_policy') != 'train_validation_only'
            or training.get('data_loading') != 'mmap' or training.get('trainable_parameters') != 'all'):
            raise ValueError('Only native TRAIN/validation-only mmap, all-parameter causal/SFT stages are supported')
        if (type(training['max_seconds']) not in (int, float) or not math.isfinite(training['max_seconds'])
            or not 0 < training['max_seconds'] <= 7200 or type(training['torch_threads']) is not int
            or not 1 <= training['torch_threads'] <= 4):
            raise ValueError('Persistent stage limit is at most two hours and four CPU threads')
        if type(training['max_steps']) is not int or not 1 <= training['max_steps'] <= 12000:
            raise ValueError('Persistent step limit must be finite')
        if stage['stopAfterStep'] != training['max_steps']:
            raise ValueError('Supervisor does not reset or extend persistent step limits')
        budget = stage.get('invocationBudgetSeconds', 3600)
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 3600:
            raise ValueError('Invocation budget must be finite and at most one hour')
        if (type(stage.get('maxRSSMiB', 6144)) is not int or type(stage.get('maxDiskMiB', 2048)) is not int
            or not 256 <= stage.get('maxRSSMiB', 6144) <= 6144 or not 256 <= stage.get('maxDiskMiB', 2048) <= 2048):
            raise ValueError('Stage resource ceilings exceed 6GiB RSS / 2GiB run disk')
        if stage.get('provenanceManifestPath'):
            provenance = read(checked_file(self.config, stage, 'provenanceManifestPath', 'provenanceManifestSha256'))
            splits = provenance.get('splitCounts', provenance.get('recordSplits', provenance.get('counts', {})))
            if (provenance.get('corpusSha256') != stage['corpusSha256'] or provenance.get('testUsedForTraining') is not False
                or type(splits.get('train')) is not int or splits['train'] < 1
                or type(splits.get('validation')) is not int or splits['validation'] < 1 or splits.get('test', 0) != 0
                or not provenance.get('sources')):
                raise ValueError('Provenance must identify this corpus, human/public source records, TRAIN/validation and explicitly unused TEST')
        model = checked_file(self.config, stage, 'modelConfigPath', 'modelConfigSha256') if stage.get('modelConfigPath') else None
        parent = self.parent(stage)
        if (model is None) == (parent is None):
            raise ValueError('Stage requires exactly one random model config or own parent stage/artifact')
        if parent and (run_dir == parent or run_dir.is_relative_to(parent) or parent.is_relative_to(run_dir)):
            raise ValueError('Stage directory overlaps own parent artifact')
        if parent and read(parent / 'manifest.json').get('training', {}).get('trainedFrom') not in ('random_weights', 'own_checkpoint'):
            raise ValueError('Only recorded own-model lineage is allowed')
        return run_dir, corpus, config_path, model, parent, training

    def guard_existing(self, stage, run_dir, training):
        design = read(run_dir / 'stage-design.json')
        if design['corpusSha256'] != stage['corpusSha256'] or Path(design['corpusPath']) != path_for(self.config, stage['corpusPath']):
            raise ValueError('Existing run corpus identity differs')
        if any(design['trainingConfig'].get(k) != v for k, v in training.items()):
            raise ValueError('Existing native configuration differs; no Adam/scheduler reset allowed')
        import run_learning_stage
        if stage.get('modelConfigPath'):
            model_config = read(path_for(self.config, stage['modelConfigPath']))
            requested, _ = run_learning_stage.training._configs(model_config, training)
            if design.get('parentPath') is not None or requested != design.get('randomModelConfig'):
                raise ValueError('Existing random model architecture differs from pinned configuration')
        else:
            parent = self.parent(stage)
            if str(parent) != design.get('parentPath') or any(sha(parent / name) != digest for name, digest in design.get('parentFiles', {}).items()):
                raise ValueError('Existing own parent path or artifact hashes differ')
        if run_learning_stage.source_hashes() != design['sourceSha256']:
            raise ValueError('Frozen native source changed')
        if design['recordSplits'].get('test', 0) or not design['recordSplits'].get('train') or not design['recordSplits'].get('validation'):
            raise ValueError('Native stage split record must be TRAIN/validation with no TEST')
        ceilings = design['resourceCeilings']
        if ceilings['rssMiB'] != stage.get('maxRSSMiB', 6144) or ceilings['runDiskMiB'] != stage.get('maxDiskMiB', 2048):
            raise ValueError('Existing native resource ceilings changed')
        return design

    def free_space_guard(self, max_disk_mib=2048):
        stat = os.statvfs(self.config['workspaceRoot'])
        available = stat.f_bavail * stat.f_frsize
        required = (max_disk_mib + 512) * 1024**2
        if available < required:
            self.update('blocked_resources', reason='workspace_free_space', freeWorkspaceBytes=available, requiredFreeBytes=required)
            return False
        return True

    def launch(self, stage, run_dir, corpus, config_path, model, parent, *, resume):
        status = jobs.status(run_dir)
        if status.get('processAlive'):
            self.update('training', worker=status, adopted=True)
            return
        if not self.free_space_guard(stage.get('maxDiskMiB', 2048)):
            return
        command = [sys.executable, str(Path(__file__).with_name('run_learning_stage.py')),
            '--corpus', str(corpus), '--expected-corpus-sha256', stage['corpusSha256'],
            '--run-dir', str(run_dir), '--training-config', str(config_path),
            '--stop-after-step', str(stage['stopAfterStep']),
            '--budget-seconds', str(stage.get('invocationBudgetSeconds', 3600)),
            '--max-rss-mib', str(stage.get('maxRSSMiB', 6144)),
            '--max-disk-mib', str(stage.get('maxDiskMiB', 2048))]
        if resume:
            command.append('--resume')
        elif model:
            command += ['--model-config', str(model)]
        else:
            command += ['--parent-model', str(parent)]
        log = self.directory / ('worker-' + self.cycle_key(stage).replace(':', '-') + '.log')
        with log.open('ab') as stream:
            process = subprocess.Popen(command, cwd=self.config['workspaceRoot'], stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        self.children[process.pid] = process
        launch = {'pid': process.pid, 'processBirth': jobs._birth(process.pid), 'startedAtUnix': time.time(),
            'resume': resume, 'command': command, 'logPath': str(log), 'stageKey': self.cycle_key(stage)}
        self.state['launch'] = launch
        self.event('child_started', **launch)
        self.update('worker_starting', launch=launch, worker={'pid': process.pid, 'processBirth': launch['processBirth'], 'processAlive': True})

    def cooperative_stop(self, stage):
        run_dir = path_for(self.config, stage['runDir'])
        status = jobs.status(run_dir)
        if not status.get('processAlive'):
            return
        _, _, _, _, _, training = self.validate_stage(stage)
        design = self.guard_existing(stage, run_dir, training)
        if not jobs._alive(status['pid'], status['processBirth']):
            return
        path = run_dir / 'stop-request.json'
        if path.exists():
            self.event('existing_stop_preserved', path=str(path))
            return
        request = {'owner': PILOT_OWNER, 'runnerId': design['runnerId'],
            'pid': status['pid'], 'processBirth': status['processBirth'],
            'reason': 'continual_supervisor_stop', 'requestedAtUtc': now()}
        # Exclusive creation preserves a concurrently written native/user stop.
        temporary = path.with_name('.stop-supervisor-' + uuid.uuid4().hex)
        try:
            with temporary.open('x', encoding='utf8') as stream:
                json.dump(request, stream); stream.write('\n')
                stream.flush(); os.fsync(stream.fileno())
            temporary.chmod(0o600)
            os.link(temporary, path)  # Exclusive atomic publication of complete bytes.
        except FileExistsError:
            return
        finally:
            temporary.unlink(missing_ok=True)
        self.event('native_cooperative_stop_requested', **request)

    def recover_export(self, stage, status):
        key = self.cycle_key(stage)
        if self.state.get('exportRecoveryAttempts', {}).get(key, 0) >= 1:
            raise ValueError('Export recovery limit reached; coherent native checkpoints preserved')
        self.state.setdefault('exportRecoveryAttempts', {})[key] = 1
        self.update('recovering_export', worker=status)
        self.event('export_recovery_started', stageKey=key, checkpoint=status.get('bestCheckpoint'), trainingPerformed=False)
        from ia_propia.conversation import training
        import run_learning_stage
        run_dir = path_for(self.config, stage['runDir'])
        if jobs.status(run_dir).get('processAlive'):
            raise RuntimeError('A native process became live before export recovery')
        snapshot = run_dir / 'snapshots' / ('recovered-' + hashlib.sha256(status['bestCheckpoint'].encode()).hexdigest()[:16])
        with training._run_lock(run_dir):
            if jobs.status(run_dir).get('processAlive'):
                raise RuntimeError('A native process became live during export recovery')
            if snapshot.exists():
                manifest = read(snapshot / 'manifest.json')
            else:
                manifest = training.export_best(run_dir, snapshot)
            if manifest['training']['selectedCheckpoint'] != status['bestCheckpoint']:
                raise ValueError('Recovered export checkpoint identity differs')
            observations = run_learning_stage.resource_snapshot(run_dir)
            if observations['peakRssBytes'] > stage.get('maxRSSMiB', 6144) * 1024**2 or observations['runDiskBytes'] > stage.get('maxDiskMiB', 2048) * 1024**2:
                raise ValueError('Recovered export exceeds configured resource ceilings')
            _, _, _, _, _, config = self.validate_stage(stage)
            self.guard_existing(stage, run_dir, config)
            value = {'schemaVersion': 1, 'nativeStatus': status, 'snapshot': str(snapshot),
                'modelId': manifest['modelId'], 'selectedStep': manifest['training']['selectedStep'],
                'snapshotFiles': {name: sha(snapshot / name) for name in FILES},
                'sourceAndInputsUnchanged': True, 'resourceMonitorErrors': [], 'cooperativeStopReasons': [],
                'exportRecovered': True, 'exportOnlyResourceObservations': observations,
                'originalInvocationResourceObservationsUnavailable': True, 'trainingPerformedByRecovery': False,
                'createdAtUtc': now()}
            atomic(run_dir / 'stage-recovery-latest.json', value)
        self.event('export_recovery_completed', stageKey=key, modelId=value['modelId'], trainingPerformed=False)
        return value

    def complete_stage(self, stage, status):
        run_dir = path_for(self.config, stage['runDir'])
        identity = (status.get('pid'), status.get('processBirth'), status.get('lastCheckpoint'))
        # Reuse an exact recovered receipt before consuming the one recovery attempt.
        # Keep its path paired with its bytes; never hash a missing/stale original.
        for receipt_path in (run_dir / 'stage-latest.json', run_dir / 'stage-recovery-latest.json'):
            if not receipt_path.exists():
                continue
            latest = read(receipt_path)
            current = latest.get('nativeStatus', {})
            if (current.get('pid'), current.get('processBirth'), current.get('lastCheckpoint')) == identity:
                break
        else:
            latest = self.recover_export(stage, status)
            receipt_path = run_dir / 'stage-recovery-latest.json'
        native = latest['nativeStatus']
        if (native.get('pid'), native.get('processBirth'), native.get('lastCheckpoint')) != (status.get('pid'), status.get('processBirth'), status.get('lastCheckpoint')):
            raise FileNotFoundError('Latest export receipt does not match terminal invocation yet')
        if not latest.get('sourceAndInputsUnchanged') or latest.get('resourceMonitorErrors'):
            raise ValueError('Stage export has invalid source/input/resource observations')
        parent = path_for(self.config, latest['snapshot'])
        for name in FILES:
            if sha(parent / name) != latest['snapshotFiles'][name]:
                raise ValueError('Exported snapshot digest changed')
        if read(parent / 'manifest.json')['modelId'] != latest['modelId']:
            raise ValueError('Snapshot model identity differs from export receipt')
        self.state['stageResults'][self.cycle_key(stage)] = {
            'snapshot': str(parent), 'modelId': latest['modelId'], 'snapshotFiles': latest['snapshotFiles'],
            'kind': stage.get('kind', 'sft' if read(path_for(self.config, stage['trainingConfigPath']))['objective'] == 'assistant_last_reply' else 'base'),
            'selectedStep': latest['selectedStep'], 'nativeFinalStep': status.get('step'),
            'nativeFinalReason': status.get('reason'), 'receiptPath': str(receipt_path), 'receiptSha256': sha(receipt_path),
            'runDir': str(run_dir), 'conversationCertification': 'awaiting_review'}
        self.event('stage_completed', stageKey=self.cycle_key(stage), **self.state['stageResults'][self.cycle_key(stage)])
        self.state['seenCorpora'] = sorted(set(self.state['seenCorpora'] + [stage['corpusSha256']]))
        self.state['seenTrainingConfigHashes'] = sorted(set(self.state['seenTrainingConfigHashes'] + [stage['trainingConfigSha256']]))
        if stage.get('provenanceManifestPath'):
            content = read(path_for(self.config, stage['provenanceManifestPath'])).get('contentSha256')
            if content:
                self.state['seenContentHashes'] = sorted(set(self.state.get('seenContentHashes', []) + [content]))
        self.state['stageIndex'] += 1
        self.state.pop('launch', None)
        if self.config.get('postStage', {}).get('enabled', False) and self.state['stageResults'][self.cycle_key(stage)]['kind'] == 'sft':
            self.state.setdefault('pendingPostStages', []).append(self.cycle_key(stage))
        self.update('stage_complete', worker=status)

    def stop_post_stage(self, active, reason):
        if jobs._alive(active['pid'], active['processBirth']):
            os.kill(active['pid'], signal.SIGTERM)
            active['stopRequestedAtUnix'] = time.time()
            active['stopReason'] = reason
            self.event('post_stage_stop_requested', stageKey=active['stageKey'], pid=active['pid'], reason=reason)

    def post_stage_tick(self):
        """Generate raw development history once; never grade or promote a model."""
        hook = self.config.get('postStage', {})
        active = self.state.get('postStageChild')
        if active:
            alive = jobs._alive(active['pid'], active['processBirth'])
            if alive:
                reason = None
                if time.time() - active['startedAtUnix'] >= hook.get('timeoutSeconds', 300):
                    reason = 'generation_time_budget'
                try:
                    with Path('/proc', str(active['pid']), 'status').open() as stream:
                        rss = next((int(line.split()[1]) * 1024 for line in stream if line.startswith('VmRSS:')), None)
                except OSError:
                    rss = None
                if rss is not None:
                    active['rssSamples'] = active.get('rssSamples', 0) + 1
                    active['peakObservedRssBytes'] = max(active.get('peakObservedRssBytes', 0), rss)
                output_dir = path_for(self.config, active['outputDir'])
                disk = sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file()) if output_dir.exists() else 0
                active['observedOutputBytes'] = disk
                if rss is not None and rss > hook.get('maxRSSMiB', 6144) * 1024**2:
                    reason = 'generation_rss_ceiling'
                if disk > hook.get('maxDiskMiB', 512) * 1024**2:
                    reason = 'generation_disk_ceiling'
                if reason and not active.get('stopRequestedAtUnix'):
                    self.stop_post_stage(active, reason)
                elif active.get('stopRequestedAtUnix') and time.time() - active['stopRequestedAtUnix'] >= 10:
                    # This is only the owned optional generation child, never a native trainer.
                    os.kill(active['pid'], signal.SIGKILL)
                    self.event('post_stage_stop_escalated', stageKey=active['stageKey'], pid=active['pid'])
                self.update('generating_development', generationChild=active)
                return True
            child = self.children.get(active['pid'])
            code = child.poll() if child else None
            path = path_for(self.config, active['outputDir']) / 'outputs.json'
            result = {'stageKey': active['stageKey'], 'pid': active['pid'], 'processBirth': active['processBirth'],
                'knownExitCode': code, 'terminalState': 'exit_observed' if code is not None else 'dead_exit_code_unknown',
                'finishedAtUtc': now(), 'startedAtUnix': active['startedAtUnix'], 'command': active['command'],
                'sourceSha256': active['sourceSha256'], 'outputDir': active['outputDir'],
                'nativeSourceManifestSha256': active['nativeSourceManifestSha256'],
                'peakObservedRssBytes': active.get('peakObservedRssBytes'), 'rssSamples': active.get('rssSamples', 0),
                'rssScope': 'sampled generation child /proc VmRSS, not kernel peak or trainer memory',
                'stopReason': active.get('stopReason'), 'conversationCertification': 'awaiting_review'}
            if path.exists():
                result.update({'outputsPath': str(path), 'outputsSha256': sha(path), 'semanticReviewPerformed': False})
                try:
                    import run_learning_stage
                    current_sources = hashlib.sha256(json.dumps(run_learning_stage.source_hashes(), sort_keys=True).encode()).hexdigest()
                    if current_sources != active['nativeSourceManifestSha256']:
                        raise ValueError('Native engine source changed during development generation')
                    for filename, digest in active['sourceSha256'].items():
                        if sha(path_for(self.config, filename)) != digest:
                            raise ValueError('Generation script or development suite changed during generation')
                    raw = read(path)
                    own = self.state['stageResults'][active['stageKey']]
                    if raw.get('modelId') != own['modelId'] or raw.get('generation') != {'temperature': 0, 'seed': 42, 'maxNewTokens': 160, 'torchThreads': 1}:
                        raise ValueError('Generation output identity or frozen sampling settings differ')
                    result.update({'turns': len(raw.get('outputs', [])),
                        'generationErrors': sum(row.get('error') is not None for row in raw.get('outputs', [])),
                        'generationEvidence': 'raw_outputs_present'})
                except (ValueError, KeyError, TypeError) as error:
                    result.update({'generationEvidence': 'raw_outputs_require_integrity_review', 'validationError': str(error)})
            else:
                result['generationEvidence'] = 'no_complete_outputs_file'
            self.state.setdefault('postStageResults', {})[active['stageKey']] = result
            self.state['stageResults'][active['stageKey']]['developmentGeneration'] = result
            self.state.pop('postStageChild', None)
            self.event('post_stage_terminal', **result)
            self.update('development_generation_terminal')
            return True
        pending = self.state.get('pendingPostStages', [])
        if not pending:
            return False
        if not hook.get('enabled', False):
            self.update('waiting_for_plan', reason='post_stage_generation_hook_disabled_with_pending_stage')
            return True
        if not self.free_space_guard(hook.get('maxDiskMiB', 512)):
            return True
        key = pending[0]
        if key in self.state.get('postStageResults', {}):
            pending.pop(0)
            return True
        result = self.state['stageResults'][key]
        import run_learning_stage
        design = read(path_for(self.config, result['runDir']) / 'stage-design.json')
        native_sources = run_learning_stage.source_hashes()
        if native_sources != design['sourceSha256']:
            raise ValueError('Frozen native engine source changed before development generation')
        snapshot = path_for(self.config, result['snapshot'])
        for name, digest in result['snapshotFiles'].items():
            if sha(snapshot / name) != digest:
                raise ValueError('Post-stage own snapshot changed')
        for filename, digest in hook['sourceSha256'].items():
            if sha(path_for(self.config, filename)) != digest:
                raise ValueError('Pinned post-stage source or development suite changed')
        parent = self.directory / 'development-generations'
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        output_dir = parent / (key.replace(':', '-') + '-' + result['modelId'].removeprefix('unidad-conversation-')[:16])
        if output_dir.exists():
            # An interrupted launch is preserved for review; do not overwrite or invent its exit.
            self.state.setdefault('postStageResults', {})[key] = {'generationEvidence': 'existing_output_directory_preserved',
                'outputDir': str(output_dir), 'conversationCertification': 'awaiting_review'}
            self.event('post_stage_existing_output_preserved', stageKey=key, outputDir=str(output_dir))
            pending.pop(0)
            self.update('development_generation_terminal')
            return True
        command = [arg.replace('{snapshot}', str(snapshot)).replace('{output_dir}', str(output_dir)) for arg in hook['argv']]
        command[1] = str(path_for(self.config, command[1]))
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        log = self.directory / ('generation-' + key.replace(':', '-') + '.log')
        with log.open('ab') as stream:
            child = subprocess.Popen(command, cwd=self.config['workspaceRoot'], env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        self.children[child.pid] = child
        active = {'stageKey': key, 'pid': child.pid, 'processBirth': jobs._birth(child.pid),
            'startedAtUnix': time.time(), 'command': command, 'sourceSha256': hook['sourceSha256'],
            'nativeSourceManifestSha256': hashlib.sha256(json.dumps(native_sources, sort_keys=True).encode()).hexdigest(),
            'outputDir': str(output_dir), 'logPath': str(log), 'modelId': result['modelId']}
        self.state['postStageChild'] = active
        pending.pop(0)
        self.event('post_stage_started', **active)
        self.update('generating_development', generationChild=active)
        return True

    def review_receipt(self, cycle):
        # Receipt approval is a project-agent record under standing owner authorization,
        # never claimed to be human review or a model-generated certification.
        if not cycle.get('reviewReceiptPath') or not cycle.get('reviewApprovalPath'):
            return 'awaiting_review'
        receipt_path = path_for(self.config, cycle['reviewReceiptPath'])
        approval_path = path_for(self.config, cycle['reviewApprovalPath'])
        if not receipt_path.exists() or not approval_path.exists():
            return 'awaiting_review'
        approval = read(approval_path)
        if approval.get('approved') is not True or approval.get('receiptSha256') != sha(receipt_path) or approval.get('approvedBy') not in self.config['approvedAgents']:
            raise ValueError('Review receipt lacks exact trusted local approval digest')
        receipt = read(receipt_path)
        result = self.state['stageResults'][self.cycle_key(cycle['stages'][-1])]
        if receipt.get('modelId') != result['modelId'] or receipt.get('artifactFiles') != result['snapshotFiles']:
            raise ValueError('Review refers to a different artifact')
        if (receipt.get('engine') != 'ConversationEngine' or receipt.get('actualGeneratedHistory') is not True
            or receipt.get('candidateFrozenBeforeReservedOpening') is not True or receipt.get('freshProspectiveSuite') is not True):
            raise ValueError('Certification requires fresh frozen 40-turn real-history common-engine evidence')
        outputs_path = checked_file(self.config, receipt, 'outputsPath', 'outputsSha256')
        assessment_path = checked_file(self.config, receipt, 'assessmentPath', 'assessmentSha256')
        outputs, assessment = read(outputs_path), read(assessment_path)
        if outputs['modelId'] != result['modelId'] or assessment['modelId'] != result['modelId'] or assessment['outputsSha256'] != sha(outputs_path):
            raise ValueError('Assessment and raw outputs do not identify the same frozen artifact')
        rows = outputs['outputs']
        if len(rows) != 40 or len({x['id'] for x in rows}) != 40 or len({x['scenarioId'] for x in rows}) != 10:
            raise ValueError('Certification requires forty unique turns in ten dialogues')
        for scenario in {x['scenarioId'] for x in rows}:
            history = []
            turns = [x for x in rows if x['scenarioId'] == scenario]
            if len(turns) != 4:
                raise ValueError('Each certified dialogue must have four turns')
            for row in turns:
                messages = row['inputMessages']
                if not messages or messages[-1]['role'] != 'user' or messages[:-1] != history:
                    raise ValueError('Certification history differs from actual prior generated replies')
                history = messages + [{'role': 'assistant', 'content': row['reply']['text']}]
                if row['error'] or not row['reply']['text'].strip() or row['reply'].get('discardedPromptTokens') or row['reply'].get('stopReason') == 'token_budget':
                    return 'rejected'
        if outputs.get('modelManifest') != read(path_for(self.config, result['snapshot']) / 'manifest.json'):
            raise ValueError('Raw generation manifest differs from certified own artifact')
        review_path = checked_file(self.config, receipt, 'reviewPath', 'reviewSha256')
        suite_dir = path_for(self.config, receipt['suiteDir'])
        frozen = read(checked_file(self.config, receipt, 'suiteFreezePath', 'suiteFreezeSha256'))
        if outputs.get('suiteFreeze') != frozen:
            raise ValueError('Outputs do not preserve this exact frozen suite')
        for name in ('cases.json', 'rubric.json'):
            if sha(suite_dir / name) != frozen['files'][name]['sha256']:
                raise ValueError('Certified suite file changed')
        cases, review = read(suite_dir / 'cases.json'), read(review_path)
        expected = {step['id']: step for scenario in cases['scenarios'] for step in scenario['steps']}
        judgments = {row['id']: row for row in review['turns']}
        if (len(review['turns']) != 40 or len(judgments) != 40 or set(judgments) != set(expected)
            or set(judgments) != {row['id'] for row in rows} or review.get('outputsSha256') != sha(outputs_path)
            or review.get('modelId') != result['modelId'] or not review.get('reviewer') or not review.get('method')):
            raise ValueError('Forty exact reviewed case IDs and reviewer provenance required')
        full, critical = {}, []
        for row in rows:
            identifier = row['id']; judge, case = judgments[identifier], expected[identifier]
            if row['inputMessages'][-1]['content'] != case['user'] or row['reply'].get('modelId') != result['modelId']:
                raise ValueError('Certified prompt or generated reply identity differs')
            if any(type(judge.get(k)) is not bool for k in ('relevance', 'contextGrounding', 'spanish')):
                raise ValueError('Semantic judgments need explicit dimensions')
            if case['critical'] and type(judge.get('criticalCorrect')) is not bool:
                raise ValueError('Critical semantic judgment missing')
            if case['critical'] and not judge['criticalCorrect']:
                critical.append(identifier)
            full[identifier] = all(judge[k] for k in ('relevance', 'contextGrounding', 'spanish')) and (not case['critical'] or judge['criticalCorrect'])
        full_dialogs = sum(all(full[step['id']] for step in scenario['steps']) for scenario in cases['scenarios'])
        if (assessment.get('fullyMetTurns') != sum(full.values()) or assessment.get('fullyMetDialogs') != full_dialogs
            or set(assessment.get('criticalFailures', [])) != set(critical) or assessment.get('reviewSha256') != sha(review_path)):
            raise ValueError('Assessment counters do not match exact frozen semantic review')
        freeze_path = checked_file(self.config, receipt, 'candidateFreezePath', 'candidateFreezeSha256')
        candidate_freeze = read(freeze_path)
        if candidate_freeze.get('modelId') != result['modelId'] or candidate_freeze.get('files') != result['snapshotFiles']:
            raise ValueError('Prospective candidate freeze does not identify own artifact')
        frozen_at = datetime.fromisoformat(candidate_freeze['createdAtUtc'].replace('Z', '+00:00')).timestamp()
        if outputs['startedAtUnix'] <= frozen_at:
            raise ValueError('Reserved generation preceded candidate freeze')
        digest = receipt['suiteFreezeSha256']
        previous = self.state.get('reviewedSuiteReceipts', {}).get(digest)
        if previous is not None and previous != sha(receipt_path):
            raise ValueError('A consumed acceptance suite cannot certify a new candidate')
        self.state.setdefault('reviewedSuiteReceipts', {})[digest] = sha(receipt_path)
        accepted = (assessment.get('accepted') is True and assessment.get('totalTurns') == 40
            and assessment.get('fullyMetTurns', 0) >= 36 and assessment.get('totalDialogs') == 10
            and assessment.get('fullyMetDialogs', 0) >= 8 and assessment.get('criticalFailures') == []
            and assessment.get('technicalFailures') == [])
        self.event('trusted_review_recorded', modelId=result['modelId'], receiptSha256=sha(receipt_path), accepted=accepted,
            projectAIReview=True, independentHumanReview=False, autopromotionPerformed=False)
        return 'accepted_basic_suite' if accepted else 'rejected'

    def abandon_cycle(self, stage, reason):
        self.state['cycleFailed'] = reason
        self.state['seenCorpora'] = sorted(set(self.state['seenCorpora'] + [stage['corpusSha256']]))
        self.state['seenTrainingConfigHashes'] = sorted(set(self.state['seenTrainingConfigHashes'] + [stage['trainingConfigSha256']]))
        if stage.get('provenanceManifestPath'):
            content = read(path_for(self.config, stage['provenanceManifestPath'])).get('contentSha256')
            if content:
                self.state['seenContentHashes'] = sorted(set(self.state.get('seenContentHashes', []) + [content]))
        self.state['stageIndex'] = len(self.state['cycle']['stages'])
        self.state.pop('launch', None)

    def next_cycle(self):
        for manifest in sorted(self.inbox.glob('*.json')):
            if manifest.name.endswith('.approval.json'):
                continue
            approval_path = manifest.with_name(manifest.stem + '.approval.json')
            if not approval_path.exists():
                continue
            digest = sha(manifest)
            if digest in self.state['seenCycleHashes']:
                continue
            approval = read(approval_path)
            if approval.get('approved') is not True or approval.get('manifestSha256') != digest or approval.get('approvedBy') not in self.config['approvedAgents']:
                continue
            cycle = read(manifest)
            safe_id(cycle['cycleId'])
            stages = cycle.get('stages')
            if not isinstance(stages, list) or not stages:
                raise ValueError('Approved cycle has no stages')
            if not any(s['corpusSha256'] not in self.state['seenCorpora'] for s in stages):
                self.event('cycle_refused_repeated_inputs', manifest=str(manifest), manifestSha256=digest)
                self.state['seenCycleHashes'].append(digest)
                continue
            novel = False
            for stage in stages:
                if stage.get('corpusSha256') in self.state['seenCorpora']:
                    continue
                provenance = read(checked_file(self.config, stage, 'provenanceManifestPath', 'provenanceManifestSha256'))
                content = provenance.get('contentSha256')
                if type(content) is not str or re.fullmatch(r'[0-9a-f]{64}', content) is None:
                    raise ValueError('New-cycle provenance requires a content-only SHA distinct from metadata')
                if content not in self.state.get('seenContentHashes', []):
                    novel = True
            if not novel:
                self.event('cycle_refused_no_new_content', manifestSha256=digest)
                self.state['seenCycleHashes'].append(digest)
                continue
            if not all(s.get('provenanceManifestPath') and s.get('provenanceManifestSha256') for s in stages):
                raise ValueError('New cycles require pinned corpus provenance manifests')
            self.state['seenCycleHashes'].append(digest)
            self.state.update({'cycle': cycle, 'stageIndex': 0, 'conversationCertification': 'awaiting_review'})
            self.event('cycle_admitted', cycleId=cycle['cycleId'], manifestSha256=digest,
                approvedBy=approval['approvedBy'], standingOwnerAuthorization=True, independentHumanReview=False)
            return True
        return False

    def collect_if_configured(self):
        collector = self.config.get('collector', {})
        if not collector.get('enabled', False):
            return
        active = self.state.get('collectorChild')
        if active and jobs._alive(active['pid'], active['processBirth']):
            if time.time() - active['startedAtUnix'] > collector.get('timeoutSeconds', 300):
                os.kill(active['pid'], signal.SIGTERM)
                self.event('collector_timeout_stop', **active)
            return
        if active:
            process = self.children.get(active['pid'])
            code = process.poll() if process else None
            self.event('collector_terminal', pid=active['pid'], knownExitCode=code,
                terminalState='exit_observed' if code is not None else 'dead_exit_code_unknown')
            self.state.pop('collectorChild', None)
        if time.time() - self.state.get('lastCollectorLaunchAt', 0) < collector.get('everySeconds', 3600):
            return
        if not self.free_space_guard(collector.get('maxDiskMiB', 2048)):
            return False
        for filename, digest in collector['sourceSha256'].items():
            if sha(path_for(self.config, filename)) != digest:
                raise ValueError('Collector pinned source changed')
        command = [x.replace('{inbox_dir}', str(self.inbox)) for x in collector['argv']]
        with (self.directory / 'collector.log').open('ab') as stream:
            process = subprocess.Popen(command, cwd=self.config['workspaceRoot'], stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        self.children[process.pid] = process
        value = {'pid': process.pid, 'processBirth': jobs._birth(process.pid), 'startedAtUnix': time.time()}
        self.state.update({'collectorChild': value, 'lastCollectorLaunchAt': value['startedAtUnix']})
        self.event('collector_started', command=command, **value)

    def tick(self):
        if sha(self.config_path) != self.config_sha:
            raise ValueError('Static supervisor config changed while running')
        cycle = self.state['cycle']
        stages = cycle['stages']
        if (self.directory / 'stop-request.json').exists():
            active = self.state.get('postStageChild')
            if active:
                self.stop_post_stage(active, 'local_supervisor_stop_request')
            if self.state['stageIndex'] < len(stages):
                self.cooperative_stop(stages[self.state['stageIndex']])
            self.update('stop_requested', reason='local_supervisor_stop_request')
            return False
        if self.post_stage_tick():
            return True
        if self.state['stageIndex'] >= len(stages):
            if not self.state.get('cycleAccounted'):
                self.state['cyclesCompleted'] += 1
                self.state['cycleAccounted'] = True
                self.event('cycle_completed', cycleId=cycle['cycleId'], cyclesCompleted=self.state['cyclesCompleted'])
            self.state['conversationCertification'] = 'rejected' if self.state.get('cycleFailed') else self.review_receipt(cycle)
            if self.config.get('maxCycles') is not None and self.state['cyclesCompleted'] >= self.config['maxCycles']:
                self.update('cycle_limit_reached')
                return False
            if self.next_cycle():
                self.state.pop('cycleAccounted', None)
                self.state.pop('cycleFailed', None)
                self.update('cycle_ready')
                return True
            # Publish the collector's required idle gate before creating its child.
            # Do not overwrite a later blocked_resources state from the free-space guard.
            self.update('waiting_for_data', reason='no_new_approved_corpus_and_config', trainingActive=False)
            if self.collect_if_configured() is False:
                return True
            return True
        stage = stages[self.state['stageIndex']]
        run_dir = path_for(self.config, stage['runDir'])
        native = jobs.status(run_dir)
        # Live PID+birth wins over recorded terminal status: export may still be active.
        if native.get('processAlive'):
            self.state['worker'] = native
            identity = self.cycle_key(stage) + ':' + str(native['pid']) + ':' + native['processBirth']
            if identity not in self.state.get('verifiedNativeIdentities', []):
                if self.cycle_key(stage) not in self.state.get('adoptedStageKeys', []) and not self.state.get('launch'):
                    if stage.get('adoptPid') is not None and stage['adoptPid'] != native['pid']:
                        raise ValueError('Initial worker PID differs from pinned adoption identity')
                    if stage.get('adoptProcessBirth') is not None and stage['adoptProcessBirth'] != native['processBirth']:
                        raise ValueError('Initial worker birth differs from pinned adoption identity')
                _, _, _, _, _, training = self.validate_stage(stage)
                self.guard_existing(stage, run_dir, training)
                self.state.setdefault('verifiedNativeIdentities', []).append(identity)
                self.state.setdefault('adoptedStageKeys', []).append(self.cycle_key(stage))
                self.event('worker_identity_verified', pid=native['pid'], processBirth=native['processBirth'], stageKey=self.cycle_key(stage))
            self.update('training' if native.get('status') in ('planned', 'running') else 'worker_finishing',
                worker=native, adopted=True, trainingActive=native.get('status') == 'running')
            return True
        launched = self.state.get('launch')
        if launched and launched.get('stageKey') == self.cycle_key(stage):
            if jobs._alive(launched['pid'], launched['processBirth']):
                self.update('worker_starting', launch=launched, trainingActive=False)
                return True
            child = self.children.get(launched['pid'])
            code = child.poll() if child else None
            if not launched.get('exitRecorded'):
                self.event('child_terminal', pid=launched['pid'], knownExitCode=code,
                    terminalState='exit_observed' if code is not None else 'dead_exit_code_unknown')
                launched['exitRecorded'] = True
            if native.get('status') == 'unplanned':
                self.update('blocked_error', reason='child_died_before_native_state', launch=launched, trainingActive=False)
                return True
        run_dir, corpus, config_path, model, parent, training = self.validate_stage(stage)
        if native['status'] == 'unplanned':
            self.launch(stage, run_dir, corpus, config_path, model, parent, resume=False)
            return True
        self.guard_existing(stage, run_dir, training)
        if native['status'] in TERMINAL:
            latest_path = run_dir / 'stage-latest.json'
            latest = read(latest_path) if latest_path.exists() else {}
            reasons = latest.get('cooperativeStopReasons', [])
            if set(reasons) & {'rss_ceiling', 'run_disk_ceiling', 'resource_monitor_failed'}:
                self.event('cycle_abandoned_resources', stageKey=self.cycle_key(stage), worker=native, reasons=reasons)
                self.abandon_cycle(stage, 'resource_stop')
                self.update('blocked_resources', worker=native, reason=','.join(reasons), trainingActive=False)
                return True
            if native['status'] == 'failed':
                self.event('cycle_abandoned_native_failure', stageKey=self.cycle_key(stage), worker=native)
                self.abandon_cycle(stage, 'native_failed')
                self.update('blocked_error', worker=native, reason='native_failed', trainingActive=False)
                return True
            capped = (native.get('step', 0) >= training['max_steps']
                or native.get('elapsedSeconds', 0) >= training['max_seconds']
                or native.get('reason') in ('validation_patience', 'total_time_budget', 'training_limit'))
            if capped or native['status'] == 'completed':
                self.complete_stage(stage, native)
                return True
            if native.get('reason') == 'invocation_time_budget' or native['status'] == 'stale':
                if not native.get('lastCheckpoint'):
                    raise ValueError('Worker ended without a resumable optimizer checkpoint')
                if native['status'] == 'stale':
                    key = self.cycle_key(stage)
                    attempts = self.state['crashRestarts'].get(key, 0)
                    if attempts >= self.config.get('maxCrashRestartsPerStage', 2):
                        self.update('blocked_error', worker=native, reason='crash_recovery_limit', trainingActive=False)
                        return True
                    self.state['crashRestarts'][key] = attempts + 1
                self.launch(stage, run_dir, corpus, config_path, model, parent, resume=True)
                return True
            self.update('paused_native', worker=native, reason='stop_is_not_an_invocation_budget_pause', trainingActive=False)
            return True
        self.update('blocked_error', worker=native, reason='unknown_native_state', trainingActive=False)
        return True

    def run(self):
        self.event('supervisor_started', pid=os.getpid(), processBirth=jobs._birth(os.getpid()), configSha256=self.config_sha)
        while True:
            try:
                keep_running = self.tick()
            except FileNotFoundError as error:
                self.update('waiting_for_plan', reason=str(error), trainingActive=False)
                keep_running = True
            except Exception as error:
                self.update('blocked_error', reason=type(error).__name__ + ': ' + str(error), trainingActive=False)
                keep_running = True
            if not keep_running:
                self.event('supervisor_exiting', operationalState=self.state['operationalState'])
                return
            time.sleep(self.config.get('pollSeconds', 10))


def supervisor_status(config):
    directory = path_for(config, config['stateDir'])
    state = read(directory / 'state.json') if (directory / 'state.json').exists() else {'operationalState': 'not_started'}
    instance = read(directory / 'instance.json') if (directory / 'instance.json').exists() else {}
    pid, birth = instance.get('pid'), instance.get('processBirth')
    alive = jobs._alive(pid, birth)
    worker = state.get('worker', {})
    cycle = state.get('cycle', {})
    index = state.get('stageIndex', 0)
    if index < len(cycle.get('stages', [])):
        worker = jobs.status(path_for(config, cycle['stages'][index]['runDir']))
    else:
        worker = {**worker, 'processAlive': jobs._alive(worker.get('pid'), worker.get('processBirth'))}
    return {**state, 'worker': worker, 'nativeWorkerAlive': bool(worker.get('processAlive')),
        'trainingActive': bool(worker.get('processAlive') and worker.get('status') == 'running'),
        'generationChildAlive': jobs._alive(state.get('postStageChild', {}).get('pid'), state.get('postStageChild', {}).get('processBirth')),
        'supervisorAlive': alive, 'supervisorPid': pid, 'supervisorBirth': birth,
        'recordedOperationalState': state.get('operationalState'),
        'operationalState': state.get('operationalState') if alive else 'supervisor_not_alive',
        'environmentLifetimeGuaranteed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'run', 'status', 'stop'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config, config_path, config_sha = load_config(args.config)
    directory = path_for(config, config['stateDir'])
    if args.command == 'status':
        print(json.dumps(supervisor_status(config), ensure_ascii=False, indent=2)); return
    if args.command == 'stop':
        atomic(directory / 'stop-request.json', {'requestedAtUtc': now(), 'reason': 'local_cli_stop'})
        print(json.dumps({'stopRequested': True, 'status': supervisor_status(config)}, ensure_ascii=False)); return
    if args.command == 'start':
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock(Path(config['workspaceRoot']) / 'private', filename='.continual-training.workspace.lock'):
            status = supervisor_status(config)
            if status['supervisorAlive']:
                raise RuntimeError('Supervisor already alive; existing process preserved')
            with (directory / 'supervisor.log').open('ab') as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run', '--config', str(config_path)],
                    cwd=config['workspaceRoot'], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
            instance = {'pid': process.pid, 'processBirth': jobs._birth(process.pid), 'startedAtUtc': now(), 'configSha256': config_sha}
            atomic(directory / 'instance.json', instance)
        print(json.dumps({'supervisorStarted': instance, 'environmentLifetimeGuaranteed': False}, ensure_ascii=False)); return
    with lock(Path(config['workspaceRoot']) / 'private', retry_seconds=10, filename='.continual-training.workspace.lock'):
        atomic(directory / 'instance.json', {'pid': os.getpid(), 'processBirth': jobs._birth(os.getpid()), 'startedAtUtc': now(), 'configSha256': config_sha})
        Supervisor(config, config_path, config_sha).run()


if __name__ == '__main__':
    main()

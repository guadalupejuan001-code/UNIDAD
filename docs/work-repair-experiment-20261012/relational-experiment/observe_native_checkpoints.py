#!/usr/bin/env python3
"""Copy stable native checkpoints without loading tensors or altering their runs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

MEMBERS = ('manifest.json', 'state.json', 'tokenizer.json', 'weights.safetensors', 'resume.pt')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fingerprint(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Expected a regular, non-symlink file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    after = path.lstat()
    if (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing: ' + str(path))
    return {'sha256': digest.hexdigest(), 'bytes': value.st_size}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def snapshot(path):
    before = path.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError('Expected a non-symlink directory: ' + str(path))
    if set(p.name for p in path.iterdir()) != set(MEMBERS):
        raise ValueError('Native checkpoint is not yet a complete five-file directory')
    files = {name: fingerprint(path / name) for name in MEMBERS}
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_mtime_ns):
        raise ValueError('Checkpoint directory changed during hashing')
    manifest = read(path / 'manifest.json')
    if manifest.get('schemaVersion') != 1 or set(manifest['files']) != set(MEMBERS) - {'manifest.json'}:
        raise ValueError('Unexpected native checkpoint manifest')
    for name, expected in manifest['files'].items():
        if files[name]['sha256'] != expected:
            raise ValueError('Native manifest SHA-256 mismatch: ' + name)
    return files


def preserve(stage, arm, source, design_sha):
    archive = stage / 'all-observed-checkpoints' / arm
    receipts = stage / 'checkpoint-preservation-receipts' / arm
    archive.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    destination = archive / source.name
    receipt_path = receipts / (source.name + '.json')
    if destination.exists():
        receipt = read(receipt_path)
        return receipt, False
    first = snapshot(source)
    temporary = Path(tempfile.mkdtemp(prefix='.copy-' + source.name + '-', dir=archive))
    try:
        for name in MEMBERS:
            with (source / name).open('rb') as original, (temporary / name).open('xb') as copied:
                shutil.copyfileobj(original, copied, 1024 * 1024)
                copied.flush()
                os.fsync(copied.fileno())
        second = snapshot(source)
        copied_files = snapshot(temporary)
        if first != second or first != copied_files:
            raise ValueError('Source before/source after/copy SHA-256 or size mismatch')
        state = read(temporary / 'state.json')
        receipt = {
            'schemaVersion': 1, 'preservedAtUtc': now(), 'arm': arm,
            'checkpoint': source.name, 'step': state['step'],
            'designSha256': design_sha, 'source': str(source), 'destination': str(destination),
            'files': copied_files, 'sourceBefore': first, 'sourceAfter': second,
            'allThreeHashPassesEqual': True, 'nativeManifestVerified': True,
            'sourceUnmodified': True, 'stdlibOnly': True, 'tensorsDeserialized': False,
        }
        atomic_json(receipt_path, receipt)
        os.rename(temporary, destination)
        return receipt, True
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def observe(stage, design_sha):
    actual = fingerprint(stage / 'design.json')['sha256']
    if actual != design_sha:
        raise ValueError('Frozen experiment design changed')
    design = read(stage / 'design.json')
    pins = {path: fingerprint(Path(path))['sha256'] == expected
            for path, expected in design['frozenFilesSha256'].items()}
    if not all(pins.values()):
        raise ValueError('A frozen experiment input changed')
    summary_path = stage / 'checkpoint-preservation-summary.json'
    previous = read(summary_path) if summary_path.exists() else {}
    records = {(item['arm'], item['checkpoint']): item for item in previous.get('checkpoints', [])}
    failures = previous.get('retryableObservations', [])
    added = []
    statuses = {}
    completions = {}
    for arm in design['arms']:
        checkpoint_root = stage / arm / 'run' / 'checkpoints'
        if checkpoint_root.exists():
            for checkpoint in sorted(checkpoint_root.iterdir()):
                if checkpoint.name.startswith('.') or not checkpoint.is_dir():
                    continue
                try:
                    record, created = preserve(stage, arm, checkpoint, design_sha)
                    records[(arm, record['checkpoint'])] = record
                    if created:
                        added.append({'arm': arm, 'checkpoint': record['checkpoint'], 'step': record['step']})
                except (OSError, ValueError, KeyError) as error:
                    failures.append({'atUtc': now(), 'arm': arm, 'checkpoint': checkpoint.name,
                                     'error': str(error), 'retryable': True})
        for filename, bucket in (('run/status.json', statuses), ('completion.json', completions)):
            path = stage / arm / filename
            try:
                bucket[arm] = read(path) if path.exists() else None
            except (OSError, ValueError):
                # The native launcher publishes completion via write_text. An
                # unfinished write is not terminal evidence and is retried.
                bucket[arm] = None
    returned = all(type(value) is dict and value.get('status') == 'completed'
                   and type(value.get('nativeStatus')) is dict
                   and value['nativeStatus'].get('status') in ('completed', 'stopped')
                   for value in completions.values())
    summary = {
        'schemaVersion': 1, 'checkedAtUtc': now(), 'designSha256': design_sha,
        'frozenInputHashesMatch': True, 'stdlibOnly': True, 'sourceStatesUnmodified': True,
        'checkpoints': list(records.values()), 'retryableObservations': failures,
        'nativeStatuses': statuses, 'armCompletions': completions,
        'allArmsReturned': returned,
        'observationMode': 'Read native checkpoint files; never start, resume, stop or deserialize training.',
    }
    atomic_json(summary_path, summary)
    return summary, added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--design-sha256', required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=1800)
    args = parser.parse_args()
    stage = Path(args.stage).resolve()
    os.umask(0o077)
    start = time.monotonic()
    while True:
        summary, added = observe(stage, args.design_sha256)
        for record in added:
            print(json.dumps({'event': 'checkpoint_preserved', **record}, ensure_ascii=False), flush=True)
        if args.once or summary['allArmsReturned'] or time.monotonic() - start >= args.max_seconds:
            for record in summary['checkpoints']:
                if snapshot(Path(record['destination'])) != record['files']:
                    raise ValueError('A preserved checkpoint changed since publication')
            summary['allPreservedHashesReverifiedAtReturn'] = True
            atomic_json(stage / 'checkpoint-preservation-summary.json', summary)
            print(json.dumps({'event': 'observer_returned', 'allArmsReturned': summary['allArmsReturned'],
                              'preservedCheckpoints': len(summary['checkpoints']),
                              'elapsedSeconds': round(time.monotonic() - start, 3)}, ensure_ascii=False), flush=True)
            return
        time.sleep(2)


if __name__ == '__main__':
    main()

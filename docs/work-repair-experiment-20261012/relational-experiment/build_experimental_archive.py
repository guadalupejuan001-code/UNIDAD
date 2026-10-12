#!/usr/bin/env python3
"""Package own relational-comparison files and verify a full restore; no ML calls."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Not a regular non-symlink file: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    after = path.lstat()
    if (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during packaging: ' + str(path))
    return {'bytes': value.st_size, 'sha256': h.hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--design-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--extra-evidence', action='append', default=[])
    args = parser.parse_args()
    os.umask(0o077)
    root, stage = Path(args.root).resolve(), Path(args.stage).resolve()
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Refusing to overwrite an existing archive')
    if output.name == 'Unidad-Pesos-Diagnostico.zip':
        raise ValueError('This new trial is distinct from the user-supplied earlier archive')
    if digest(stage / 'design.json')['sha256'] != args.design_sha256:
        raise ValueError('Frozen design SHA-256 mismatch')
    design = read(stage / 'design.json')
    frozen = {path: digest(Path(path))['sha256'] == expected
              for path, expected in design['frozenFilesSha256'].items()}
    if not all(frozen.values()):
        raise ValueError('Frozen experiment input changed')
    summary = read(stage / 'checkpoint-preservation-summary.json')
    if not summary.get('allArmsReturned') or not summary.get('allPreservedHashesReverifiedAtReturn'):
        raise ValueError('Both arms and checkpoint observer must have returned before packaging')
    logical = {}
    pins = {}

    def add(path, name=None, expected=None):
        path = Path(path)
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            return
        name = name or path.relative_to(root).as_posix()
        pure = PurePosixPath(name)
        if pure.is_absolute() or pure.as_posix() != name or '..' in pure.parts or '\\' in name:
            raise ValueError('Unsafe archive logical name: ' + name)
        if any(token in name.lower() for token in ('prospective-basic-final', 'reserved-case', 'private-lexical-audit')):
            raise ValueError('A reserved evaluation body must not be archived: ' + name)
        if name in logical and logical[name] != path:
            raise ValueError('Conflicting logical file: ' + name)
        if not path.is_file() or path.is_symlink():
            raise ValueError('Expected a regular source file: ' + str(path))
        logical[name] = path
        if expected is not None:
            pins[name] = expected

    for path in sorted((root / 'src').rglob('*')):
        if path.is_file() and not any(part.endswith('.egg-info') for part in path.parts):
            add(path)
    add(root / 'pyproject.toml')
    add(root / 'README.md', 'source-metadata/README.original.md')
    for name in ('examples/run_basic_acceptance.py', 'tests/test_basic_acceptance.py'):
        add(root / name)

    model_ids = {}
    model_directories = {'c7_reference': Path(design['initialModel'])}
    for arm in design['arms']:
        completion = read(stage / arm / 'completion.json')
        status = read(stage / arm / 'run' / 'status.json')
        if completion.get('status') != 'completed' or status.get('status') not in ('completed', 'stopped'):
            raise ValueError('Native arm is not terminal: ' + arm)
        for selection in ('endpoint', 'validation-selected'):
            model_directories[arm + '_' + selection] = stage / arm / selection
    for label, model in model_directories.items():
        manifest = read(model / 'manifest.json')
        if manifest['training'].get('trainedFrom') not in ('random_weights', 'own_checkpoint'):
            raise ValueError('Only own models may be attributed to this archive')
        model_ids[label] = manifest['modelId']
        for name, expected in manifest['files'].items():
            add(model / name, expected=expected)
        add(model / 'manifest.json')

    # Preserve native five-file states for every observed checkpoint, including
    # those no longer pointed to by the run's last/best pointers.
    for checkpoint in summary['checkpoints']:
        for name, expected in checkpoint['files'].items():
            add(Path(checkpoint['destination']) / name, expected=expected)
            native_name = (stage.relative_to(root) / checkpoint['arm'] / 'run' / 'checkpoints'
                           / checkpoint['checkpoint'] / name).as_posix()
            add(Path(checkpoint['destination']) / name, native_name, expected=expected)

    for path in sorted(stage.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(stage)
        if '__pycache__' in relative.parts or path.suffix == '.pyc':
            continue
        if any(part.startswith('.') for part in relative.parts):
            continue
        # These exact bytes are already included in the observer's preserved
        # five-file copies. Keep native metadata/logs, not redundant states.
        if len(relative.parts) >= 3 and relative.parts[1:3] == ('run', 'checkpoints'):
            continue
        if relative.parts[0] == 'archive-staging' or path.name.endswith('.zip'):
            continue
        add(path)

    for filename, expected_sha in design['frozenFilesSha256'].items():
        path = Path(filename)
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else (
            'external-own-evidence/' + path.relative_to('/workspace').as_posix())
        add(path, name, {'bytes': path.stat().st_size, 'sha256': expected_sha})
        if path.is_relative_to('/workspace/ia-propia'):
            # Restore the known development case at its usual project-relative
            # path as well, while keeping its original-worktree provenance.
            add(path, path.relative_to('/workspace/ia-propia').as_posix(),
                {'bytes': path.stat().st_size, 'sha256': expected_sha})
    for folder in args.extra_evidence:
        path = Path(folder).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Extra evidence must be inside the repair worktree')
        if path.is_dir():
            for member in sorted(path.rglob('*')):
                if member.is_file() and not any(part.startswith('.') for part in member.relative_to(path).parts):
                    add(member)
        else:
            add(path)

    records = []
    objects = {}
    for name, path in sorted(logical.items()):
        info = digest(path)
        if name in pins and info != pins[name]:
            raise ValueError('Pinned bytes/SHA-256 mismatch: ' + name)
        records.append({'path': name, **info, 'mode': 0o600})
        if info['sha256'] in objects:
            if objects[info['sha256']]['bytes'] != info['bytes']:
                raise ValueError('Inconsistent identical-hash object sizes')
        else:
            objects[info['sha256']] = {'path': path, 'bytes': info['bytes']}

    manifest = {
        'schemaVersion': 1, 'format': 'sha256_objects_full_files_v1',
        'createdAtUtc': now(), 'status': 'EXPERIMENTAL_NOT_ACCEPTED',
        'designSha256': args.design_sha256, 'modelIds': model_ids,
        'logicalFiles': len(records), 'uniqueObjects': len(objects),
        'files': records, 'allObservedNativeCheckpoints': len(summary['checkpoints']),
        'ownWeightsOnly': True, 'trainingPerformedByPackager': False,
        'inferencePerformedByPackager': False, 'automaticContinuation': False,
        'reviewKind': 'AI reviews of the project; not an independent human evaluation',
        'priorUserArchive': {'name': 'Unidad-Pesos-Diagnostico.zip',
                             'relationship': 'Distinct archive; not claimed restored or replaced'},
    }
    stage_path = stage.relative_to(root).as_posix()
    readme = '''# Unidad: contraste relacional propio, archivo experimental

Este ZIP conserva C7, ambos brazos originales de la comparación acotada, sus
artefactos de inferencia, todos los checkpoints nativos completos observados,
corpus y procedencia, código, decisiones congeladas y respuestas reales disponibles.
No convierte ningún candidato en un modelo aceptado ni acredita conversación
básica o competitiva. Las revisiones incluidas son de IA del proyecto.

## Restaurar los archivos

Extrae `restaurar.py` y `contenido.json` o extrae todo el ZIP en una carpeta
temporal. Luego ejecuta el restaurador contra el ZIP y un directorio nuevo:

```bash
python restaurar.py Unidad-Contraste-Relacional-Experimental.zip unidad-restaurada
```

El restaurador reconstruye archivos completos desde objetos SHA-256, verifica
tamaños y hashes y rechaza un destino existente. No entrena ni carga un modelo.
`contenido-restaurado.json` conserva el resultado de la verificación.

## Conversar con la referencia conservada

Desde la raíz restaurada, con Python 3.11+ y las dependencias de `pyproject.toml`:

```bash
python -m pip install -e '.[conversation]'
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-reference-c7 --max-new-tokens 160 --temperature 0 --seed 42
```

C7 sigue siendo una referencia experimental que falla la aceptación. Para explorar
los brazos de este ensayo, cambia `--model` por `%STAGE%/original/endpoint` o
`%STAGE%/relational/endpoint`. `/salir` termina y `/reiniciar` reinicia el historial.
Las respuestas proceden del motor y pesos propios; no se incluye un servicio
generador externo ni una sustitución por respuestas de demostración.

## Estados nativos conservados

`%STAGE%/all-observed-checkpoints/` conserva los cinco miembros completos de cada
instantánea: manifest, state, tokenizer, weights y resume. `resume.pt` contiene
optimizador, programador y RNG de Torch; `state.json` conserva el RNG de Python,
orden, cursor, configuración y contadores. Se incluyen recibos de copia y hashes.
Los archivos se conservan como bytes; el empaquetador no deserializó pickle.

Ambos brazos tienen un límite congelado de 128 actualizaciones; el límite ya
alcanzado no se amplía al restaurar ni al ejecutar el restaurador. Los presupuestos
anteriores están cerrados y no hay continuación automática. El estado matemático
BF16 requiere el mismo entorno Torch/CPU e hilos para una reanudación exacta.
Las rutas absolutas históricas del diseño/estado no se modificaron; al continuar
en otra máquina habría que proporcionar explícitamente el corpus restaurado y
decidir una etapa nueva o un cambio autorizado de presupuesto, distinguiendo
conservación del optimizador de inicializar pesos con un optimizador nuevo.

Este ZIP tiene un nombre propio y no reemplaza `Unidad-Pesos-Diagnostico.zip`.
La creación y la restauración verificadas de este archivo no afirman que se haya
recibido o reconstruido aquel archivo anterior.
'''.replace('%STAGE%', stage_path)

    # The editable package requires README.md at its root. Preserve the original
    # separately and restore this archive's own documented entry point there.
    archive_staging = stage / 'archive-staging'
    archive_staging.mkdir(exist_ok=True)
    restored_readme = archive_staging / 'README.restored.md'
    restored_readme.write_text(readme, encoding='utf-8')
    add(restored_readme, 'README.md')
    info = digest(restored_readme)
    records.append({'path': 'README.md', **info, 'mode': 0o600})
    records.sort(key=lambda record: record['path'])
    objects.setdefault(info['sha256'], {'path': restored_readme, 'bytes': info['bytes']})
    manifest['files'] = records
    manifest['logicalFiles'] = len(records)
    manifest['uniqueObjects'] = len(objects)

    restore_source = (stage / 'restaurar.py').read_bytes()
    expected_members = {'contenido.json', 'restaurar.py', 'README.md'} | {
        'objects/sha256/' + key for key in objects}
    with output.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            archive.writestr('contenido.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
            archive.writestr('README.md', readme)
            archive.writestr('restaurar.py', restore_source)
            for key, obj in sorted(objects.items()):
                archive.write(obj['path'], 'objects/sha256/' + key)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(output) as archive:
        if set(archive.namelist()) != expected_members or archive.testzip() is not None:
            raise ValueError('Archive member list or CRC failure')
        for key, obj in objects.items():
            h, count = hashlib.sha256(), 0
            with archive.open('objects/sha256/' + key) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                    count += len(block)
            if h.hexdigest() != key or count != obj['bytes']:
                raise ValueError('Archived object SHA-256 or size mismatch')
    for record in records:
        if digest(logical[record['path']]) != {key: record[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Source changed during packaging: ' + record['path'])

    spec = importlib.util.spec_from_file_location('own_verified_archive_restorer', stage / 'restaurar.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = []
    for value in ('../a', '/a', 'a\\b', './a', 'a/../b', 'a:b'):
        try:
            module.safe_relative(value)
        except ValueError:
            tests.append({'path': value, 'rejected': True})
        else:
            raise ValueError('Unsafe path was not rejected: ' + value)
    test_root = Path(tempfile.mkdtemp(prefix='unidad-relational-archive-verify-', dir=output.parent))
    restored = module.restore(output, test_root / 'restaurado')
    for record in records:
        if digest(Path(restored['destination']) / record['path']) != {
                key: record[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Restored file differs: ' + record['path'])
    existing_destination_rejected = False
    try:
        module.restore(output, Path(restored['destination']))
    except ValueError:
        existing_destination_rejected = True
    if not existing_destination_rejected:
        raise ValueError('Existing restore destination was accepted')
    archive_info = digest(output)
    receipt = {
        'schemaVersion': 1, 'createdAtUtc': now(), 'archivePath': str(output),
        'archiveSha256': archive_info['sha256'], 'archiveBytes': archive_info['bytes'],
        'designSha256': args.design_sha256, 'modelIds': model_ids,
        'logicalFiles': len(records), 'uniqueObjects': len(objects),
        'logicalBytes': sum(record['bytes'] for record in records),
        'uniqueObjectBytes': sum(obj['bytes'] for obj in objects.values()),
        'allObservedCheckpoints': len(summary['checkpoints']),
        'testzipPassed': True, 'allArchivedObjectHashesMatch': True,
        'allSourceAndRestoredHashesMatch': True, 'restoreTest': restored,
        'safePathTests': tests, 'existingDestinationRejected': True,
        'status': 'EXPERIMENTAL_NOT_ACCEPTED', 'activeTrainingByPackager': False,
        'trainingPerformed': False, 'inferencePerformed': False,
        'oldUserNamedArchiveUntouched': True, 'sourceStatesUnmodified': True,
    }
    write(output.with_suffix('.receipt.json'), receipt)
    output.with_suffix('.zip.sha256').write_text(archive_info['sha256'] + '  ' + output.name + '\n')
    write(stage / 'archive-receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

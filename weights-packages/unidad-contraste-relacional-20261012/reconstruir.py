"""Rebuild the original own-weight ZIP and verify every byte; never trains."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = '5978cf805f005ed3692117d9584ca7b515dc369458292289eaf42cb768619e88'
EXPECTED_BYTES = 339107857


def main():
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder / 'parts-manifest.json').read_text())
    if manifest['archiveSha256'] != EXPECTED_SHA256 or manifest['archiveBytes'] != EXPECTED_BYTES:
        raise ValueError('La identidad del paquete no coincide.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(manifest['archiveName']))
    output = parser.parse_args().output.resolve()
    temporary = output.with_name(output.name + '.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError('Elige una salida nueva; no se sobrescriben archivos.')
    total, full = 0, hashlib.sha256()
    try:
        with temporary.open('xb') as target:
            for index, part in enumerate(manifest['parts'], 1):
                expected_name = f"{manifest['archiveName']}.part{index:02d}"
                if part['name'] != expected_name or Path(part['name']).name != part['name']:
                    raise ValueError('Nombre u orden de parte inválido.')
                digest, count = hashlib.sha256(), 0
                with (folder / part['name']).open('rb') as source:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                        digest.update(block)
                        full.update(block)
                        count += len(block)
                if count != part['bytes'] or digest.hexdigest() != part['sha256']:
                    raise ValueError(f"La parte {index} está incompleta o alterada.")
                total += count
        if total != EXPECTED_BYTES or full.hexdigest() != EXPECTED_SHA256:
            raise ValueError('El ZIP final no coincide con el original.')
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f'ZIP reconstruido y verificado: {output}')
    print(f'SHA-256: {EXPECTED_SHA256}')


if __name__ == '__main__':
    main()

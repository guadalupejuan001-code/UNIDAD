# Dos tareas DEV nuevas: comparación congelada

Tratamiento y control completaron **0 de 2 tareas** cada uno. Los cuatro intentos
conservaron los originales, no ejecutaron herramientas y no crearon informes.
Conservar archivos por sí solo no cumple la tarea.

| Tarea | Tratamiento | Control |
| --- | --- | --- |
| Objetivo breve genérico | JSON con `cogo.csv`, fuera de la whitelist; `blocked_policy` | JSON con `co.txt`, fuera de la whitelist; `blocked_policy` |
| Orden explícito inspeccionar → leer → guardar | `read_file` con nombre repetitivo; salida truncada a 128 tokens y JSON inválido; `blocked_model` | JSON con `pau.txt`, fuera de la whitelist; `blocked_policy` |

El [freeze previo](task-freeze.json) fijó los candidatos propios BEST256, los
archivos, objetivos, familias, opciones y SHA de fuentes antes de generar.
Una revisión de IA del proyecto comprobó que los nombres, familias y objetivos
no coinciden con TRAIN/VALIDATION v1/v2 ni con el demo anterior. La revisión no
fue humana. La [exclusión futura](future-training-exclusions.json) mantiene estas
tareas como DEV después de examinarlas; sus respuestas no son targets positivos.

Se usaron Brain, API autenticada de localhost, ModelActionProposer y las tres
herramientas genéricas de SafeFileTools. Cada intento usó un CPU con nice 19,
temperatura 0, semilla 42, 128 tokens nuevos, ocho pasos, dos fallos, 90 segundos
de controlador y cero reparaciones de argumentos. No hubo fixture de respaldo,
sustitución de argumentos, filtros de decodificación ni entrenamiento.

El [resumen limitado](summary.json) y el [manifiesto](manifest.json) enlazan los
cuatro resultados reales con sus SHA. Cada `evidence.json` conserva las
propuestas originales, llamadas API, journal persistente, resultados de handlers,
hashes e informes observados. Los mismos bytes permanecen en el directorio
privado correspondiente. Estos resultados son desarrollo examinado; no son una
aceptación conversacional reservada ni prueban autonomía fiable.

El helper histórico está en
[compare_action_copy_candidates.py](../../../examples/compare_action_copy_candidates.py).
Su `--help` y compilación Python se comprobaron. El helper fija las rutas para
preservar esta corrida: **no se puede ejecutar directamente `freeze` o `run`
sobre un clon que ya contiene los resultados publicados**. `freeze` rechaza
carpetas existentes; `run` necesita el freeze privado ausente de Git y rechaza
salidas públicas existentes. No borres o muevas las pruebas publicadas para
sortear esas comprobaciones.

Para reproducir estas mismas tareas DEV, prepara un espacio temporal nuevo desde
la raíz del clon. Copia el helper y las fuentes completos, ambos modelos propios,
ambos corpus públicos y únicamente el freeze y las exclusiones. La copia privada
del freeze se crea con escritura exclusiva. Esto conserva el helper histórico y
sus pins sin copiar resultados anteriores a las salidas nuevas:

```sh
UNIDAD_DEV_REPLAY_ROOT=$(python3 - <<'PY'
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import signal
import tempfile

origin = Path.cwd()
os.nice(19)
os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
signal.alarm(120)
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
public_relative = Path("docs/learning-repair-20261013/action-copy-v2-development-tasks-v1")
frozen = json.loads((origin / public_relative / "task-freeze.json").read_bytes())
workspace = Path(tempfile.mkdtemp(prefix="unidad-dev-replay-"))
shutil.copytree(origin / "src", workspace / "src")
helper = workspace / "examples/compare_action_copy_candidates.py"
helper.parent.mkdir()
shutil.copyfile(origin / "examples/compare_action_copy_candidates.py", helper)
for pin in frozen["models"].values():
    destination = workspace / pin["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(origin / pin["path"], destination)
    for name, expected in pin["files"].items():
        assert (destination / name).stat().st_size == expected["bytes"]
        assert digest(destination / name) == expected["sha256"]
for relative in frozen["previousTrainingValidationCorpora"]:
    destination = workspace / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin / relative, destination)
public = workspace / public_relative
public.mkdir(parents=True, exist_ok=False)
for name in ("task-freeze.json", "future-training-exclusions.json"):
    with (public / name).open("xb") as stream:
        stream.write((origin / public_relative / name).read_bytes())
private = workspace / "private/learning-repair-20261013/action-copy-v2-development-tasks-v1"
private.mkdir(parents=True, exist_ok=False, mode=0o700)
with (private / "task-freeze.json").open("xb") as stream:
    stream.write((public / "task-freeze.json").read_bytes())
spec = importlib.util.spec_from_file_location("frozen_dev_replay", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.ROOT == workspace
assert module.read_freeze() == frozen
assert sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file()) < 512 * 1024**2
print(workspace)
PY
)
```

La preparación descrita se ejecutó y validó en un espacio temporal separado:
128.912.463 bytes copiados, pins de helper/fuentes/corpus/modelos comprobados,
pruebas públicas originales intactas y cero inferencias adicionales. El
[recibo de preparación](replay-preparation-verification.json) conserva esa
comprobación; su espacio temporal se destruyó al finalizar.

Para generar una repetición en el nuevo espacio preparado por el comando:

```sh
cd "$UNIDAD_DEV_REPLAY_ROOT"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  nice -n 19 python3 examples/compare_action_copy_candidates.py run
```

Las salidas nuevas quedan dentro de ese espacio. Los timestamps, UUID y hashes
de recibos pueden cambiar; los candidatos, datos y opciones están fijados.
Repetir tareas ya examinadas no las convierte en evaluación nueva ni autoriza
incorporarlas a TRAIN. Si algún pin falla en un clon posterior, conserva el
error y usa la revisión histórica con los archivos identificados por el freeze;
no sobrescribas el freeze ni las pruebas publicadas.

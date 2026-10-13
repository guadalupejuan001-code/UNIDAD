# Checkpoints nativos recuperables

El entrenamiento guarda cinco archivos nativos: `weights.safetensors`,
`resume.pt`, `tokenizer.json`, `state.json` y `manifest.json`. En la base propia
de 15.179.520 parámetros, el estado Adam/RNG ocupa 121.517.938 bytes y excede
el límite de un archivo de GitHub. Se distribuye en partes de como máximo
32 MiB, sin comprimir, interpretar pickle ni cambiar sus bytes.

`examples/package_native_checkpoint.py` verifica los hashes originales,
divide los archivos y publica un directorio nuevo de forma atómica. Su
manifiesto registra tamaños y SHA256 de cada archivo y cada parte. La
restauración verifica ese manifiesto, recompone los cinco archivos y conserva
el nombre original del checkpoint. El helper usa la biblioteca estándar de
Python y requiere Linux con `renameat2`; no importa PyTorch ni entrena.

## Respaldo intermedio entregado

Se capturó el checkpoint cerrado
`step-000003072-fa5cfc2a1cd64cf8b7555da7f7c4af86` en un staging independiente.
Los hashes coincidieron antes de copiar, en la copia y después de copiar.
El empaquetado verificó de nuevo la copia y las nueve partes resultantes.
El [recibo de entrega](foundation-15m-step-3072-delivery.json) conserva esa
evidencia. El paquete local está en
`checkpoints/learning-repair-20261013/foundation-15m/step-000003072-fa5cfc2a1cd64cf8b7555da7f7c4af86`.

| Archivo original | Bytes | Partes |
| --- | ---: | ---: |
| `weights.safetensors` | 60.727.960 | 2 |
| `resume.pt` | 121.517.938 | 4 |
| `tokenizer.json` | 49.502 | 1 |
| `state.json` | 110.339 | 1 |
| `manifest.json` | 361 | 1 |
| Total | 182.406.100 | 9 |

SHA256 del manifiesto del paquete:
`7c6534c70916993e291ab25fdc6a06806294f054c4b2d9dc5ae3b8db95ac2b86`.
El checkpoint describe 3.072 updates completos, una época completa y 6.576
ventanas de la siguiente. La contabilidad desde su estado y el índice congelado
registra **7.886.740 posiciones de entrenamiento** en 24.571 ventanas; el
[recibo exacto](foundation-15m-step-3072-target-exposure.json) conserva la fórmula
y los hashes. La pérdida de validación registrada es 3,7733838.

Es un respaldo intermedio de una corrida todavía abierta. No es una selección
por conversación, no sustituye C7 y no acredita que UNIDAD ya converse. La subida
de este paquete a GitHub se comprobará por separado. Durante esta entrega no
se ejecutaron pruebas de software, restauración ni otro entrenamiento.

## Reponer los bytes

Desde la raíz del repositorio, con un destino nuevo:

```bash
mkdir -p private/checkpoint-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/foundation-15m/step-000003072-fa5cfc2a1cd64cf8b7555da7f7c4af86 \
  private/checkpoint-recovery/checkpoints/step-000003072-fa5cfc2a1cd64cf8b7555da7f7c4af86
```

El comando recompone y comprueba los bytes. No crea una nueva etapa SFT, no
reinicia el optimizador y no arranca una corrida. Se niega a sobrescribir un
destino existente. El directorio del paquete debe contener sólo sus partes y
sus dos manifiestos; la documentación se guarda fuera de ese directorio.

La reanudación también necesita el corpus exacto con SHA256
`5f1089ee7e272ce9f717c8e044ccc64b7c003c9d9dcba41a466607453931be17`, el
[diseño de etapa](foundation-15m-stage-design.json), su código congelado y un
`status.json` que apunte al checkpoint. Si el corpus o el código no están
disponibles con esos hashes, todavía no hay una reanudación equivalente.
La caché mmap puede reconstruirse con el tokenizador restaurado; el entrenador
comprueba su identidad.

Una recuperación del proceso se realiza sólo cuando la corrida original y su
supervisor están detenidos. Para reponer el estado en un directorio nuevo en
el mismo checkout `/workspace/UNIDAD`, las instrucciones siguientes conservan
el diseño y registran un puntero detenido:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from ia_propia.conversation import training

run = Path("private/checkpoint-recovery")
name = "step-000003072-fa5cfc2a1cd64cf8b7555da7f7c4af86"
state = json.loads((run / "checkpoints" / name / "state.json").read_text())
design = Path("docs/learning-repair-20261013/foundation-15m-stage-design.json")
target = run / "stage-design.json"
if target.exists() or (run / "status.json").exists():
    raise FileExistsError("Use a new recovery run")
target.write_bytes(design.read_bytes())
training._status_write(run, state, "stopped", "restored_without_training")
PY
python3 examples/run_learning_stage.py \
  --corpus private/learning-repair-20261013/corpus/causal-spanish-v1/causal-spanish.train-validation.jsonl \
  --expected-corpus-sha256 5f1089ee7e272ce9f717c8e044ccc64b7c003c9d9dcba41a466607453931be17 \
  --run-dir private/checkpoint-recovery --resume \
  --stop-after-step 12000 --budget-seconds 3600 \
  --max-rss-mib 6144 --max-disk-mib 2048
```

La última orden sí reanuda entrenamiento real. Conserva los pasos, el tiempo
acumulado y el límite total de 7.200 segundos del checkpoint, además de Adam,
scheduler, orden de ventanas y RNG. Es una instrucción preparada para
recuperación, todavía no ejecutada en esta entrega; no se lanza mientras la
corrida original siga activa.

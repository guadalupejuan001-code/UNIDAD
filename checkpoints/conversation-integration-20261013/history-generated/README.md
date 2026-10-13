# Checkpoints propios de Unidad: history-generated

Este paquete conserva dos estados nativos completos: **selección por validación, paso 384**, y **último checkpoint, paso 512**. Incluye sus pesos FP32, tokenizador, configuración/estado, Adam, programador y RNG de Python/Torch. Los archivos de checkpoint se copiaron sin modificar sus bytes. No implica aceptación conversacional ni inicia entrenamiento.

[Identidades y hashes](checkpoint-package.json) identifica ambos punteros y el corpus requerido: `7ea1ebf926ebd810d622c53a7041684310a3b3b019e75e22b8cf1c5eacd214ff`. Los modelos de inferencia están en [models/conversation-integration-20261013/history-generated](../../../models/conversation-integration-20261013/history-generated/README.md). El archivo `resume.pt` pertenece al entrenador; la conversación usa el artefacto con safetensors.

Desde la raíz del repositorio, con las dependencias de conversación:

```bash
PYTHONPATH=src python docs/conversation-integration-20261013/training/restore_checkpoint.py --package checkpoints/conversation-integration-20261013/history-generated --run-dir runs/restored-history-generated --checkpoint best
```

Esta operación valida hashes y el estado de Adam/programador/RNG, copia los checkpoints a un directorio nuevo y señala el paso 384. No entrena ni ejecuta una generación. `--checkpoint endpoint` restaura el paso 512. Los dos checkpoints siguen disponibles. No se ejecutó una nueva reanudación durante esta publicación.

Para continuar después desde ese estado, una invocación explícita de la API nativa puede usar:

```python
from ia_propia.conversation import training

result = training.train(
    "examples/on-policy-history-v1/generated.train-validation.jsonl",
    "runs/restored-history-generated",
    resume=True,
    expected_corpus_sha256="7ea1ebf926ebd810d622c53a7041684310a3b3b019e75e22b8cf1c5eacd214ff",
    budget_seconds=60,
)
```

La reanudación conserva el optimizador y la configuración del checkpoint. El presupuesto de 60 segundos es sólo un ejemplo para una nueva invocación; no renueva presupuestos históricos ni significa que exista un proceso activo. El corpus exacto permite reconstruir el cache mmap excluido. Las rutas históricas guardadas en el estado no son las rutas públicas actuales; la API recibe las nuevas rutas explícitamente. El runner del piloto fija también rutas/identidades en su diseño y no admite directamente esta reubicación.

La continuidad numérica depende de mantener las versiones, la precisión BF16, los hilos y el entorno de cálculo guardados. Se conservaron y validaron los estados; no se afirma haber reproducido una continuación en otra máquina. Las evaluaciones reservadas permanecen fuera del corpus.

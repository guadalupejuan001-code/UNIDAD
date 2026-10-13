# Entrenamiento selectivo de la cabeza de copia

Unidad permite ajustar la cabeza de copia de un modelo propio conservando los
pesos del Transformer. Es una opción experimental para comparar aprendizaje y
retención; **congelar pesos no garantiza conservar las respuestas**, mejorar la
conversación ni sustituir una IA comercial.

## Política nativa

En el JSON de configuración del entrenamiento:

```json
{"trainable_parameters": "copy_head"}
```

El modelo inicial debe tener una cabeza de copia compatible. Sólo reciben
gradientes `copy_query.weight`, `copy_key.weight`, `copy_gate.weight` y
`copy_gate.bias`. En C7 son 16.641 parámetros; los otros 6.967.296 quedan
congelados. El optimizador conserva el orden de todos los parámetros, pero sólo
guarda momentos de los cuatro tensores entrenables. Los parámetros sin gradiente
no reciben el decaimiento de pesos de AdamW.

La opción `"all"` ajusta todo el modelo. Omitir la opción mantiene el camino
anterior, sin añadir el campo a los defaults o a los metadatos de checkpoints
antiguos. Una opción explícita guarda nombres y conteos en
`training.trainableParameters` del manifiesto; `copy_head` añade el SHA-256 de
los bytes, formas, tipos y nombres de los tensores congelados. El entrenador
comprueba ese recibo al guardar, cargar y exportar.

El entrenamiento sigue siendo el nativo de PyTorch, con supervisión de la última
respuesta y el motor conversacional compartido para evaluar. Este cambio no
añade un generador de respuestas ni modifica la arquitectura de C7. NERVA tiene
su integración separada, descrita en [nerva.md](nerva.md).

## Reanudar una corrida

`resume=True` mantiene la política guardada, pesos, optimizador, scheduler,
orden de lotes y RNG. Se exige el mismo corpus y configuración. Cambiar
`copy_head` por `all`, o añadir o quitar una opción explícita, requiere una
etapa nueva: no se acepta como reanudación del checkpoint anterior.

La CLI usa el subcomando `conversation resume` y recupera la configuración del
checkpoint. Para una etapa desde C7 con auxiliares de copia, indica también su `config.json` mediante
`--model-config`: el contrato existente exige declarar la cabeza al usar
`copy_supervision`. Las nuevas etapas conservan la procedencia propia y empiezan
con un optimizador nuevo; reanudar conserva además su estado.

## Reproducir el contraste acotado

El [runner público](../examples/experiment_copy_head_retention.py) recibe rutas
explícitas y no depende de un workspace concreto. Requiere instalar Unidad con
sus dependencias `conversation`, los cuatro archivos de C7 en
`models/conversation-reference-c7/`, el corpus existente
`examples/conversation-focus-curriculum.jsonl` y un archivo de casos de desarrollo
expuestos. Ese panel JSON usa `episodes`, cada episodio con `id` y `turns`, y cada
turno con `user`. No se usa un conjunto final reservado para diseñar este ensayo.

El panel expuesto utilizado se publica en
`docs/brain-nerva-20261013/development-cases.json`. Desde la raíz del repositorio:

```bash
python -m pip install -e '.[conversation]'
python examples/experiment_copy_head_retention.py prepare \
  --output private/copy-head-retention-nueva \
  --development-cases docs/brain-nerva-20261013/development-cases.json
```

`prepare` sólo congela los datos y el diseño; no carga pesos en un modelo ni
entrena. Conserva las líneas, IDs, grupos, objetivos y particiones originales de
432 registros TRAIN y 42 VALIDATION, en 144 y 14 familias. Excluye los 66 registros
TEST; rechaza coincidencias literales con mensajes del panel de desarrollo.
Con el panel de 16 turnos utilizado en el ensayo del 13 de octubre, el corpus
resultante tiene SHA-256
`7aad5655446aa4afb4adc61b6bc65cd437de33218cb572eea39086832fef2f6e`.

La acción siguiente **inicia una comparación nueva**; no reabre presupuestos ni
procesos históricos:

```bash
python examples/experiment_copy_head_retention.py run \
  --output private/copy-head-retention-nueva \
  --development-cases docs/brain-nerva-20261013/development-cases.json
```

También admite `--root /ruta/al/repositorio` desde otra carpeta. Los dos brazos
parten de C7 con la misma semilla y datos: `all`, seguido de `copy_head`. Cada uno
está limitado a 128 pasos y 600 segundos de presupuesto nativo, FP32, un hilo,
lote de cuatro, aprendizaje 0,0001 y warmup de 16 pasos. Validación y guardado
finales pueden añadir tiempo al límite del bucle. Los endpoints están fijados
antes de generar conversaciones; no se eligen por sus respuestas ni por menor
pérdida. Los recibos indican el paso realmente alcanzado si el tiempo impide
completar 128.

El runner conserva checkpoints locales, eventos, manifiestos y recibos de
integridad. No genera ni califica conversaciones, ejecuta TEST, cambia el modelo
predeterminado o promueve candidatos. La evaluación debe cargar el endpoint
exacto con `ConversationEngine` y conservar sus respuestas reales como historia.
Una pérdida menor con respuestas ideales no acredita conversación mejor.

La versión pública deriva del runner histórico; es un archivo distinto. El
original y sus hashes se preservan en
`docs/brain-nerva-20261013/original-runner.py`, junto a la evidencia del ensayo. La adaptación
pública se verificó con `--help`, compilación de sintaxis y `prepare` en un
directorio temporal: reprodujo los bytes del corpus y rechazó reemplazar inputs
congelados. Esa verificación no repitió el entrenamiento.

## Pruebas y límites

Las nueve [pruebas de política](../tests/test_conversation_trainable_policy.py)
comprueban reanudación exacta de dos más dos pasos frente a cuatro pasos,
FP32/BF16, bytes congelados bajo AdamW, estados inválidos, compatibilidad con
ausencia de la opción y exportación de procedencia propia. La comprobación
técnica completa de esta etapa aprobó 399 pruebas. Son fixtures y contratos de
software; no una calificación de inteligencia.

El contraste histórico completó 128 pasos por brazo y conservó la identidad de
C7. En el panel expuesto de 16 turnos, ambas revisiones de IA del proyecto
contaron una respuesta completa y dos fallos críticos para C7, y dos completas
y un fallo crítico para cada candidato. Ninguno completó un diálogo de varios
turnos. El brazo congelado no demostró superioridad conversacional sobre el
ajuste completo; ambos perdieron una respuesta antes parcial. C7 sigue siendo
referencia experimental y no se promovió un candidato por este panel.

Las revisiones pertenecen a IA del mismo proyecto, sin panel humano independiente.
La aceptación sigue requiriendo 36/40 respuestas completas, 8/10 diálogos
completos y cero fallos críticos o técnicos mediante una prueba nueva reservada.
El ensayo acotado no cumple ni reemplaza esa aceptación.

El candidato congelado se conserva por separado en
`models/conversation-copy-head-20261013/`, con su delta y reconstrucción verificable.
Cargar ese candidato experimental no cambia la referencia C7 ni inicia
entrenamiento. El informe de esta etapa es
`docs/work-brain-nerva-20261013.md`; sus recibos y revisiones preservan las respuestas
reales y la diferencia entre las etiquetas parciales de los revisores.

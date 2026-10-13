# Aprendizaje de patrones: resultado real

UNIDAD ya reconoce patrones mediante atención y capas neuronales aprendibles.
Esta entrega mejora el flujo para entrenar y medir relaciones conversacionales;
no añade reglas que fabriquen respuestas. El control confirma aprendizaje de
ejemplos conocidos, pero sigue sin demostrar conversación fiable con casos nuevos.

**C7 es un candidato anterior de UNIDAD, no una IA externa.** Tiene 6.983.937
parámetros y se conserva como referencia. La línea de este experimento nació
también desde pesos aleatorios propios, con una base de 15.179.520 parámetros,
preentrenamiento de español y ajuste conversacional. Su extensión de contexto
tiene 15.769.344 parámetros. Estas continuaciones reutilizan esa línea propia;
no redimensionan automáticamente los pesos de C7.

## Qué se integró y ejecutó

- [Preparador](../../../examples/prepare_conversation_pattern_curriculum.py):
  currículo ficticio de personas, correcciones, negación, incertidumbre, petición
  actual y distractores. Declaración sintética, método y CC0 incluidos.
- [Evaluador](../../../examples/run_conversation_pattern_probe.py): generación
  real con `ConversationEngine`, pares contrastantes, tokens, historial, límites
  y probabilidad de targets idealizados identificada aparte.
- [Integración con la cola](../../../examples/enqueue_conversation_pattern_stage.py):
  valida pesos propios, procedencia y permisos antes de publicar una etapa
  exclusiva. Conserva el supervisor y la corrida existentes.

El corpus contiene **48 TRAIN y 24 VALIDATION**, doce y seis familias,
respectivamente. Los nombres, valores y formulaciones de desarrollo se separan.
No contiene TEST. Hay 403 tokens supervisados TRAIN por pasada; aumentar las
pasadas no equivale a añadir conocimiento. Doce apariciones de respuestas
históricas erróneas quedan fuera de la pérdida. Todos los ejemplos caben en una
ventana, como máximo de 233 tokens. Los datos están en
[pattern-relations-v2](../../../examples/learning-repair-20261013/pattern-relations-v2/README.md).
V2 conserva el corpus inicial y alinea el criterio antes del entrenamiento;
la preparación anterior permanece archivada, sin editar su historia.

El padre cerrado es `c0ee5f24b2424eb652f3ca9bda5fcb8841d703b4f8f5efc4db23427a8ed123c4`.
Se ejecutaron **512 actualizaciones en 72,283 segundos**. El criterio nativo de
validación seleccionó **BEST del paso 128**, identificado como
`af515755f6eb176ed9968235231f24d8abdebca002f881a327ba69651bed1591`.
No se eligió un checkpoint mirando las respuestas del probe.

Se entrenaron todos los parámetros, con la última respuesta y EOS, mmap,
BF16 de CPU, batch lógico cuatro y microbatch dos. Los límites previos eran
512 actualizaciones, 600 segundos cooperativos, 6 GiB RAM y 1 GiB de corrida.
El pico del proceso, incluida la exportación, fue **1.039,281 MiB** y el pico
observado de archivos de corrida, **483,249 MiB**. Se usaron cuatro hilos CPU,
sin GPU ni servicios contratados. La estimación previa y los tiempos reales
se conservan por separado; las latencias del evaluador no son un benchmark
aislado de rendimiento.

## Qué aprendió y qué sigue fallando

Con el mismo motor, temperatura cero, semilla 42 y máximo 64 tokens:

| Medida | Padre | Candidato BEST 128 |
| --- | ---: | ---: |
| Coincidencia exacta con EOS, doce prefijos TRAIN | 0/12 | 11/12 |
| Pares TRAIN informativos con ambas respuestas exactas | 0/6 | 5/6 |
| Coincidencia exacta con EOS, doce prefijos de desarrollo | 0/12 | 0/12 |
| Respuestas completas de desarrollo según revisión semántica de IA | 0/12 | 1/12 |
| Diálogos completos de desarrollo con historia generada | 0/2 | 0/2 |
| Paradas por presupuesto de tokens en los 28 turnos | 5 | 0 |

Los prefijos utilizan historia ideal declarada. Los diálogos conservan cada
respuesta real desde un contexto inicial ficticio declarado; no se inyectan
targets para corregir la historia. La revisión permite paráfrasis y conserva
ambigüedades: «No me dio ningún horario» se consideró completa respecto a Zumira,
con el pronombre implícito marcado como incierto. «Delunao es viernes a las
trece» conserva la hora, pero se calificó parcial por la frase y nombre deformados.
El candidato registra tres fallos críticos en esos doce prefijos nuevos; los
criterios y cada juicio están en [la revisión semántica](candidate-semantic-review.json).
Son juicios de un agente de IA del proyecto, sin evaluación humana independiente.

El modelo aprende los ejemplos del control. Los casos nuevos muestran errores
de atribución: responde con el horario de Caelun al preguntar por Zumira o con
el objeto de Ulmera al preguntar por Dovian. No se encontraron copias literales
de nombres o valores completos TRAIN en esos fallos de desarrollo. Todavía no
se separaron los efectos de cambiar nombres, valores, formulación y orden de
mención. El resultado no identifica una causa única.

**No alcanzó el criterio diagnóstico previo**: diez TRAIN, ocho de desarrollo,
dos diálogos completos, mejora y cero fallos técnicos/críticos. Tampoco se
evaluó la aceptación final de 36/40, ocho de diez diálogos y cero fallos. La
reserva final sigue cerrada. El candidato se entrega como experimental y queda
fuera del padre automático del colector mediante `kind: pattern_diagnostic`.

La intervención usa SFT, coherente con la etapa supervisada de las
[fuentes originales consultadas](../../conversation-integration-20261013/methods-original-sources.md).
No incorpora PPO, GRPO, DPO, evaluación del modelo sobre sí mismo ni respuestas
de otra IA en inferencia. Es una comparación antes/después del mismo padre,
no un control con otro entrenamiento de igual presupuesto.

## Reproducir las respuestas

Desde la raíz del repositorio, con `pip install -e '.[conversation]'`, usa una
ruta nueva para no sobrescribir evidencia:

```bash
python examples/run_conversation_pattern_probe.py \
  --corpus examples/learning-repair-20261013/pattern-relations-v2/corpus.jsonl \
  --cases examples/learning-repair-20261013/pattern-relations-v2/development-cases.json \
  --manifest examples/learning-repair-20261013/pattern-relations-v2/manifest.json \
  --model-dir models/learning-repair-20261013/pattern-relations-v1/candidate-best \
  --output-dir private/pattern-candidate-development-new \
  --mode validation-prefix
```

Repite con `--mode train-prefix` y `--mode validation-history`, siempre con
otra ruta. Para el antes cambia el modelo por
`models/learning-repair-20261013/pattern-relations-v1/reference-wiki1`.
El evaluador impone tiempo, RAM y disco finitos, un hilo CPU y salidas exclusivas.
Sus tokens, textos, errores y recibos originales se publican en los seis
directorios `reference-*` y `candidate-*` de esta entrega. Los hashes de los
24 prefijos coinciden entre brazos; los turnos posteriores con historia
generada conservan las diferencias producidas por cada modelo.

La CLI existente carga estos mismos pesos:

```bash
unidad conversation converse \
  --model models/learning-repair-20261013/pattern-relations-v1/candidate-best \
  --temperature 0 --seed 42 --max-new-tokens 64
```

## Repetir el entrenamiento o integrarlo en una cola

La configuración exacta y los recibos están en [technical](technical/).
Una corrida nueva desde el padre reinicia el optimizador; no es reanudación
exacta ni reabre el presupuesto original. Ejecútala sólo en recursos libres:

```bash
python examples/run_learning_stage.py \
  --corpus examples/learning-repair-20261013/pattern-relations-v2/corpus.jsonl \
  --expected-corpus-sha256 c38db1b3f3464048145df46be4d23b048896d19952bd37417f8a208936b6fed2 \
  --run-dir private/pattern-learning-new-run \
  --training-config docs/learning-repair-20261013/pattern-relations-v2/technical/training-config.json \
  --parent-model models/learning-repair-20261013/pattern-relations-v1/reference-wiki1 \
  --stop-after-step 512 --budget-seconds 900 \
  --max-rss-mib 6144 --max-disk-mib 1024
```

Para incorporarlo a un supervisor existente, usa el preparador de cola con
`--supervisor-config CONFIG_EXISTENTE`, los mismos padre/corpus/manifiesto,
`--run-dir`, `--output-dir` y `--cycle-id` nuevos. `--dry-run` sólo verifica;
`--enqueue` publica una aprobación fijada al agente permitido por ese
supervisor. La cola espera al trabajador actual y no modifica su presupuesto.
No copies PID o estado de la sesión histórica a una instalación nueva.

Los pesos y tokenizador completos están en la carpeta de modelos indicada;
también se entrega el checkpoint nativo BEST, con optimizador, repartido en
partes verificables. Su restauración y alcance están en el recibo técnico.
Conservar un checkpoint no significa que haya un proceso activo.

## Evidencia y siguiente diagnóstico

[Comparación exacta](exact-comparison.json),
[comparación semántica](semantic-comparison.json),
[freeze previo](pretraining-freeze.json) y
[copias byte a byte](raw-publication-integrity.json) conservan entradas y salidas.
El [diseño previo](../pattern-learning.md) se mantiene tal como estaba antes
de ejecutar, incluido su estado entonces pendiente; este documento informa
el resultado posterior. Las 21 pruebas nuevas pasan, incluida una parada real
del evaluador; sus logs están en esta entrega. Eso comprueba software, no
conversación competente.

El siguiente diagnóstico propuesto separa nombres nuevos, valores nuevos,
formulación y orden de mención con el modelo fijo. Está pendiente de ejecución;
no se cuenta como aprendizaje realizado ni se repite indefinidamente este
ajuste fallido. El supervisor anterior conserva su colector y límites.

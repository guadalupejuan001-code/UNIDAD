# Aprender español antes del ajuste conversacional

La versión 0.6.0 permite dos objetivos explícitos con el mismo Transformer,
entrenador reanudable, formato de pesos y `ConversationEngine`:

| Objetivo | Corpus | Tokens que enseñan |
| --- | --- | --- |
| `causal_lm` | Documentos con `text` | Cada token de texto y EOS, predichos desde el contexto anterior. |
| `assistant_last_reply` | Diálogos con `messages` | La última respuesta del asistente y su EOS. |

Omitir `objective` conserva el comportamiento y las reanudaciones anteriores.
Aprender documentos enseña lenguaje; el posterior ajuste conversacional enseña
a responder al usuario. La calidad debe comprobarse con respuestas generadas.

## Documentos y particiones

Cada línea del corpus TEXT es un objeto con estas cinco claves exactas:

```json
{"id":"documento-001-fragmento-01","groupId":"documento-001","split":"train","source":{"dataset":"ejemplos-del-proyecto","kind":"authored-synthetic","language":"es"},"text":"Un texto ficticio escrito para practicar español."}
```

`text` conserva sus bytes UTF-8, espacios y saltos de línea. Admite hasta 65.536
bytes, debe ser no vacío y no puede contener tokens de control reservados.
La procedencia no se introduce como texto de entrenamiento. El cargador rechaza
IDs repetidos, familias que crucen particiones y duplicados exactos de texto
incluso dentro de una misma partición. Los fragmentos de un documento conservan
el mismo `groupId` y `split`; no se distribuyen individualmente entre train,
validation y test. Debe haber ejemplos utilizables en las tres particiones.

El tokenizador BPE se ajusta exclusivamente con los documentos de `train`, sin
envolturas de conversación. El entrenador codifica cada documento como
`BOS + texto + EOS`, desplaza entradas y objetivos antes de formar ventanas y
supervisa cada posición una vez. `sliding_context` conserva contexto solapado
y enmascara los objetivos ya enseñados. El padding tampoco enseña tokens.

## Configuración nativa mayor

Este `model.json` da 6.967.296 parámetros cuando BPE alcanza 8.192 tokens:

```json
{
  "vocab_size": 8192,
  "d_model": 256,
  "n_layers": 6,
  "n_heads": 8,
  "context_length": 512,
  "ffn_multiplier": 4,
  "dropout": 0.0,
  "seed": 42
}
```

El modelo arranca con pesos aleatorios propios. No amplía ni rellena los pesos
de las versiones pequeñas. El artefacto guarda el vocabulario y el número de
parámetros realmente obtenidos. La implementación actual entrena en CPU.

Un `pretrain.json` de ejemplo:

```json
{
  "objective": "causal_lm",
  "batch_size": 4,
  "learning_rate": 0.0005,
  "warmup_steps": 200,
  "max_steps": 6000,
  "max_seconds": 7200,
  "validation_every": 500,
  "checkpoint_every": 500,
  "patience": 10,
  "torch_threads": 4,
  "window_policy": "contiguous_nonoverlapping",
  "loss_reduction": "token_mean",
  "projection_mode": "supervised"
}
```

```bash
unidad conversation start --corpus private/textos-es.jsonl \
  --run-dir private/pretrain --model-config model.json \
  --training-config pretrain.json
unidad conversation status --run-dir private/pretrain
unidad conversation snapshot --run-dir private/pretrain \
  --output private/modelo-lenguaje
```

No se evalúa test al entrenar ni al exportar una instantánea. `snapshot` selecciona
por validación y declara `trainingObjective="causal_lm"`,
`assistantTargets="all_text_tokens"` y `testEvaluated=false`. Puede exportarse
durante el entrenamiento; el snapshot queda separado del checkpoint reanudable.

## Ajustar conversación con el mismo modelo

Después de la etapa de lenguaje, un JSON de entrenamiento conversacional puede
usar `objective="assistant_last_reply"`, `loss_reduction="sequence_mean"` y
`projection_mode="supervised"`. El corpus conserva el formato de diálogos
explicado en la [guía de conversación](conversation.md).

```bash
unidad conversation start --corpus private/conversaciones-es.jsonl \
  --run-dir private/ajuste-conversacional \
  --initial-model private/modelo-lenguaje --training-config sft.json
```

Esta es una nueva etapa: conserva exactamente los pesos y el tokenizador del
artefacto propio y reinicia Adam y el plan de tasa de aprendizaje. `resume`, en
cambio, restaura también Adam, orden pendiente, cursores y estados aleatorios.
El objetivo, corpus, tokenizador y configuración guardados deben coincidir.
Cambiar el objetivo exige una nueva etapa, no una reanudación.

Los corpus públicos se conservan con fuente, revisión, licencia y particiones.
Los ejemplos ficticios del proyecto se identifican como `authored-synthetic`.
El historial personal del creador no se usa como entrenamiento. Una prueba de
desarrollo repetida sirve para comparar cambios; no sustituye una evaluación
final ni demuestra conversación general o programación autónoma.

## Preparar textos públicos y diálogos

El [preparador de corpus públicos](../examples/prepare_public_spanish_language.py)
recibe el corpus español canónico de OASST y su manifiesto de importación, la
mezcla conversacional anterior y los diálogos ficticios nuevos. Exige los hashes
SHA-256 de estos dos últimos inputs. Sustituye el subconjunto humano anterior
por su fuente completa y conserva los ejemplos propios y sus particiones.
Descarga textos de la revisión fija `20231101.es` de Wikipedia, conserva la
atribución y no carga modelos ni ajusta un tokenizador.

```bash
python examples/prepare_public_spanish_language.py --help
```

El directorio de salida debe ser nuevo. Produce `text.jsonl`, `sft.jsonl` y un
manifiesto con las fuentes y hashes. Elimina familias OASST implicadas en
duplicados de texto o diálogos entre particiones; limita validación y test de
texto por familias completas antes de entrenar. No selecciona datos usando
respuestas o pérdidas de un modelo. Los duplicados semánticos y los errores
factuales de las fuentes requieren evaluaciones distintas.

El [registro de preparación](unidad-spanish-language-base-v1.json) documenta
los corpus utilizados: 17.906 textos y 48.862 diálogos de entrenamiento. El
recuento del BPE propio distingue palabras, tokens y objetivos EOS; contar
los textos de test no equivale a puntuarlos con el modelo. Las declaraciones
de licencia de la ficha fija de Wikipedia y de su documentación legal actual
se conservan por separado con sus enlaces.

# Diálogos variados de varios turnos: v1

Los archivos `clarity.jsonl`, `practical.jsonl` y `state.jsonl` contienen 120
familias ficticias completas, escritas por **asistentes de IA del proyecto**.
No son textos de autores humanos independientes, salidas generadas por el modelo
Unidad ni conversaciones privadas del creador. Se conservaron las fuentes y
particiones originales.

El preparador produjo 480 prefijos sintéticos de uno a cuatro intercambios y
añadió 150 anclajes públicos de OASST, provenientes del
[corpus humano v3](../human-short-es-v3/). Cada prefijo supervisa únicamente la
última respuesta assistant y EOS. La palabra «natural» en el nombre describe
el objetivo de variedad del estilo, no la autoría ni una calidad demostrada.

| Archivo | Contenido | SHA256 |
| --- | --- | --- |
| `clarity.jsonl` | 40 familias ficticias de aclaraciones | `3a0f73f5cda2a9f2c594aedf56e2fd5256f723d40a984ed7e6d7ff16124e3e51` |
| `practical.jsonl` | 40 familias ficticias prácticas | `7b6c58bde9e7f0e512176c175839f8be50771a51711ac1c52f955f070678e211` |
| `state.jsonl` | 40 familias ficticias de contexto | `c452b9262520d97ea398fb46cbf5c54488d4ba79d35e2a0757fad89c032bc47e` |
| `train-validation.jsonl` | 460 TRAIN y 110 VALIDATION: 420 sintéticos y 150 anclajes OASST | `c97f5ed2d15602caaa6bc036dbae9df4ecae339d4de2badc0f3adc38f38c0cb7` |
| `authored-test.jsonl` | 60 prefijos sintéticos TEST, separados | `b9728afda403e17c0b5206dec5471bd0e24759654dd4397f237a29a09c90ce2f` |

## Atribución y alcance

Los 150 anclajes OASST mantienen su procedencia original y su licencia declarada
Apache-2.0; la atribución y el texto de licencia están en
[`human-short-es-v3`](../human-short-es-v3/README.md).
Los textos sintéticos del proyecto no declaran una licencia de datos
independiente. La declaración Apache-2.0 del manifest corresponde a las fuentes
OASST, no a una licencia nueva para todos los ejemplos o para el repositorio.

`manifest.json` conserva autores, fuentes, hashes, límites de preparación y
particiones. No certifica factualidad, seguimiento de instrucciones o
conversación competente. Las revisiones de IA del proyecto no equivalen a una
evaluación humana independiente; las estadísticas de palabras tampoco acreditan
capacidad lingüística.

Para repetir la preparación desde la raíz del repositorio:

```bash
python examples/prepare_natural_multiturn.py \
  --input examples/natural-multiturn-v1/clarity.jsonl \
  --input examples/natural-multiturn-v1/practical.jsonl \
  --input examples/natural-multiturn-v1/state.jsonl \
  --human-corpus examples/human-short-es-v3/human-short-es.train-validation.jsonl \
  --expected-human-corpus-sha256 7016322d12f49b8de53abbaee57826c2a0cbbb50164a109cf94b4edfb51d7a3c \
  --tokenizer models/conversation-language-v7/tokenizer.json \
  --output /tmp/unidad-natural-multiturn-v1
```

La salida debe ser nueva. El preparador registra longitudes de ingestión y
mantiene TEST separado; no ejecuta evaluación de aceptación. Esta entrega se
limitó a copiar/verificar archivos y no cargó tokenizadores ni examinó casos
reservados de evaluación.

# Selector de contexto propio v1

Regresión logística independiente con **20 características y 21 parámetros**,
entrenada desde una inicialización aleatoria propia. Sus puntuaciones ordenan
fragmentos del historial; no genera respuestas ni extrae hechos nuevos.

La corrida completó 2.000 épocas en 7,218 segundos. Se seleccionaron los pesos
por pérdida de validación y se fijaron antes de consultar TEST. El corpus incluye
144 casos TRAIN, 48 de validación y 48 TEST, con entidades separadas y ejemplos
ficticios escritos por Codex. Las plantillas se repiten entre particiones: los
resultados no acreditan transferencia a conversación arbitraria.

En TEST, la fuente elegida en primer lugar fue correcta en 48/48 casos, frente
a 28/48 con puntuación léxica y recencia. Recall@3 fue 48/48 frente a 44/48.
SOVENGI usa la puntuación de forma relativa, con candidatos respaldados por
coincidencia léxica, hilo, objetivos o restricciones; no emplea un umbral de 0,5
como si la salida estuviera calibrada para toda conversación.

- [Pesos y contrato](ranker.json).
- [Corpus ficticio y particiones](corpus.json).
- [Recibo de entrenamiento](training.json).
- [Checkpoint reanudable](checkpoint.json).
- [Selecciones y referencia](evaluation.json).
- [Integración, rendimiento y respuestas reales](../../docs/context-model-20261013/README.md).

Identidad: `unidad-context-90befedd5f5534a3f5da59de9040c169e41e62fb16175b5e5fc2ece97dce7fbf`.
SHA-256 del archivo `ranker.json` y de su copia empaquetada:
`c90dfebbd49d505a8d137c0ca645a7fe64dfc72d9d9bc02df1571adf419ab844`.
La identidad del modelo usa el hash canónico del contenido sin los campos de
identidad; es distinto del hash del archivo.

Para reproducir sin sobrescribir esta entrega:

```bash
python -m ia_propia.context.training --output private/context-reproduction --epochs 2000
```

Una corrida nueva debe utilizar una carpeta vacía. `--resume` comprueba corpus,
configuración y checkpoint y continúa exactamente el mismo entrenamiento.
La inferencia sólo carga los pesos; no inicia entrenamiento.

# Español público breve: v3

Este directorio conserva el corpus exacto preparado para los experimentos de
Unidad. TRAIN y VALIDATION comparten un archivo; TEST permanece en otro archivo.
Los registros, respuestas, identificadores, procedencia y particiones se copiaron
sin cambios. Esta entrega no ejecutó tokenización ni evaluación de TEST.

| Archivo | Contenido | SHA256 |
| --- | --- | --- |
| `human-short-es.train-validation.jsonl` | 3153 TRAIN y 392 VALIDATION | `7016322d12f49b8de53abbaee57826c2a0cbbb50164a109cf94b4edfb51d7a3c` |
| `human-short-es.test-reserved.jsonl` | 404 TEST | `9e3f5f6434038d90b26fed09750cc61f7ea0b3df7e8b0b3eb351348bb6488dc3` |
| `human-short-es.manifest.json` | Preparación, filtros y fuentes | `cb5d566e66c3c0a26c135348fbd9b23f25b1df7fa1e3e4f325e9fb6c3b5d6838` |

## Atribución y licencia

Los textos proceden de OpenAssistant y sus contribuidores. Se conservaron las
etiquetas originales de revisión humana positiva y se excluyeron mensajes
marcados como sintéticos externos, rechazados o incompatibles con los filtros del
importador. Esas etiquetas no certifican el origen de cada contribución ni sus
hechos, gramática o calidad conversacional.

- [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b): revisión `fdf72ae0827c1cda404aff25b6603abec9e3399b`; archivo `2023-04-12_oasst_ready.trees.jsonl.gz`; SHA256 `2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7`.
- [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2/tree/179dd21fc55192153d94adb0e0ce8f69e222bf75): revisión `179dd21fc55192153d94adb0e0ce8f69e222bf75`; archivo `2023-11-05_oasst2_ready.trees.jsonl.gz`; SHA256 `7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b`.

Ambas fichas fijadas declaran Apache-2.0. Se incluye el
[texto de licencia original de OASST1](LICENSE-oasst-Apache-2.0.txt), SHA256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
Las [fichas y recibos de fuentes](../../docs/conversation-integration-20261013/data/human-short-es-v3/)
conservan sus URLs y hashes. La ficha fijada de OASST2 es su declaración de
licencia; su repositorio no proporciona un archivo LICENSE separado.

## Preparación y alcance

Se mantuvo la asignación histórica por árbol completo, sin mover familias entre
particiones. La unión prioriza OASST1 para identificadores repetidos. Los filtros
de duplicados y longitud quedaron fijados antes de seleccionar los ejemplos.
La longitud se midió originalmente con el BPE propio de 8192 tokens: secuencia
completa de hasta 513 tokens y respuesta de hasta 95 tokens.

V3 añade una regla explícita para autorrepresentación externa del assistant.
Excluye familias completas cuando sus respuestas dicen ser OpenAssistant,
ChatGPT o un modelo creado por OpenAI; conserva explicaciones, negaciones, citas
y código. Se eliminaron 138 registros de 54 familias del snapshot anterior, sin
añadir ejemplos de reemplazo ni reescribir textos. Los recibos detallan la regla
y su revisión. Es una corrección de fuente motivada por desarrollo, no una mejora
demostrada del modelo ni una certificación factual.

La revisión de ocho muestras TRAIN corresponde al snapshot histórico v1, que
se conserva identificado en los recibos. Es una revisión de IA del proyecto,
no una evaluación humana independiente de v3. Persisten errores y afirmaciones
imprecisas de la fuente; predominan preguntas y respuestas breves.

Para repetir la preparación desde las fuentes públicas, desde la raíz del repo
y con las dependencias `conversation` instaladas:

```bash
python examples/prepare_human_dialogues.py \
  --download \
  --raw-directory /tmp/unidad-oasst-pinned \
  --output /tmp/unidad-human-short-es-v3 \
  --tokenizer models/conversation-language-v7/tokenizer.json
```

El directorio de salida debe ser nuevo. El tokenizador esperado tiene SHA256
`fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d`.
La reproducción prepara datos y comprueba límites de longitud; no entrena ni
evalúa el modelo. El corpus publicado permite repetir un entrenamiento sin
descargar de nuevo los archivos grandes de las fuentes.

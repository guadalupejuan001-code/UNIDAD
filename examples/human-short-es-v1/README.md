# Snapshot humano breve v1: pilotos históricos

Este snapshot conserva el archivo exacto de los primeros pilotos C7/v7:
`human-short-es.train-validation.jsonl`, con 3269 TRAIN y 394 VALIDATION.
SHA256: `8fee3c1f78196b05ae8df2cca5e6f8ba7ca412117d32dfaae73110b70fb3afcd`.
El manifest original tiene SHA256
`aa411c5ac6de3d9bc188a64fc5556b193202a9c636d5eb0718d012a27f8b57fd`.

El archivo publicado no contiene TEST. Su manifest histórico también describe
un archivo TEST reservado original que no se incluye en esta carpeta ni debe
mezclarse con TRAIN/VALIDATION. Este snapshot no aplica la corrección posterior
de autorrepresentación externa de [v3](../human-short-es-v3/README.md); se conserva
para reproducir el corpus de los pilotos, sin reemplazar silenciosamente sus datos.

Los textos son públicos de OpenAssistant y sus contribuidores, con etiquetas
originales de revisión positiva. OASST1 corresponde a la revisión
`fdf72ae0827c1cda404aff25b6603abec9e3399b` y OASST2 a
`179dd21fc55192153d94adb0e0ce8f69e222bf75`.
Las [fuentes fijadas y sus hashes](../human-short-es-v3/README.md) son las mismas
de v3. Ambas fichas declaran Apache-2.0; aquí se incluye el
[texto original de licencia OASST1](LICENSE-oasst-Apache-2.0.txt).

Los filtros conservaron las particiones históricas por familia. Las etiquetas
de revisión no certifican factualidad, gramática, origen de cada contribución
ni conversación competente. La revisión de ocho muestras TRAIN fue realizada
por un asistente de IA del proyecto y no es una evaluación humana independiente.
No se incluyen mensajes privados del creador.

Para repetir los entrenamientos históricos, consume este archivo fijado y
verifica su SHA256. El preparador público actual produce v3; la
[fuente histórica v1](../../docs/conversation-integration-20261013/source/prepare_human_dialogues-v1.py)
se conserva con SHA256
`8818a2ee12e876d92aadce7528b83e68436f36f85336d646f0bc0c199e427452`.
Los [recibos](../../docs/conversation-integration-20261013/data/README.md)
distinguen claramente ambos snapshots. Esta entrega sólo copió y verificó datos;
no cargó modelos o tokenizadores ni abrió casos reservados de aceptación.

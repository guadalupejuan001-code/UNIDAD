# Datos fijados de la integración del 13 de octubre de 2026

Los corpus se almacenan una sola vez en `examples/`; aquí se conservan recibos,
fuentes de atribución y referencias ligeras. No se copiaron cachés de tokens,
pesos ni checkpoints a este directorio.

- [Español público breve v3](../../../examples/human-short-es-v3/README.md): TRAIN/VALIDATION y TEST separados, fuentes OASST fijadas y licencia Apache-2.0.
- [Snapshot humano v1](../../../examples/human-short-es-v1/README.md): archivo TRAIN/VALIDATION exacto de los pilotos iniciales, conservado con su manifest histórico.
- [Historia ideal/generada v1](../../../examples/on-policy-history-v1/README.md): corpus sintéticos pareados, recibos y salidas reales del teacher propio C7.
- [Diálogos variados v1](../../../examples/natural-multiturn-v1/README.md): fuentes ficticias escritas por asistentes de IA del proyecto, con anclajes públicos OASST identificados aparte.

`delivery-receipt.json` verifica las copias byte por byte, con tamaños y SHA256.
La entrega mantuvo los roles, hechos, objetivos, etiquetas y particiones; no
cargó modelos/tokenizadores, tokenizó TEST ni abrió casos reservados de aceptación.

## Recibos y límites

`human-short-es-v3/` conserva las fichas originales OASST1/OASST2, URLs,
revisiones, hashes, licencia y prueba de que v3 sólo excluye familias del corpus
anterior. La inspección y revisión históricas de ocho ejemplos TRAIN pertenecen
a **v1**; se incluyen con ese nombre, sin atribuir una revisión completa a v3.
`human-short-es-v1/` conserva el manifest histórico adicional; su corpus
TRAIN/VALIDATION se entrega en `examples/human-short-es-v1/` sin copiar TEST.

`on-policy-history-v1/` conserva manifest y diseño. Su materia es ficticia y
sintética; sólo las respuestas anteriores del brazo generado proceden de Unidad
C7. Sus respuestas finales supervisadas siguen siendo las originales sintéticas.
El path absoluto de su revisión dentro del manifest es el path original del
entorno (`/workspace/UNIDAD/examples/conversation-grounded-curriculum-v2.review.json`);
se mantuvo literal. La revisión pública equivalente está en `examples/`.

`natural-multiturn-v1/` conserva el manifest original con 480 prefijos sintéticos
y 150 anclajes OASST en el corpus completo. Apache-2.0 es la declaración de esas
fuentes públicas; no se añadió una licencia independiente a los textos
sintéticos ni al proyecto. Ningún recibo de preparación demuestra por sí mismo
calidad conversacional o factual.

Los preparadores fijados se conservan en [`../source/`](../source/); sus copias
permiten inspeccionar el código usado. Las instrucciones de reproducción desde
la raíz del repositorio están en cada README de corpus.

# Control LAST 4.096: generación y dos revisiones reales

Esta entrega conserva los cuarenta turnos reales de DEV del modelo propio
`unidad-conversation-a2cae3b623142153487760d2f09e1a9c2fc851a5a1a09f1cad83b9ffaa6c2d7d`.
Hubo cero excepciones de generación. Se usaron los mismos casos, historia de
respuestas realmente generadas, temperatura cero, semilla 42, hasta 160 tokens
y un hilo Torch. LAST conserva las actualizaciones del control; su BEST 0 es
el padre heredado, un artefacto distinto.

Se copiaron **45 JSON byte por byte**: salidas agregadas, cuarenta turnos,
plantilla original y configuración/manifiesto/tokenizador del artefacto usado.
La plantilla no es una revisión realizada. Se añaden cuatro copias exactas
de revisiones/assessments, conservando la generación y su recibo original.
Ambas revisiones de IA del proyecto obtienen **0 completas, 1 parcial, 39
fallidas, 0/10 diálogos, 2 críticas y 7 técnicas**; aceptación falsa.
Los IDs críticos son `outing_day_revision` y `outing_one_sentence_final`.
Se compararon los IDs y las cuatro claves booleanas de cada turno: coinciden.
Las notas y métodos originales permanecen completos, sin adjudicar un juicio
humano independiente o promover el control.

- [Respuestas y contexto efectivo](outputs.json), SHA
  `585e4166e4fdb6c5af3a5cd25c636c700327ea9b073cee05c195bc186518fa52`.
- [Resumen de generación](generation-summary.json).
- [Metadatos del artefacto](artifact-metadata/).
- [Recibo exacto por archivo](publication-integrity-generation.json).

Los casos son ficticios y expuestos de desarrollo, sin datos personales del
creador. No se abrió TEST ni una suite reservada. No se ejecutaron modelos,
entrenamiento o pruebas de software para preparar esta publicación; los pesos
y documentos originales permanecen intactos.

La [primaria](review-primary.json), [cruzada](review-cross.json),
[assessment primario](assessment-primary.json) y [cruzado](assessment-cross.json)
están copiados byte por byte. El [resumen](comparison-control4096-v1.json) y
[recibo adicional](publication-integrity-reviews.json) conservan hashes y la
comparación directa de juicios. Menos críticas no acredita mejora ante las
omisiones y cero respuestas completas.

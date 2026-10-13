# Tratamiento LAST 4.096: generación y dos revisiones reales

La generación real contiene cuarenta respuestas de diez diálogos ficticios de
DEV, sin excepciones. Las dos revisiones de IA del proyecto obtienen:

| Revisión | Completas | Parciales | Fallidas | Diálogos completos | Críticas | Técnicas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primaria | 0/40 | 1 | 39 | 0/10 | 14 | 1 |
| Cruzada | 0/40 | 1 | 39 | 0/10 | 14 | 1 |

No se comprobó mejora conversacional y no se aceptó ni promovió LAST. La única
parcial es `library_initial_deadline`: se reconocen viernes y seis, pero
«es de las seis seis» no es redacción natural. El fallo técnico es
`library_day_correction`, agotamiento de tokens sin descarte de contexto.
Los otros treinta y nueve turnos terminaron por EOS; no hubo excepciones ni
texto vacío. El archivo de generación original mantiene `reviewStatus=pending`
y `accepted=null`; los juicios posteriores se guardan por separado.

El modelo es
`unidad-conversation-d6d860222baf5e8952eade42fcda3a0bd52dc77000c0c254ca895a91fdab7160`,
con 15.179.520 parámetros. Es LAST del tratamiento cerrado en 4.096 updates;
BEST seleccionado por validación sigue en 1.024. Generó con
`ConversationEngine`, temperatura cero, semilla 42, hasta 160 tokens nuevos y
un hilo Torch. Cada diálogo conserva sólo sus respuestas anteriores reales,
sin insertar respuestas ideales.

Se copiaron **45 JSON de generación/metadatos byte por byte**: `outputs.json`, cuarenta archivos por
turno, la plantilla de revisión original y configuración/manifiesto/tokenizador
del artefacto realmente usado. La plantilla no representa una revisión hecha.
Los pesos permanecen en su artefacto público y se verificaron por SHA-256;
no se duplicaron dentro de documentación.

- [Respuestas, historial, tokens y contexto efectivo](outputs.json), SHA
  `8011eefafab0b823fe49fe689607f8225eb8c36a7a9b486c37bbe8b28ba114ac`.
- [Resumen de generación](generation-summary.json), sin grados inventados.
- [Referencia al artefacto real y hashes](artifact-reference.json) y
  [metadatos exactos](artifact-metadata/).
- [Recibo de copia por archivo](publication-integrity-generation.json).

Se comprobó la identidad del modelo en cada respuesta, el orden de usuarios
del caso congelado y que cada historial incorpora exactamente las respuestas
previas generadas. El manifiesto embebido en las salidas coincide con el
manifiesto del artefacto real. Sus pesos tienen SHA
`5888240335df305a5f80aa438200b524af741278d8e8c7af234d8a60f86f7f6e`.

No se ejecutaron inferencia, entrenamiento o pruebas de software para esta
publicación. No se leyó ni publicó una suite reservada o TEST. Los casos son
originales y ficticios, sin datos personales del creador. Las revisiones y evaluaciones se añadieron como archivos separados,
conservando las salidas y el recibo inicial de generación.

## Revisiones y discrepancias

Se añaden **cuatro JSON originales copiados byte por byte**:
[revisión primaria](review-primary.json), [cruzada](review-cross.json),
[evaluación primaria](assessment-primary.json) y [cruzada](assessment-cross.json).
El [recibo adicional](publication-integrity-reviews.json) conserva sus hashes
sin modificar el recibo inicial; el [resumen conjunto](comparison-last4096-v1.json)
registra todos los IDs críticos y las diferencias booleanas observadas.

Se compararon los IDs directamente: ambas revisiones señalan los mismos catorce
críticos. Difieren en español para `cinema_attendee_reduction` y
`outing_one_sentence_final`: la primaria lo considera incumplido y la cruzada,
cumplido. No se impone un juicio único ni se cambia el texto real. La primaria
incluye una auditoría auxiliar de siete casos mediante IA del proyecto; no es
revisión humana independiente. La revisión cruzada conserva su propio método.
La aceptación sigue siendo 36/40 completas, 8/10 diálogos y cero críticas o
técnicas en una evaluación nueva con candidato fijado; ambos juicios fallan.
No se abrió una suite reservada para estas revisiones o su publicación.

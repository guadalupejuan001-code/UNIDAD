# Tratamiento natural + LATAM: desarrollo en el paso 1.024

Snapshot propio intermedio `unidad-conversation-ec9ab262aaaa5761dfe77e51232ca9fcf78bea93b7af97ec17d971557f0166df`,
procedente del tratamiento que sigue entrenando. No se comprobó mejora de
respuestas completas ni diálogos; no se promovió un candidato.

| Revisión de IA del proyecto | Completas | Parciales | Fallidas | Diálogos completos | Críticas | Técnicas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primaria | 0/40 | 0 | 40 | 0/10 | 1 | 11 |
| Cruzada | 0/40 | 0 | 40 | 0/10 | 0 | 11 |

La discrepancia crítica está en `cinema_initial_total`. Ante tres entradas y
dieciocho euros, la respuesta real es «Tus entradas en dos entradas.». La
primaria interpreta dos como una cantidad errónea; la cruzada considera que la
construcción fragmentaria no afirma inequívocamente un total. Los dos juicios
permanecen intactos. `criticalCorrect=null` frente a `true` en los dos turnos
no críticos es una diferencia de representación, sin criterio crítico aplicado.

Se usaron los mismos casos ficticios y `ConversationEngine`, con historia de
las respuestas realmente generadas, temperatura cero, semilla 42 y hasta 160
tokens por respuesta. Siete turnos agotaron tokens y cinco descartaron contexto,
con uno común: once fallos técnicos distintos. Treinta y tres terminaron por
EOS; no hubo excepciones ni respuestas vacías. Las críticas inferiores no
convierten omisiones, ecos o textos rotos en conocimiento comprobado.

Los **47 JSON originales** están copiados byte por byte:

- [Salidas e historial real](outputs.json), SHA
  `0d8a6bfef6d9c35b98afb94c480bf565236c0652db03034d2eb4daafedb3d46b`.
- [Revisión primaria](review-primary.json) y [cruzada](review-cross.json).
- [Evaluación primaria](assessment-primary.json) y [cruzada](assessment-cross.json).
- [Integridad cruzada original](integrity-cross.json),
  [recibo de publicación](publication-integrity.json) y
  [resumen con discrepancias](comparison-treatment1024-v1.json).

Son revisiones de IA del proyecto, no evaluaciones humanas independientes. El
paso 1.024 no constituye cierre del tratamiento, evaluación del brazo control
ni aceptación reservada. Se conserva 36/40 completas, 8/10 diálogos y cero
críticas/técnicas como aceptación. La suite expuesta no se incorpora al corpus
ni a sus filtros.

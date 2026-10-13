# LAST 8.192: generación real y dos revisiones

El artefacto propio LAST 8.192 produjo los cuarenta turnos de DEV con historial
real, temperatura cero, semilla 42 y hasta 160 tokens nuevos. Su identidad es
`unidad-conversation-f06b96e6c4c5149979b4846f9d03a4dd43dc8521dd13e3d5733bb164f90268f5`.
LAST conserva el cierre del entrenamiento; BEST seleccionado por validación
sigue siendo 4.096. Esta generación no promueve LAST ni abre una aceptación
reservada.

| Revisión | Completas | Parciales | Fallidas | Diálogos completos | Críticas | Técnicas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primaria, IA del proyecto | 0/40 | 3 | 37 | 0/10 | 11 | 8 |
| Cruzada, IA del proyecto | 0/40 | 3 | 37 | 0/10 | 12 | 8 |

La primaria revisó estos cuarenta textos, su historial realmente generado y los
casos/rúbrica congelados. No consultó otros juicios del LAST ni juicios de modelos
anteriores para revisar estas respuestas. No es revisión humana independiente.
La revisión cruzada ya está publicada por separado. Sus doce críticas incluyen
`tap_flow_correction`: interpreta «el calor sale bien» como buen funcionamiento
de la caliente. La primaria considera insuficientemente identificable esa
afirmación de caudal. La primaria suma once críticas. Ambas conservan además
diferencias sobre relevancia en `library_initial_deadline` y español en
`quiet_scene_preference`. No se impone un consenso ni se cambia el texto real.

Las tres parciales son los primeros tres turnos de biblioteca: aportan viernes,
jueves a las seis y las seis respectivamente, con omisiones y redacción
incompleta. El resto falla por contenido, instrucciones, críticos o aspectos
técnicos. Las respuestas de cine y reunión entran en repetición: ocho agotan
tokens, y cuatro de esos mismos turnos descartan contexto. Los descartes de
34 y 9 tokens afectan al inicio del sistema; los de 226 y 209 eliminan también
el primer mensaje del usuario. Las repeticiones iniciales ya aparecen antes de
esos descartes. No se atribuye todo el fallo a pérdida de contexto.

La revisión contiene las claves booleanas que requiere `assess`. Además,
`dimensionScores` registra 0, 0,5 o 1 por criterio y `rubricGrade` el grado del
turno; el booleano es verdadero sólo para 1. Cada crítico identifica una
afirmación concreta en `criticalError`. Un fragmento incoherente, omisión o eco
sin una afirmación falsa identificable no se marca crítico automáticamente.

Se conservan byte por byte **48 archivos** de generación y revisión primaria,
y se añaden **tres archivos** de revisión cruzada e integridad de su publicación.
Las copias y recibos primarios originales permanecen intactos:

- [Salidas, tokens, contexto e historial](outputs.json), SHA
  `31ba2a5e74002fe10407ad540a07e0290d9acc91512c442c760011bff0077282`.
- [Revisión primaria](review-primary.json), SHA
  `dd40776a7cbda4fd1a3fbcf7d0a3b6d17ae725b174685bf3e9622556ebe1063c`.
- [Evaluación nativa](assessment-primary.json), SHA
  `f51b75cd0463c5626920d0a8182d555039a835c4ce06533b616dbdf029909b83`.
- [Grados y diagnóstico técnico](summary-primary.json).
- [Revisión cruzada](review-cross.json), SHA
  `6a3e7d7239c97b83cd5da5705b41be5c2144540e288ba748ab2763a369f930a9`,
  y [evaluación cruzada](assessment-cross.json), SHA
  `02be5ec724af5b84b14448563dfaada943c83e8ffef2b37fea15dc75ae59e23a`.
- [Comparación con discrepancias por turno](comparison-last-v1.json),
  [integridad de publicación cruzada](review-cross-publication-integrity.json)
  y [recibo adicional de copia](publication-integrity-cross.json).
- [Integridad de fuentes](review-primary-integrity.json) y
  [recibo de copia](publication-integrity-primary.json).
- [Recursos de `assess`](assessment-primary-resources.json): una CPU, límite de
  60 segundos, 0,03155 segundos observados y pico RSS de 12.992.512 bytes.

No se cargaron pesos, se generaron nuevas respuestas, se entrenó ni se ejecutaron
pruebas de software durante revisión/assessment/publicación. Los casos son
ficticios, sin datos personales del creador. La suite expuesta sigue excluida del
corpus y de sus filtros. La aceptación exige 36/40 completas, 8/10 diálogos y
cero críticas o técnicas; LAST no la cumple en ninguna de las dos revisiones.

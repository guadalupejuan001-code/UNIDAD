# Continuación del diagnóstico conversacional — 12 de octubre de 2026

**Unidad todavía no supera la conversación básica. C7 se conserva como referencia experimental; no se ha promovido otro modelo.** Esta continuación corrigió dos huecos del evaluador, auditó la separación del corpus y ejecutó una comparación de datos desde C7. Los dos candidatos de 128 pasos fallan la conservación de capacidades en los 40 turnos conocidos; no se amplía esta intervención. Se conservan todas las respuestas, incluidas las incoherentes.

Entrega cerrada el **12 de octubre de 2026**. Ambos entrenamientos, las siete generaciones posteriores y todas sus revisiones terminaron. El ensayo produjo 480 respuestas nuevas conservadas: 448 con historial real y 32 con historia ideal, separadas como diagnóstico. La referencia y los candidatos siguen sin superar la aceptación. No hay entrenadores ni generaciones activos en este entorno. Los presupuestos anteriores de 24 horas y dos horas están cerrados. Esta comparación nueva y acotada usó los recursos gratuitos existentes bajo la misión vigente; no reinició esos presupuestos.

## Versiones y referencia conservada

La continuación parte del [PR en borrador 1](https://github.com/guadalupejuan001-code/UNIDAD/pull/1), cuya base es `a1a283caabf89b3191c8ce64c2a950e9edbb0edf`. El código auditado aquí es `8b466cc43015289a4568639304ef9841397ad6fa`, en la rama `codex/conversation-wording-repair-20261012`, con un worktree separado. El worktree original de `main` se conserva.

Los experimentos del [informe anterior](work-conversation-diagnosis-20261012.md) utilizaron `b5fc677dfeb90b5e9cbd50f26cf878fad6a79d6a`. Sus candidatos de copia y sus pérdidas pertenecen a esa versión. El soporte de supervisión de respuestas largas de `a1a283c` sigue conservado; **no se aplicó `source-pointer-v2.patch`**.

La referencia sigue siendo `models/conversation-reference-c7`, un Transformer propio de 6.983.937 parámetros, incluidos sus parámetros de copia v1, con contexto de 512 y BPE de 8.192 tokens. Su procedencia continúa desde pesos aleatorios propios; no se incorporaron pesos externos ni una API de conversación.

| Archivo C7 | SHA-256 |
| --- | --- |
| `weights.safetensors` | `666527de240a7fc9b082f233699a1c808c6bdd8f929651824ce4975377c7c836` |
| `config.json` | `d30b5cd427fb9711d58e8eb55daf9a32529f872248b706805001e9eebd02f12b` |
| `tokenizer.json` | `fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d` |
| `manifest.json` | `12e5199ef762ce2a24bb0fbd118c2671d77b1e2d305f1f864add3a2f79999242` |

Se publican su [configuración](work-repair-experiment-20261012/reference/config.json) y [manifiesto](work-repair-experiment-20261012/reference/manifest.json). Los pesos y los estados de optimizador se conservan también en [la rama de artefactos de GitHub](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012), separados de main y de los cambios de código de este PR.

La [comprobación de equivalencia](work-repair-experiment-20261012/reference/c7-baseline-equivalence.json) compara las 40 respuestas de C7 del PR, generadas con 32 tokens máximos, con las obtenidas en el motor preservado con 160: son iguales en preguntas, historial, tokens generados y texto. Todas terminan por EOS. Esta equivalencia acredita **C7 en esos casos**, no otros candidatos ni otros límites.

## Criterios que no cambian

En los 40 turnos conocidos, C7 mantiene **7 completas, 5 parciales y 28 fallidas; 6 fallos críticos y 0/10 diálogos completos** según la revisión primaria archivada.

- Meta parcial: como máximo **14 fallidas de esas 40**, sin superar los **6 fallos críticos**. La comparación ejecutada no consiguió esa reducción.
- Aceptación general: al menos **36/40 completas**, **8/10 diálogos completamente correctos**, y **cero fallos críticos o técnicos**, confirmados con una prueba nueva reservada después de seleccionar el candidato.

Las pruebas anteriores de 40 y 20 turnos, el diagnóstico descrito aquí y la nueva validación son desarrollo. Ninguna constituye ahora una aceptación final independiente. Una coincidencia literal sirve para diagnosticar; una respuesta semánticamente válida puede usar palabras diferentes.

## Evaluador incorporado y dos errores corregidos

Se conserva la corrección del PR: 40 turnos en diez diálogos de cuatro turnos, requisito de ocho diálogos completos, hashes de casos y rúbrica, comprobación del historial realmente generado y rechazo de respuestas vacías o incorrectas críticas.

La auditoría encontró además dos registros artificiales que el evaluador aceptaba como 40/40: uno identificaba otro modelo en el manifiesto; el otro declaraba cantidades contradictorias de recorte entre respuesta y contexto efectivo. Se corrigió `assess` para rechazar ambos. Son **controles técnicos fabricados**, no conversaciones atribuidas a Unidad.

Pasaron **26 pruebas pertinentes: 21 del evaluador y cinco del motor**, sin fallos, errores ni saltos. Se conservan [controles antes](work-repair-experiment-20261012/evaluator/before-gap-controls.json), [controles después](work-repair-experiment-20261012/evaluator/after-gap-controls.json), [registro de pruebas](work-repair-experiment-20261012/evaluator/integration-tests.log), [recibo de auditoría](work-repair-experiment-20261012/evaluator/receipt.json) y [commit comprobado](work-repair-experiment-20261012/evaluator/integration-commit.json).

La auditoría comprobó que los 14 archivos del núcleo conversacional y 89 archivos existentes de evidencia pública/C7 permanecían iguales a `eb0f811aabdf4ba5bfe8957ceeaeebc942b9adb5`, el HEAD auditado del PR. Esa cifra no incluye los nuevos documentos de esta entrega. Las pruebas de software no acreditan calidad conversacional ni prueban por sí solas el origen de un texto; por eso se conservan artefactos, hashes y registros de ejecución.

No se cambiaron silenciosamente condiciones históricas: el generador de aceptación heredado sigue con **24 tokens / temperatura 0 / semilla 109**; los experimentos del PR usaron **32 / 0 / 42**; el diagnóstico nuevo usa **160 / 0 / 42**. No se reinterpretan como una única configuración.

## Corpus: separación por mundos, pero formulaciones compartidas

Se reprodujo exactamente el corpus del PR: SHA-256 `862996c131ea17e3623f55530b2f6d61e1d4f64e37672bc99581c19df17a7186`, con **4.224 TRAIN y 384 VALIDATION**, 88 y ocho mundos respectivamente. No se movió ni editó ningún registro original.

La [auditoría completa](work-repair-experiment-20261012/corpus-audit/audit-summary.json) encontró que VALIDATION comparte con TRAIN **56/56 formulaciones de mensajes, 70/70 formulaciones de prefijos, 115/115 formulaciones de diálogos completos y 12/12 composiciones de eventos**, tras sustituir los valores ficticios por sus campos. Los mundos completos están separados; las maneras de preguntar y combinar eventos no lo están. Por eso la pérdida anterior mide principalmente transferencia de valores dentro de formas conocidas.

La marca de historia neutral aparece en el 25 % de los registros, pero sólo el **18,75 %** sustituye alguna respuesta anterior del asistente: los prefijos del primer turno no tienen respuestas anteriores. Se documentó esa diferencia sin cambiar el corpus.

También se registraron dos ambigüedades del objetivo original: `project_change` daba por supuesta la propiedad del proyecto, y `daily_sequence` no explicitaba todas las relaciones temporales conservadas tras el cambio. Los registros antiguos siguen disponibles. Ocho prefijos nuevos aclaran esas ambigüedades y se identifican por separado; no son pares limpios de cambio exclusivo de redacción.

Se congeló [una validación adicional de desarrollo](work-repair-experiment-20261012/heldout-development/validation.jsonl), SHA-256 `e305adb7fec7c045d2b3ca2d066297aac9dcc04bc5f580c54dae55ac664ab37c`: **20 diálogos completos y 80 prefijos**, en los ocho mundos de la antigua validación. Reserva 48 formulaciones de usuario dentro de doce diálogos y ocho nuevas composiciones completas con otros 32 mensajes. No coincide con las formas completas de TRAIN ni con los mensajes/prefijos de las pruebas de desarrollo protegidas.

El [manifiesto congelado](work-repair-experiment-20261012/heldout-development/freeze-manifest.json) conserva hashes, procedencia y límites: entrada máxima de 260 tokens, última respuesta supervisada con EOS de hasta 34, sin recortes. Otra IA del mismo proyecto revisó los 20 diálogos y aceptó sus objetivos; **no es evaluación humana independiente ni evaluación de respuestas del modelo**. Los 80 prefijos comparten veinte recetas y ocho mundos, y no equivalen a 80 observaciones independientes. El español común y operaciones elementales siguen compartidos entre particiones.

## Diagnóstico nuevo: redacción, orden y acumulación del historial

Se congelaron [casos](work-repair-experiment-20261012/wording-history/cases.json) y [protocolo](work-repair-experiment-20261012/wording-history/protocol.json) antes de generar respuestas. Hay **48 consultas individuales**: doce preguntas base, dos redacciones y dos órdenes de los mismos hechos. Otras cuatro conversaciones de cuatro turnos se generaron con historia real y con historia ideal, dando **32 respuestas** adicionales.

En la rama real, cada turno conserva exactamente las respuestas anteriores producidas por C7. En la ideal sólo se suministran respuestas ideales de turnos anteriores: nunca la respuesta esperada actual o futura. Es una ayuda diagnóstica y no cuenta como conversación autónoma. Los primeros cuatro turnos coinciden exactamente entre ramas, porque todavía no hay historial de asistente.

Todas las respuestas proceden del mismo `ConversationEngine`, con **160 tokens máximos, temperatura 0, semilla 42 y un hilo de PyTorch**. Las 80 terminaron por EOS: **cero límites alcanzados, respuestas vacías, errores técnicos o recortes de entrada**. El [recibo de contexto](work-repair-experiment-20261012/wording-history/c7-default-system-context-receipt.json) reconstruye las entradas efectivas mediante la ruta del motor, incluido el mensaje de sistema; no es una captura independiente de activaciones internas.

| Subconjunto | Revisión | Completas | Parciales | Fallidas | Críticas | Diálogos completos |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 48 consultas individuales | Primaria | 1 | 2 | 45 | 9 | — |
| 48 consultas individuales | Cruzada | 1 | 4 | 43 | 9 | — |
| 16 turnos con historia real | Ambas | 1 | 1 | 14 | 1 | 0/4 |
| 16 turnos con historia ideal, sólo diagnóstico | Primaria | 2 | 2 | 12 | 1 | 0/4 |
| 16 turnos con historia ideal, sólo diagnóstico | Cruzada | 2 | 1 | 13 | 1 | 0/4 |
| Total de 64 turnos sin historia ideal | Primaria | 2 | 3 | 59 | 10 | 0/4 de varios turnos |
| Total de 64 turnos sin historia ideal | Cruzada | 2 | 5 | 57 | 10 | 0/4 de varios turnos |

Las críticas son un atributo de respuestas ya clasificadas C/P/F, no otra categoría que deba sumarse. La rúbrica crítica congelada de este diagnóstico es acotada y **no cubre todas las contradicciones factuales**; las demás se conservan como fallidas y se describen en las revisiones. No se modificaron sus criterios después de leer las respuestas.

Las [80 respuestas completas en texto](work-repair-experiment-20261012/wording-history/c7.actual.txt) y [JSON con historial y tokens](work-repair-experiment-20261012/wording-history/c7.actual.json) incluyen todos los fallos. Se conservan [revisión primaria](work-repair-experiment-20261012/wording-history/c7.primary-review.json), [revisión cruzada](work-repair-experiment-20261012/wording-history/c7.cross-review.json) y [matriz de comparación con discrepancias](work-repair-experiment-20261012/wording-history/c7.comparison-matrix.json). Ambas son revisiones de IA del proyecto, sin cegamiento; no se presentan como evaluaciones humanas independientes.

### Qué respuestas localizan el problema

Con «La taza es violeta. La botella es verde», tanto preguntar por la taza como por la botella produce, en la redacción simple, **«La taza es verde.»**. Al invertir sólo el orden de los hechos, ambas preguntas producen **«La botella es violeta.»**. En esos cuatro casos combina la primera entidad con el último color, sin conservar correctamente la relación solicitada. Una paráfrasis con los hechos invertidos sí produce **«La taza es violeta.»**. Es una observación de un mundo sintético pequeño, no una demostración de que la arquitectura de copia cause el fallo.

Esta conversación completa se generó con historial real; no se corrigieron las respuestas:

| Turno | Usuario | Unidad C7 |
| --- | --- | --- |
| 1 | El paraguas es de Elisa y la linterna es de Raúl. ¿De quién es cada cosa? | Surgió y la linterna. |
| 2 | Corrijo sólo el paraguas: es de Noelia. ¿De quién queda la linterna? | Queda de Elisa. |
| 3 | ¿De quién es ahora el paraguas? | El paraguas: es de Noelia. |
| 4 | Resume las dos pertenencias actuales en una frase. | ¿Qué quieres contar el paraguas. |

La historia ideal tampoco rescata la pertenencia de la linterna: en su turno 2 C7 responde **«Es de Elisa.»**. En otra conversación sí ayuda una vez: tras indicar que Mateo fabricó una caja amarilla, el historial real produce **«La hizo amarilla.»**, y el ideal **«La fabricó Mateo.»**. Después de corregir sólo el color a morado, ambas ramas vuelven a fallar al conservar al fabricante. La historia ideal añade una respuesta completa; no logra ningún diálogo completo.

Por habilidad, las 48 consultas individuales dan en la primaria **1/16 completas al recuperar hechos explícitos, 0/16 al resolver relaciones y referentes, y 0/16 al producir lenguaje siguiendo instrucciones**. La cruzada coincide en las completas. Algunas respuestas son gramaticales pero falsas; otras fallan además en sintaxis y pertinencia. No queda acreditada una explicación única basada sólo en redacción o sólo en acumulación del historial.

El antiguo contraste de ocho paráfrasis pertenecía a `pointer-v1-best`, no a C7: pasó de **7/8 a 1/8 respuestas semánticamente completas**, sin historial anterior. El diagnóstico nuevo de C7 no reproduce un desplome comparable entre dos bases buenas: sus redacciones simples ya fallan casi siempre. Los dos resultados se conservan y no se mezclan.

## Intervención ejecutada: contrastes relacionales en los datos

Los resultados respaldan probar **una intervención en los datos: equilibrar orden de hechos, entidad preguntada y atributo corregido o conservado**. Se prepararon contrastes en cuatro familias del TRAIN existente: nombres y apelativos, lugar actual/futuro, dos personas y dos objetos. Se mantienen pesos iniciales C7, motor, arquitectura, tokenizador, objetivos y condiciones de generación. Los mundos reservados, las nuevas formulaciones/composiciones de validación y las pruebas de desarrollo se excluyen del entrenamiento. No se usan las respuestas de este diagnóstico como objetivos de TRAIN.

Cada brazo conserva 4.224 registros TRAIN y exactamente los mismos 80 de la nueva validación. Los 2.816 registros de las otras ocho familias permanecen byte a byte iguales. Se revisaron 40 diálogos representativos y sus 160 respuestas objetivo, además de los caminos del compilador: son revisiones de IA del proyecto, no una revisión manual exhaustiva de los 352 diálogos modificados. Se conserva el [congelado de datos](work-repair-experiment-20261012/relational-data/freeze-manifest.json), junto a [comprobación exacta del par](work-repair-experiment-20261012/relational-data/exact-pair-audit.json) y revisiones [de relaciones](work-repair-experiment-20261012/relational-data/peer-review-relations.json), [de objetos](work-repair-experiment-20261012/relational-data/peer-review-objects.json) y [su aclaración](work-repair-experiment-20261012/relational-data/peer-review-objects-addendum.json).

El [diseño preregistrado](work-repair-experiment-20261012/relational-experiment/design.json), SHA-256 `bdffd461423d68b8c507b0d156e6d98d5483b8d6565e161a4e511e0dd1d02f98`, fija 128 actualizaciones máximas por brazo, límite del bucle de 600 segundos, semilla 61, lote 8, aprendizaje 0,0001, BF16 en CPU con cuatro hilos, objetivo de última respuesta y auxiliares de copia 0,2/0,2. Ambos parten de C7 con optimizador nuevo. Se ejecuta un solo entrenador a la vez y se conservan checkpoints 0/64/128; la validación y el guardado final pueden exceder el límite del bucle. No se amplía automáticamente.

El control original completó 128 pasos en **40,554 segundos**, y el brazo relacional en **41,299 segundos**, con las entradas congeladas intactas. La [auditoría de actualización y exposición](work-repair-experiment-20261012/relational-experiment/actual-training-integrity-and-dose.json) verifica que ambos checkpoints iniciales coinciden tensor a tensor con C7, que cambiaron sus 80 tensores y que cada exportación coincide exactamente con su último checkpoint nativo. Configuración y tokenizador conservan sus bytes C7. Se conservan [cierre original](work-repair-experiment-20261012/relational-experiment/original/completion.json), [cierre relacional](work-repair-experiment-20261012/relational-experiment/relational/completion.json) y sus registros de lanzamiento y entrenamiento.

| Candidato experimental, no promovido | ID del modelo | SHA-256 de pesos |
| --- | --- | --- |
| Original, paso 128 | `unidad-conversation-0f42de7d8a5e542deb4a42ef4cfe0ddbaec91098189214478431b777b29c0979` | `0ae17e41e812b43babe7facd4eac1cc4dde0541fd4b86ced070d777934c05aea` |
| Relacional, paso 128 | `unidad-conversation-22c606b1d3bf73a6f7bdc2e44001d29eeb4977fe6909927582e9268b056ad21b` | `448cd17dbb6899d181a43553c45434efcc18f19ae1494118deaba9adc128eaca` |

El preregistro exige, antes de ampliar, al menos **5 completas de los 64 turnos reales**, al menos **tres más que el control igualado**, con ganancias en dos mundos y dos categorías de tarea, sin aumentar críticas respecto a C7 ni al control. Además, en los 40 turnos conocidos exige **al menos 7 completas, como máximo 28 fallidas, 6 críticas y cero errores técnicos**, e identificar cualquier pérdida de un acierto anterior. Son condiciones para estudiar una continuación, no la aceptación general ni la reducción del 50 %. La validación nueva aporta evidencia secundaria de transferencia. Si falla este filtro, se detiene esta intervención y se conserva C7.

El objetivo nativo enmascara las respuestas anteriores del asistente y supervisa sólo la última, incluido EOS. El total efectivo de una época es **48.434 tokens objetivo en original y 48.487 en relacional (+53)**. Los valores 104.519/104.872 del preregistro describen una exposición hipotética si se supervisaran todas las respuestas; **no son los tokens optimizados por este objetivo**. Se conserva [la aclaración sin cambiar el diseño congelado](work-repair-experiment-20261012/relational-experiment/supervision-dose-clarification.json). La auditoría reconstruyó desde el orden y cursor guardados los **mismos 1.024 registros muestreados, sin repeticiones**, y contó **11.679 objetivos efectivos en original frente a 11.672 en relacional (−7)**. Igual número de pasos y registros no significa idéntico número de tokens.

Se registraron tres errores previos de los ayudantes de ejecución: un literal incorrecto del hash esperado del manifiesto detuvo la primera evaluación antes de cargar el motor o generar texto; dos borradores de auditoría suponían una ruta de checkpoint ya podada y una ubicación incorrecta del dato de linaje. Se corrigieron leyendo los artefactos reales preservados. **No cambiaron los datos, pesos, configuración ni diseño para corregirlos**. El [recibo de estas correcciones](work-repair-experiment-20261012/relational-experiment/execution-preflight-corrections.json) conserva sus efectos y límites.

La evidencia anterior ya demuestra que el sistema puede memorizar: los controles de 16 ejemplos, desde C7 y desde pesos aleatorios comparables, consiguieron 16/16 respuestas exactas de TRAIN a 64 pasos. No se repitieron. Tampoco se promociona el candidato anterior con pérdida 0,053719: produjo sólo cinco respuestas completas y once críticas en 40 turnos.

## Resultado de conservación en los 40 turnos conocidos

La [matriz definitiva de la revisión primaria](work-repair-experiment-20261012/relational-results/matrix-retention-known40.json) compara las generaciones con el mismo motor y condiciones **160 / 0 / 42**, conservando historial real. La rúbrica conocida sigue intacta. La revisión pertenece a una IA del proyecto; no es una revisión humana independiente. Se conserva también la [revisión cruzada](work-repair-experiment-20261012/relational-results/known40-cross-matrix.json), sin sustituir las etiquetas primarias.

| Modelo | Completas | Parciales | Fallidas | Críticas | Diálogos completos | Aciertos previos C7 conservados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C7 | 7 | 5 | 28 | 6 | 0/10 | 7/7 |
| Original, paso 128 | 8 | 1 | 31 | 7 | 0/10 | 3/7 |
| Relacional, paso 128 | 6 | 2 | 32 | 9 | 0/10 | 2/7 |

La revisión cruzada dio original **9 completas, 8 parciales, 23 fallidas y 7 críticas**; relacional **7 completas, 6 parciales, 27 fallidas y 8 críticas**. También obtuvo cero diálogos completos. Hay discrepancias importantes al interpretar fragmentos gramaticalmente defectuosos como parciales o fallidos, y una discrepancia crítica en relacional. Ambas revisiones están disponibles por turno; no se eligió la más favorable. Incluso la cruzada incumple la meta de 14 fallidas y el máximo de seis críticas. Es revisión de otra IA del mismo proyecto, no evaluación humana independiente; no consultó la primaria, pero sí las órdenes obligatorias con agregados históricos.

Original añade cinco aciertos y pierde cuatro anteriores; relacional añade cuatro y pierde cinco. **Ambos aumentan respuestas fallidas y críticas respecto a C7 y rechazan el filtro de conservación**, la meta parcial y la aceptación general. Relacional tampoco mejora al control igualado en esta batería: dos completas menos, una fallida más y dos críticas más. No se anunció una reducción del 50 %.

Por ejemplo, donde C7 había respondido **«Quieres dibujar un faro.»**, relacional responde **«Tehora te llamaré Zelomara.»**; el caso se vuelve fallido y crítico. Donde C7 había invitado **«¿Qué quieres contarme sobre eso?»**, ambos nuevos responden **«¿Quién pesado?»**. Las calificaciones conservan también los aciertos nuevos y los desacuerdos de habilidades; no se recortaron las respuestas para ocultar deterioro.

Original termina 39 respuestas por EOS y una al alcanzar 160 tokens, con una repetición larga de «color»; relacional termina las 40 por EOS. Ninguno tuvo respuestas vacías, recortes de historial ni errores técnicos. No se atribuye el deterioro a un recorte oculto, ni se diagnostica automáticamente olvido catastrófico sólo con estos resultados conversacionales.

Se conservan las conversaciones de [original, JSON](work-repair-experiment-20261012/relational-results/original128-known40.actual.json) y [texto](work-repair-experiment-20261012/relational-results/original128-known40.actual.txt), y de [relacional, JSON](work-repair-experiment-20261012/relational-results/relational128-known40.actual.json) y [texto](work-repair-experiment-20261012/relational-results/relational128-known40.actual.txt), con sus [revisiones original](work-repair-experiment-20261012/relational-results/original128-known40.primary-review.json) y [relacional](work-repair-experiment-20261012/relational-results/relational128-known40.primary-review.json).

## Diagnóstico de transferencia posterior: resultados completos

Los dos candidatos generaron los mismos 80 casos factoriales/de historial que C7. Se conservan [original, JSON](work-repair-experiment-20261012/relational-results/original128-factorial.actual.json) y [texto](work-repair-experiment-20261012/relational-results/original128-factorial.actual.txt), y [relacional, JSON](work-repair-experiment-20261012/relational-results/relational128-factorial.actual.json) y [texto](work-repair-experiment-20261012/relational-results/relational128-factorial.actual.txt). Las 16 respuestas con historia ideal de cada candidato siguen separadas de las 64 obtenidas sin esa ayuda.

Además, los tres modelos generaron 20 conversaciones reales de cuatro turnos derivadas de la nueva validación. El [congelado de evaluación](work-repair-experiment-20261012/relational-results/heldout-val80-freeze.json) se hizo después de entrenar, **antes de cualquier respuesta de esta batería**, y no se presenta como preregistro anterior al entrenamiento. Se mantuvieron aparte la [rúbrica](work-repair-experiment-20261012/relational-results/heldout-val80-rubric.json), [casos](work-repair-experiment-20261012/relational-results/heldout-val80-cases.json) y [conversión](work-repair-experiment-20261012/relational-results/heldout-val80-conversion.json). El corpus de validación original permanece intacto.

Antes de generar se aclararon dos mensajes sólo en esos casos derivados: «identificarlo» se convirtió en «identificar su color», y «El proyecto pasará a llamarse Lince» en «El proyecto ha pasado a llamarse Lince», para evitar interpretar otro referente o un cambio todavía futuro. Los textos de origen y efectivos y sus hashes se conservan; no se movieron registros ni se actualizaron objetivos de TRAIN.

Las 80 respuestas completas de cada modelo están disponibles: [C7, JSON](work-repair-experiment-20261012/relational-results/c7-heldout-val80.actual.json) y [texto](work-repair-experiment-20261012/relational-results/c7-heldout-val80.actual.txt); [original, JSON](work-repair-experiment-20261012/relational-results/original128-heldout-val80.actual.json) y [texto](work-repair-experiment-20261012/relational-results/original128-heldout-val80.actual.txt); [relacional, JSON](work-repair-experiment-20261012/relational-results/relational128-heldout-val80.actual.json) y [texto](work-repair-experiment-20261012/relational-results/relational128-heldout-val80.actual.txt).

La [matriz factorial definitiva](work-repair-experiment-20261012/relational-results/factorial-candidate-comparison-matrix.json) conserva primaria, cruce y todos los desacuerdos:

| Modelo | 64 turnos reales C/P/F | Críticas | Diálogos completos reales | 16 turnos con historia ideal C/P/F | Críticas ideales primaria/cruce |
| --- | --- | ---: | ---: | --- | --- |
| C7 | 2/3/59 | 10 | 0/4 | 2/2/12 | 1/1 |
| Original128 | 4/11/49 | 8 | 0/4 | 2/5/9 | 2/2 |
| Relacional128 | 5/7/52 | 13 | 0/4 | 0/2/14 | 3/2 |

Las primarias y cruces coinciden en las completas de los candidatos. El cruce de C7 clasifica dos fallidas como parciales. En historia ideal relacional se conserva el desacuerdo sobre «EsCelia»; el resto de sus fallos no se vuelve éxito por recibir respuestas ideales anteriores. Ambos candidatos conservan sólo uno de los dos aciertos reales de C7 de esta batería. Relacional añade **una completa frente al control**, no las tres exigidas, y aumenta las críticas de ocho a trece. Dos de sus completas individuales son la misma extracción bajo dos órdenes; no equivalen a dos habilidades generalizadas.

La [matriz de la validación adicional](work-repair-experiment-20261012/relational-results/heldout-val80-primary-matrix.json) contiene sus tres revisiones primarias de IA y razones en los 240 turnos:

| Modelo | Completas | Parciales | Fallidas | Críticas | Diálogos completos |
| --- | ---: | ---: | ---: | ---: | ---: |
| C7 | 3 | 15 | 62 | 14 | 0/20 |
| Original128 | 8 | 15 | 57 | 17 | 0/20 |
| Relacional128 | 8 | 18 | 54 | 17 | 0/20 |

En las 48 preguntas con nuevas formulaciones, C7/original/relacional consiguen 3/6/7 completas; en los 32 turnos de nuevas composiciones, 0/2/1. Relacional no añade completas totales frente al control. Los tres terminan las 80 por EOS, sin recortes ni fallos técnicos. Esta rúbrica nueva considera toda contradicción factual interpretable y toda negación de un hecho conocido: **sus críticas no se comparan directamente con la rúbrica estrecha del factorial**. Esta batería tiene una revisión primaria de IA del proyecto; no una evaluación humana independiente ni una nueva aceptación FINAL.

La pérdida de validación original pasó de 3,039334 a 2,024251; la relacional, del mismo punto a 1,975823. Ninguna de estas bajadas acredita conversación. La [auditoría de exclusión efectiva](work-repair-experiment-20261012/relational-results/effective-user-exclusion-audit.json) registró cero coincidencias exactas, de formulaciones completas y de prefijos entre los 80 USER efectivos y ambos TRAIN. No demuestra independencia semántica absoluta. El [metadato final](work-repair-experiment-20261012/relational-results/heldout-val80-final-metadata.json) y [su aclaración](work-repair-experiment-20261012/relational-results/effective-user-loss-input-clarification.json) precisan que el corpus fuente de 80 registros se usó para pérdida de VALIDATION, nunca para actualizaciones TRAIN; los dos USER aclarados se derivaron después de entrenar y no fueron esos mismos inputs de pérdida. Se conservaron los datos fuente intactos.

La [decisión registrada](work-repair-experiment-20261012/relational-experiment/selection-decision.json), SHA-256 `b19f55ec655fe406adc75db4da625898f40a16c5d4283fbcb85fe31bfbdf748b`, detiene esta intervención: mejora puntual insuficiente, regresión de retención y aumento de críticas. Se conserva C7 sin promover candidatos. Estos resultados no prueban que ninguna intervención futura pueda funcionar. Sus cantidades no se suman a las de los 40 turnos conocidos para simular una aceptación. **No se abrió una evaluación FINAL nueva para aprobar una versión rechazada en desarrollo.**

## Trabajo concurrente descubierto en el PR 2

Una lectura del estado remoto a las 19:02 UTC encontró la continuación [PR 2](https://github.com/guadalupejuan001-code/UNIDAD/pull/2), rama `work/generacion-posicion-final-20261012`, commit `cabc203b627e1c486a9fc423cbc1d5c461944d51`, también basada en `a1a283c`. Se conserva [el recibo de descubrimiento y los cuerpos exactos de ambos PR](work-repair-experiment-20261012/concurrent-pr2-discovery.json).

**Los resultados siguientes son los publicados por ese PR, no una ejecución ni restauración de sus pesos en esta comparación.** Allí se informa una optimización de proyección sólo en la última posición, mediciones ABBA de aproximadamente 2× y reproducción exacta de las 40 respuestas C7. La etapa distinta de redacción cerró a 679 pasos; un candidato de 448 obtuvo 13 completas, 12 parciales, 15 fallidas, seis críticas y dos diálogos completos en desarrollo. En otra prueba nueva de 40 turnos, informó C7 1/4/35 con nueve críticas y candidato 3/5/32 con quince, sin diálogos completos. El PR rechaza su promoción y tampoco declara reducción del 50 %.

Su archivo `Unidad-Prueba-Propia.zip`, SHA-256 `4a68aa2cff49c6f3734d11f9946e05ca1acac2440c223bf3af1a687ea678828a`, tampoco está restaurado aquí. No se aplicaron su código, corpus ni pesos en la rama `8b466cc` de esta comparación. Las pruebas nuevas de ese PR son casos diferentes: no se combinan con nuestra batería ni con sus calificaciones. Se preservan las otras ramas y el worktree original. No se afirma haber inspeccionado procesos del otro entorno.

## Conservación y prueba de la referencia

El archivo aportado por el usuario, `Unidad-Pesos-Diagnostico.zip`, con SHA-256 `4ef4754937306157238a4027b4551f09b94949489c66224faa9b6f690c18cacc`, no está disponible en este entorno. **No se ha verificado ni restaurado aquí**; el recibo del PR describe un archivo de otro entorno. Los pesos de C7 sí están disponibles y tienen el hash indicado arriba. Los archivos experimentales conservados localmente no se presentan como sustitutos de ese ZIP.

Esta consola carga la referencia experimental mediante el mismo motor; fue comprobada con historial real y no acredita que C7 converse bien:

```bash
cd /workspace/unidad-conversation-repair
PYTHONPATH=src /workspace/unidad-training-env/bin/python -m ia_propia.conversation converse \
  --model models/conversation-reference-c7 --max-new-tokens 160 --temperature 0 --seed 42
```

Se comprobó esta consola en el worktree activo con los cuatro turnos del paraguas y la linterna: [entrada](work-repair-experiment-20261012/reference/cli-active-worktree/input.txt), [salida literal](work-repair-experiment-20261012/reference/cli-active-worktree/output.txt) y [verificación del modelo e historial](work-repair-experiment-20261012/reference/cli-active-worktree/verification.json). Sus cuatro respuestas coinciden exactamente con las del diagnóstico, sin errores en stderr.

También se conserva, como evidencia anterior del worktree original, [entrada](work-repair-experiment-20261012/reference/cli-c7-input.txt), [salida](work-repair-experiment-20261012/reference/cli-c7-output.txt) y [verificación](work-repair-experiment-20261012/reference/cli-c7-verification.json) de otra prueba de tres turnos. En ella falla con **«Ra: losRa: losRa:.»**. Una consola funcional no convierte esas respuestas en conversación competente.

El objetivo conversacional sigue pendiente. La entrega de esta etapa son las correcciones comprobadas, el diagnóstico con respuestas reales completas y la preservación de una referencia identificada. El [índice de evidencia](work-repair-experiment-20261012/evidence-index.json) registra hashes y rutas de origen de las copias públicas sin editar sus bytes.

## Paquete de pesos y estados entregado

`Unidad-Contraste-Relacional-Experimental.zip` contiene C7, pesos/configuración/tokenizador/manifiesto de ambos endpoints128 y sus alias de validación, los ocho checkpoints completos observados (0,64,best128,last128 por brazo), optimizador, RNG, orden y cursor; código, corpus sintético, procedencia, congelados y respuestas/revisiones completas. No reemplaza el ZIP ausente aportado por el usuario. Conserva los archivos anteriores.

SHA-256: `5978cf805f005ed3692117d9584ca7b515dc369458292289eaf42cb768619e88`; tamaño: **339.107.857 bytes**. El [recibo del paquete](work-repair-experiment-20261012/archive-receipt.json) verificó ZIP, 290 objetos únicos y 370 archivos reconstruidos en un directorio nuevo, con todos los hashes correctos. Además, la consola se ejecutó desde esa copia restaurada: cargó exactamente C7 y reprodujo las cuatro respuestas anteriores con historial real. Se conservan [salida](work-repair-experiment-20261012/reference/restored-cli-c7/output.txt), [stderr vacío](work-repair-experiment-20261012/reference/restored-cli-c7/stderr.txt) y [verificación](work-repair-experiment-20261012/reference/restored-cli-c7/verification.json). Los 14 archivos del motor restaurado coinciden byte a byte con los evaluados. El código y evidencia publicados después son la entrega Git complementaria; el ZIP conserva el estado de código y evidencia de su creación y no se vuelve a escribir para incrustar su propio hash.

Para restaurarlo, extrae `restaurar.py`, lee las instrucciones incluidas y usa un directorio nuevo:

```bash
python restaurar.py Unidad-Contraste-Relacional-Experimental.zip unidad-restaurada
cd unidad-restaurada
python -m pip install -e '.[conversation]'
PYTHONPATH=src python -m ia_propia.conversation converse --model models/conversation-reference-c7 --max-new-tokens 160 --temperature 0 --seed 42
```

El restaurador no inicia entrenamiento. Ambos runs alcanzaron su límite congelado de128; conservar un estado reanudable no autoriza ampliarlo automáticamente. Las rutas absolutas históricas permanecen intactas para no falsificar procedencia. La referencia sigue teniendo errores de idioma, relaciones, correcciones e instrucciones. La conversación cotidiana competente **no está conseguida**.

## Pesos subidos a GitHub

A petición del usuario se subió el paquete completo a [artifacts/unidad-contraste-relacional-20261012](https://github.com/guadalupejuan001-code/UNIDAD/tree/artifacts/unidad-contraste-relacional-20261012/weights-packages/unidad-contraste-relacional-20261012), commit `9747b207a078a4c0c51cc2d3492b9179f050a900`. GitHub recibió seis partes consecutivas del ZIP, su manifiesto, hash, recibo y script de reconstrucción. Cada parte tiene como máximo 64 MiB, por debajo del límite de archivos de GitHub. Los 11 hashes de objetos Git y tamaños remotos coinciden con los locales.

La reconstrucción se ejecutó y conservó exactamente el SHA-256 original `5978cf805f005ed3692117d9584ca7b515dc369458292289eaf42cb768619e88`. No se alteraron pesos, checkpoint ni código del modelo. El [recibo de subida](work-repair-experiment-20261012/github-weights-upload-receipt.json) registra la verificación y que main sigue intacta. El endpoint de adjuntos de release rechazó la credencial con 401; se usó la vía Git autenticada disponible y se retiró la release vacía creada por este intento.

Desde un clon de la rama de artefactos:

```bash
python weights-packages/unidad-contraste-relacional-20261012/reconstruir.py
```

Obtendrás el ZIP completo para usar el restaurador descrito arriba. Esta subida conserva modelos experimentales; no los promueve ni inicia entrenamiento.

## Coherencia actualizada después de la entrega

La auditoría posterior actualizó [el estado actual de main](https://github.com/guadalupejuan001-code/UNIDAD/blob/main/docs/ESTADO_ACTUAL.md), README, órdenes, guía, índice de modelos y cuerpos de los cuatro PR. La corrección comprobada del evaluador de 40 turnos se incorporó a main sin fusionar los cambios experimentales del motor o exportador. Los commits main a1 y upload 9747 citados en recibos fechados describen sus observaciones anteriores; no son una promesa de que esas ramas permanezcan siempre en aquellos commits.

Se verificaron 131 copias exactas de evidencia, 480 respuestas nuevas y 720 filas de revisión, sin corregir las respuestas ni sus etiquetas. Los 44 archivos de once modelos publicados coinciden con los manifiestos y GitHub. C7 ya tenía sus pesos de inferencia en main antes de subir el paquete relacional: la afirmación global de que los pesos no estaban en GitHub era imprecisa. Lo subido después añade los nuevos candidatos y checkpoints completos. La revisión administrativa no cambia la selección, las métricas ni los hashes de los modelos.

La descripción genérica del repositorio sigue siendo verdadera; GitHub rechazó su edición con 403 «Resource not accessible by integration». Esta limitación no impidió actualizar los archivos y cuerpos de PR. El [recibo de metadatos](github-coherence-audit-20261012/pr-metadata-receipt.json) conserva los resultados. No se reinició entrenamiento ni se declaró aceptada la conversación.

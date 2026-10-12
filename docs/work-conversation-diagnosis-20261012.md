# Diagnóstico y reparación de evaluación — 12 de octubre de 2026

> Informe histórico del experimento indicado abajo. La corrección del evaluador se incorporó después a main; el estado vigente y la disponibilidad de artefactos están en [ESTADO_ACTUAL.md](ESTADO_ACTUAL.md). Las métricas, respuestas, hashes y presupuestos de este informe no se reinterpretan como procesos activos.

**No se consiguió reducir a la mitad los fallos de conversación. Se conserva C7 como referencia.** Ningún candidato de este experimento se recomienda para promoción. Las corridas están terminadas; no hay entrenamiento pendiente en segundo plano.

Se trabajó sobre `b5fc677dfeb90b5e9cbd50f26cf878fad6a79d6a`, con una copia independiente. La captura aportada por el creador mostró otra sesión de Codex evaluando ajustes de textos de 256/512 pasos. Se cambió el experimento de Work para investigar el mecanismo de copia. No se inspeccionó ni modificó la carpeta privada de esa otra sesión.

## Resultado medido

La referencia C7 reprodujo exactamente sus **40 respuestas archivadas**. Su SHA-256 de pesos es `666527de240a7fc9b082f233699a1c808c6bdd8f929651824ce4975377c7c836`.

Se fijó como objetivo pasar de **28 F a un máximo de 14 F sobre los mismos 40 turnos**, sin superar los 6 fallos críticos de la revisión primaria. La aceptación general del proyecto sigue siendo 36 respuestas completas, 8 diálogos completos y cero fallos críticos.

| Modelo | Pasos nuevos | Completas C | Parciales P | Fallidas F | Fallos críticos | Diálogos completos |
|---|---:|---:|---:|---:|---:|---:|
| C7 original | 0 | 7 | 5 | 28 | 6 | 0/10 |
| Copia original, comparación igualada | 128 | 7 | 4 | 29 | 7 | 0/10 |
| Clave de copia modificada, comparación igualada | 128 | 1 | 5 | 34 | 5 | 0/10 |
| Copia original, seleccionado por validación | 254 | 5 | 10 | 25 | 11 | 0/10 |
| Clave modificada, seleccionado por validación | 238 | 2 | 9 | 29 | 8 | 0/10 |

La variante de 254 pasos reduce F un 10,7%, pero produce menos respuestas completas y más fallos críticos. Ese resultado **no constituye una mejora aprobable**. Los extremos de 254 y 238 pasos tienen presupuestos efectivos distintos y no se usan como comparación causal de arquitectura.

En otra prueba de 20 turnos, congelada antes del entrenamiento y con tareas de transferencia, C7 y ambos candidatos finales obtuvieron **0 respuestas completamente correctas de 20**. Esta prueba no se utilizó para entrenar ni seleccionar checkpoints; tampoco reemplaza una aceptación final independiente.

## Qué se comprobó

- Un control pequeño pasó de **0/16 a 16/16 respuestas exactas** sobre sus propios ejemplos de entrenamiento. El ajuste puede aprender y memorizar; esta cifra no acredita generalización.
- En la variante original, la pérdida de validación pasó de **2,741646 a 0,053719**. Aun así, las respuestas completas de desarrollo bajaron de 7 a 5. La pérdida con respuestas ideales no predijo la calidad con historia realmente generada.
- Las familias ficticias se separaron por mundo completo: 4.224 registros TRAIN y 384 VALIDATION. No se reutilizaron los mensajes personales ni las preguntas exactas de las pruebas. Las formulaciones del generador sí se comparten entre los mundos de entrenamiento y validación: esta validación comprueba transferencia entre valores dentro de plantillas conocidas, no conversación abierta.
- El tokenizador conserva los textos examinados y el modelo original reproduce los archivos publicados. No apareció evidencia de que una corrupción de pesos o una diferencia gruesa de ejecución explicara estos fallos.

**Diagnóstico sustentado:** la transferencia a otras formulaciones y la estabilidad al usar respuestas propias siguen fallando. La causa completa no quedó aislada; memorizar más ejemplos o elegir únicamente por menor pérdida puede ocultar este problema.

### Control adicional de inicialización y redacción

Mientras se preparaba la entrega, Codex publicó `a1a283caabf89b3191c8ce64c2a950e9edbb0edf`, con soporte para respuestas largas y órdenes actualizadas. La rama de esta entrega parte de ese commit y conserva sus cambios. El entrenamiento descrito arriba sigue identificado con el código original del experimento.

Se ejecutó además el control solicitado desde **pesos aleatorios**, conservando arquitectura, tokenizador propio, 16 ejemplos, semilla de entrenamiento, optimizador, configuración y 64 pasos. Tanto ese control como el heredado de C7 a los mismos 64 pasos produjeron **16/16 respuestas de TRAIN exactas**. No es una prueba de conversación general.

Una prueba posterior aisló la redacción en ocho primeros turnos de un mundo de validación: mismos pesos, mismos hechos, misma generación y **sin historial de respuestas del asistente**. Las coincidencias exactas fueron **7/8 con las formulaciones del corpus y 0/8 con paráfrasis**. La revisión semántica contó **7/8 y 1/8 respuestas completas**, respectivamente; la invitación a contar lo ocurrido siguió siendo válida. Por tanto, cero coincidencias exactas no significa cero respuestas válidas.

Esto demuestra una fragilidad frente a la formulación en ese control y descarta que todos esos errores necesiten acumulación de historia previa. Es un diagnóstico posterior, no un test final ni un conjunto utilizado para ajustar los pesos. Las [respuestas emparejadas](work-experiment-20261012/wording-probe.json) permiten revisar el juicio.

## Hipótesis del mecanismo de copia

La versión actual usa la representación anterior a la palabra para copiar la palabra siguiente. Se probó una versión optativa cuya clave incluye la propia palabra fuente. Ambos brazos comenzaron con todos los tensores y el tokenizador idénticos; sólo cambió esa alineación. Se mantuvieron corpus, semilla, optimizador inicial, aprendizaje, supervisión y generación.

La variante modificada pasó pruebas de causalidad, restricción a texto USER, gradientes y conservación de artefactos. **Empeoró la conversación y se descarta para promoción.** El [parche experimental](work-experiment-20261012/source-pointer-v2.patch) se conserva como evidencia, separado del código de ejecución predeterminado de esta propuesta.

La literatura sobre [pointer-generator networks](https://aclanthology.org/P17-1099/) motivó investigar la copia de texto fuente. No demuestra que esta modificación concreta mejore Unidad; el resultado local fue negativo.

## Corrección de código propuesta

`examples/run_basic_acceptance.py` seguía exigiendo 30 turnos y no exigía los 8 diálogos enteros correctos. La corrección exige 10 diálogos de 4 turnos, al menos 36 respuestas completas y 8 diálogos completos, y cero fallos críticos o técnicos. Un turno crítico incorrecto tampoco cuenta como respuesta completa.

Además, vincula casos y rúbrica a sus hashes, verifica las historias reales y evita aprobar respuestas vacías o asignadas a otro diálogo. No cambia las condiciones de generación. Las pruebas históricas de 30 turnos se conservan como evidencia histórica; se requiere una suite nueva congelada para la aceptación vigente.

**Validación:** 15 pruebas del evaluador aprobadas. El experimento de copia aprobó 3 pruebas nuevas y 10 existentes. Del módulo de entrenamiento aprobaron 30/31: una prueba de detección de procesos separados falló también en el código original de este entorno, con estado `stale` en vez de `planned/failed`. Los entrenamientos directos sí finalizaron y exportaron artefactos verificables.

## Evidencia y límites

Los [resultados estructurados](work-experiment-20261012/results.json) contienen identidades, hashes, configuraciones, estados y conteos. En esa carpeta se conservan las generaciones reales, revisiones, planes y scripts exactos ejecutados. Los scripts de la corrida asumen una carpeta `unidad-fix` junto a un checkout llamado `UNIDAD`; son registros reproducibles del experimento, no un nuevo comando de producto.

Los dos brazos tenían límite de entrenamiento de 600 segundos y objetivo de 256 pasos. Terminaron a 254 y 238 pasos por tiempo, con una validación y guardado finales que llevaron el tiempo registrado a 621 y 615 segundos. Antes de alcanzar 128 pasos se fijó y conservó una comparación común a ese punto por la velocidad de cómputo observada, sin consultar resultados de generación.

Las revisiones semánticas son de IA del mismo proyecto, no humanas, externas ni ciegas. La referencia conserva su revisión primaria archivada; las variantes se revisaron con la misma rúbrica. Algunas fronteras C/P/F admiten discrepancias, pero ningún resultado se acerca al objetivo de 14 F ni a la aceptación del proyecto.

## Siguiente intervención recomendada

Congelar una validación que separe también formulaciones y composiciones, no sólo nombres o mundos. Comparar recuperación de hechos, aplicación de correcciones y respuesta libre, guardando historia real. Estudiar errores de idioma, referentes y afirmaciones de desconocimiento en el corpus. Elegir una intervención corta mediante ese diagnóstico y exigir mejora en generación antes de ampliar el entrenamiento. Conservar C7 hasta que un candidato mejore respuestas completas y fallos críticos en condiciones comparables.

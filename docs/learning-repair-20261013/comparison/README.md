# Comparación real de desarrollo de los modelos propios

Comparación inicial publicada el 13 de octubre de 2026, 08:23 UTC;
LAST y su revisión cruzada se añaden a las 08:40 UTC, y el tratamiento 1.024
a las09:14 UTC. Los endpoints iniciales dieron0/40 completas y0/10 diálogos.
La continuación español LATAM v3 + retención obtuvo **1/40 completa**,
manteniendo **0/10 diálogos, seis críticas/dos técnicas**. No cumple aceptación.
C7 sigue siendo la referencia propia, también rechazada por conversación.
La pérdida de entrenamiento y los archivos de pesos no aceptan modelos.

| Modelo | Revisión | Completas | Parciales | Fallidas | Diálogos completos | Críticas | Técnicas |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C7, 6.983.937 parámetros | Primaria | 0/40 | 1 | 39 | 0/10 | 6 | 0 |
| C7, mismos pesos | Cruzada | 0/40 | 1 | 39 | 0/10 | 6 | 0 |
| SFT 15.179.520, BEST 4.096 | Primaria | 0/40 | 1 | 39 | 0/10 | 12 | 6 |
| SFT, mismos pesos | Cruzada | 0/40 | 1 | 39 | 0/10 | 12 | 6 |
| SFT 15.179.520, LAST 8.192 | Primaria | 0/40 | 3 | 37 | 0/10 | 11 | 8 |
| LAST, mismos pesos | Cruzada | 0/40 | 3 | 37 | 0/10 | 12 | 8 |
| Natural + LATAM, paso 1.024 | Primaria | 0/40 | 0 | 40 | 0/10 | 1 | 11 |
| Natural + LATAM, mismos pesos | Cruzada | 0/40 | 0 | 40 | 0/10 | 0 | 11 |
| Natural + LATAM, LAST 4.096 | Primaria | 0/40 | 1 | 39 | 0/10 | 14 | 1 |
| Natural + LATAM, LAST 4.096 | Cruzada | 0/40 | 1 | 39 | 0/10 | 14 | 1 |
| Control natural, LAST 4.096 | Primaria | 0/40 | 1 | 39 | 0/10 | 2 | 7 |
| Control natural, LAST 4.096 | Cruzada | 0/40 | 1 | 39 | 0/10 | 2 | 7 |
| ACTION2048 BEST256, antes de base v3 | Primaria | 0/40 | 1 | 39 | 0/10 | 5 | 6 |
| ACTION2048 BEST256, mismos pesos | Cruzada | 0/40 | 3 | 37 | 0/10 | 5 | 6 |
| Español LATAM v3 + retención 1.024 | Primaria | 1/40 | 4 | 35 | 0/10 | 6 | 2 |
| Retención1.024, mismos pesos | Cruzada | 1/40 | 1 | 38 | 0/10 | 6 | 2 |

La aceptación exige 36/40 respuestas completas, 8/10 diálogos completos y cero
fallos críticos o técnicos. Son dos revisiones de **IA del proyecto**, realizadas
por turno; no son evaluaciones humanas independientes. La suite es desarrollo
expuesto, con diez situaciones originales ficticias sin datos personales del
creador. No se abrió ni publicó la suite reservada para preparar esta entrega.

Los modelos contestaron los mismos casos con `ConversationEngine`, temperatura
cero, semilla 42, hasta 160 tokens nuevos y un hilo Torch. Cada diálogo empieza
con historia vacía y conserva sólo las respuestas que produjo ese modelo. Los
modelos tienen distinta base, tamaño, vocabulario y corpus: esta comparación no
aísla causalmente el efecto del SFT. El modelo `072eedd2…` se generó cuando su
paso 4.096 era intermedio; al terminar la etapa en 8.192, ese mismo checkpoint
permaneció BEST por validación. Las salidas conservan el campo original
`reviewStatus=pending`; los juicios posteriores están en archivos separados.

## Evidencia conservada sin editar

El [endpoint ACTION 256 anterior](development-action2048-best256-v1/README.md)
y el [endpoint tras dos etapas causales y retención](development-wide-latam-retention1024-v1/README.md)
conservan 52 copias byteexact cada uno: 40 turnos, outputs/template, ambos juicios y
assessments,freeze y metadatos del artefacto. Cada manifiesto identifica todos
los SHA; los proofs verifican historias realmente generadas, sin descartes,
excepciones ni corrección de respuestas. Comparan la misma rúbrica,
contexto 2.048 y generación 160 tokens/temperatura 0/semilla 42. Primaria y cruzada
se hicieron por separado como IA del proyecto, no panel humano.

La completa posterior es `library_initial_deadline`. Las tres diferencias de
grado posterior son `library_delegate_change`,`library_late_collection` y
`cinema_current_plan`; antes, la cruzada añade dos parciales de invitación.
Los summaries conservan también todas las discrepancias de dimensiones.
Cambian pesos y varias etapas/corpus; no hay atribución causal a una sola
variable ni comparación de pérdidas entre objetivos/corpus distintos.

| Evidencia | C7 | SFT BEST 4.096 |
| --- | --- | --- |
| Respuestas, historial, tokens y contexto efectivo | [outputs.json](development-c7-comparable-v1/outputs.json) | [outputs.json](development-sft-step4096-v1/outputs.json) |
| Revisión primaria completa | [review-primary.json](development-c7-comparable-v1/review-primary.json) | [review-primary.json](development-sft-step4096-v1/review-primary.json) |
| Revisión cruzada completa | [review-cross.json](development-c7-comparable-v1/review-cross.json) | [review-cross.json](development-sft-step4096-v1/review-cross.json) |
| Evaluación primaria | [assessment-primary.json](development-c7-comparable-v1/assessment-primary.json) | [assessment-primary.json](development-sft-step4096-v1/assessment-primary.json) |
| Evaluación cruzada | [assessment-cross.json](development-c7-comparable-v1/assessment-cross.json) | [assessment-cross.json](development-sft-step4096-v1/assessment-cross.json) |
| Integridad primaria | [review-primary-integrity.json](development-c7-comparable-v1/review-primary-integrity.json) | [review-primary-integrity.json](development-sft-step4096-v1/review-primary-integrity.json) |
| Recuento y diagnóstico complementarios | [primary-rubric-derived.json](development-c7-comparable-v1/primary-rubric-derived.json) | [summary-cross.json](development-sft-step4096-v1/summary-cross.json), [integrity-cross.json](development-sft-step4096-v1/integrity-cross.json) |

Se copiaron **97 archivos JSON byte por byte**: las salidas agregadas, 40 archivos
por turno en cada modelo y los juicios y recibos disponibles. `review-template.json`
es una plantilla original, no una revisión realizada. El
[recibo de publicación](publication-integrity.json) identifica origen, destino,
tamaño y ambos SHA-256 para cada archivo. El
[resumen estructurado](comparison-v1.json) conserva todas las discrepancias en
las cuatro dimensiones, los fallos técnicos y los hashes de evaluaciones.
Los [casos, rúbrica y congelación](../development-suite/README.md) siguen intactos.

La [entrega adicional de LAST 8.192](development-sft-last8192-v1/README.md)
conserva otras 51 copias exactas, sus respuestas y ambos juicios. Su
[resumen propio](development-sft-last8192-v1/comparison-last-v1.json) retiene
las tres parciales de biblioteca, las diferencias entre las dos revisiones y
ocho fallos técnicos. BEST sigue siendo 4.096; LAST no se seleccionó por
validación ni se promovió por esta evaluación. La tabla mantiene los resultados
de las tres identidades sin modificar recibos históricos.

La única parcial de C7 es `tap_flow_correction`; la de SFT es
`library_initial_deadline`. Para el mismo primer turno de biblioteca, C7 produjo
«El plazo seCuál es el plazo en la biblioteca…» y SFT produjo literalmente
«El viernes a las seis, pero el viernes a las seis.». La segunda respuesta
recupera el plazo, pero no pasa el criterio completo por su redacción. Las
salidas enlazadas conservan el texto íntegro de ambos modelos.

## Discrepancias y fallos técnicos

En SFT, la revisión primaria marca crítica `outing_activity_revision` y no marca
crítica `cinema_attendee_reduction`; la cruzada toma la decisión inversa. La
primaria interpreta «La inversión de WhatsApp es una herramienta de seguridad»
como una afirmación sin base; la cruzada considera que el fragmento no permite
identificar un hecho crítico. En cine, ambas ven el mismo texto «El cine se
encuentra en la ocho y media vamos en la ocho.»; la cruzada interpreta una hora
contraria al contexto, y la primaria no considera inequívoca esa afirmación.
Ambas suman doce críticas, pero no son los mismos doce turnos. Se conservan
además las otras diferencias de español, relevancia y contexto; no se resolvieron
reescribiendo las respuestas ni imponiendo un único juicio.

SFT agotó los 160 tokens en cuatro turnos y descartó contexto en tres, con un
turno común: **seis turnos técnicos distintos**. Los descartes fueron de 13,
212 y 48 tokens. El descarte de 212 elimina el primer mensaje del usuario; los
otros dos eliminan el comienzo del sistema y mantienen los hechos pertinentes
del usuario. Las primeras desviaciones del paseo ya ocurrieron con historial
íntegro. No se atribuye todo el fallo semántico al contexto. C7 terminó los
cuarenta turnos por EOS, sin texto vacío, excepciones ni descarte de contexto.

Estas respuestas sirven para formular la siguiente hipótesis de desarrollo.
No son corpus de entrenamiento, filtro de corpus, prueba reservada nueva ni
demostración de planificación autónoma o calidad comercial.

## Tratamiento intermedio, paso 1.024

La [nueva entrega de tratamiento](development-natural-latam-step1024-v1/README.md)
añade 47 archivos copiados byte por byte y su recibo, sin tocar las fuentes ni
los recibos anteriores. El tratamiento estaba activo al generar ese snapshot. Ahora tratamiento y
control terminaron en 4.096 cada uno; sus juicios se conservan por separado.
Ningún snapshot es un candidato promovido. La única discrepancia
crítica aplicable es `cinema_initial_total`, «Tus entradas en dos entradas.»:
primaria interpreta cantidad errónea y cruzada un fragmento no inequívoco.
Las críticas inferiores no acreditan mejora ante cero respuestas completas,
cero parciales y once fallos técnicos. El [resumen](development-natural-latam-step1024-v1/comparison-treatment1024-v1.json)
conserva juicios e IDs completos.

## Cierres del tratamiento y control

La [entrega LAST del tratamiento](development-natural-latam-last4096-v1/README.md)
conserva 45 archivos de generación/metadatos y cuatro revisiones/assessments
exactos. Ambas revisiones de IA obtienen 0 completas, 1 parcial, 39 fallidas,
14 críticas y 1 técnica. Se comprobaron los IDs críticos, iguales en ambas;
difieren en español para `cinema_attendee_reduction` y
`outing_one_sentence_final`. La primaria incluye auditoría auxiliar de siete
casos mediante IA, no humanos independientes.

La [entrega LAST del control](development-natural-control-last4096-v1/README.md)
conserva otras 45 copias de generación/metadatos y cuatro de juicios. Ambas
revisiones dan 0 completas, 1 parcial, 39 fallidas, 2 críticas y 7 técnicas,
0/10 diálogos completos. Los IDs críticos son `outing_day_revision` y
`outing_one_sentence_final`, iguales en ambos juicios; se compararon las cuatro
claves booleanas de los cuarenta turnos y no difieren. Los textos y notas
originales siguen completos. Menos críticas no acredita mejor conocimiento
ante las omisiones y cero respuestas completas. BEST del control es el padre
heredado en paso 0, sin aprendizaje nuevo de control; estos resultados son LAST.

## Significados regionales

El [probe completo de cuatro modelos](latam-knowledge-development-probe-v1/README.md)
conserva 92 archivos originales exactos y sesenta respuestas. Todos obtienen
0/12 significados TRAIN y 0/6 familias. El padre presenta incertidumbre en
1/3 VALIDATION; los otros, 0/3. Agotamientos: 8/7/3/8, sin excepciones, vacíos
ni descartes. El mismo autor creó el probe e hizo la primaria, con ese vínculo
declarado; la cruzada es de otro agente de IA. No son humanos independientes.
La clasificación semántica de `val07u` del control difiere y se conserva.
El endpoint del control quedó fijado antes de generar, y TEST/reservados
permanecieron fuera. No se comprobó mejora regional ni aceptación.

## Autonomía y publicación

El [intento real de acciones256](../autonomy-demonstration/action256-own-v1/README.md)
produjo JSON válido con `k.txt` sin permiso, rechazado antes del handler.
La [prueba real con un reintento de lectura](../autonomy-demonstration/action256-own-read-repair-v1/README.md)
conservó esa primera propuesta y dio otra mal formada: trabajo fallido, sin
informe ni fallback. Se observó una mejora parcial de formato en una muestra,
sin aceptar planificación fiable. La base causal amplia v3 está entrenando
desde esos pesos propios con fuente nueva y límites finitos.

PR10 se [fusionó sin borrador](https://github.com/guadalupejuan001-code/UNIDAD/pull/10)
en `62cf4d43780204f8d09dce6715cefcb6110aa310`, con cuatro checks CI aprobados.
El [recibo de publicación](../pr10-publication-verification.json) conserva ese
corte de código, datos y pesos. Cierres, juicios y pruebas posteriores se
preparan como entrega posterior; la integración de bytes no acepta modelos.
Todos los recibos y salidas anteriores permanecen intactos.

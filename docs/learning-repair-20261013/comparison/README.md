# Comparación real de desarrollo: C7 y SFT propio de 15 M

Comparación inicial publicada el 13 de octubre de 2026, 08:23 UTC;
LAST y su revisión cruzada se añaden a las 08:40 UTC, y el tratamiento 1.024
a las 09:14 UTC. **No se comprobó mejora
conversacional**: ninguno de los snapshots publicados obtiene respuestas completas ni
diálogos completos. C7 sigue siendo la referencia propia, también rechazada por
el criterio de conversación. La pérdida de entrenamiento y la existencia de
pesos no cambian esta conclusión.

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
[recibo de publicación](publication-integrity.json) identifica origen, destino,+tamaño y ambos SHA-256 para cada archivo. El
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
los recibos anteriores. El entrenamiento sigue activo, el control aún no se ha
evaluado y este snapshot no es un candidato promovido. La única discrepancia
crítica aplicable es `cinema_initial_total`, «Tus entradas en dos entradas.»:
primaria interpreta cantidad errónea y cruzada un fragmento no inequívoco.
Las críticas inferiores no acreditan mejora ante cero respuestas completas,
cero parciales y once fallos técnicos. El [resumen](development-natural-latam-step1024-v1/comparison-treatment1024-v1.json)
conserva juicios e IDs completos.

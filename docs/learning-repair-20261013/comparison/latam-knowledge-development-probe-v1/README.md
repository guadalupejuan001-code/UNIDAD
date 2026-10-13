# Probe regional: cuatro modelos propios y sesenta respuestas reales

Las revisiones primaria y cruzada observan **cero significados correctos de
doce y cero familias correctas de seis en cada modelo**. No se comprobó mejora
regional del tratamiento frente al padre o al control. Son quince turnos por
modelo: seis paráfrasis TRAIN, seis composiciones TRAIN y tres casos de
incertidumbre de VALIDATION.

| Modelo | Significados TRAIN | Familias con ambos casos correctos | Incertidumbre explícita VALIDATION | Agotamientos de 80 tokens |
| --- | ---: | ---: | ---: | ---: |
| Padre SFT 4.096, `072eedd2…` | 0/12 | 0/6 | 1/3 | 8 |
| Tratamiento BEST 1.024, `ec9ab262…` | 0/12 | 0/6 | 0/3 | 7 |
| Tratamiento LAST 4.096, `d6d86022…` | 0/12 | 0/6 | 0/3 | 3 |
| Control LAST 4.096, `a2cae3b6…` | 0/12 | 0/6 | 0/3 | 8 |

No hubo excepciones, texto vacío ni descarte de contexto. Menos agotamientos no
convierten respuestas irrelevantes o incoherentes en significados aprendidos.
La incertidumbre de VALIDATION es un diagnóstico separado y no cuenta como
conocimiento aprendido de TRAIN. Ninguna revisión acepta ni promueve modelos.

La primaria la realizó **el mismo autor del probe**, mediante IA del proyecto;
su declaración se conserva. La revisión cruzada es de otro agente de IA del
proyecto, separada de la primaria. Ninguna es evaluación humana independiente.
Los juicios y sus métodos originales permanecen completos. En el control,
`val07u` difiere: la primaria lo considera irrelevante/incoherente y la cruzada
una afirmación sin base. La cruzada conserva además estados semánticos y estados
con prioridad técnica; sus clasificaciones no se mezclan con las de la primaria.

## Datos y decisiones conservados

| Evidencia | Padre | BEST tratamiento | LAST tratamiento | LAST control |
| --- | --- | --- | --- | --- |
| Respuestas e historial real | [outputs](parent072/outputs.json) | [outputs](treatment-best1024/outputs.json) | [outputs](treatment-last4096/outputs.json) | [outputs](control-last4096/outputs.json) |
| Primaria | [review](parent072/review-primary.json) | [review](treatment-best1024/review-primary.json) | [review](treatment-last4096/review-primary.json) | [review](control-last4096/review-primary.json) |
| Cruzada | [review](parent072/review-cross.json) | [review](treatment-best1024/review-cross.json) | [review](treatment-last4096/review-cross.json) | [review](control-last4096/review-cross.json) |
| Evaluación primaria | [assessment](parent072/assessment-primary.json) | [assessment](treatment-best1024/assessment-primary.json) | [assessment](treatment-last4096/assessment-primary.json) | [assessment](control-last4096/assessment-primary.json) |
| Evaluación cruzada | [assessment](parent072/assessment-cross.json) | [assessment](treatment-best1024/assessment-cross.json) | [assessment](treatment-last4096/assessment-cross.json) | [assessment](control-last4096/assessment-cross.json) |

Se copiaron **92 archivos byte por byte**, incluidos sesenta archivos por turno,
cuatro outputs agregados, ocho reviews, ocho assessments, tres agregados
originales y el paquete congelado completo. El [recibo de copia](publication-integrity.json)
registra origen, destino, bytes y ambos hashes. El
[resumen verificable](comparison-probe-v1.json) conserva ambos recuentos y las
discrepancias sin imponer consenso. Los agregados originales están en
[primary-review-v1](primary-review-v1/primary-comparison-four-arms-v1.json) y
[aggregate-cross.json](aggregate-cross.json).

El [paquete fuente original](../../latam-knowledge-development-probe-v1/README.md)
permanece intacto y tiene una [copia congelada](frozen-suite/freeze.json), SHA
`8897a6a048d6706eb30989cb4567d8e468f78d5a9191d8bb492dbfda47687f6b`.
Casos, rúbrica, soporte de fuentes, auditoría de novedad y prerregistro se
preservan sin cambios. La [vinculación previa del control](frozen-suite/control-endpoint-binding.json)
fijó LAST 4.096 antes de su generación, no por sus respuestas.

Se usaron temperatura cero, semilla 42, hasta 80 tokens y un hilo Torch. La
historia contiene exclusivamente solicitudes del caso y respuestas previas
reales; no incluye rúbrica ni respuestas ideales en el prompt. Es un probe
pequeño y expuesto de fuentes TRAIN/VALIDATION, sin lectura de TEST ni suite
reservada. No cubre todos los regionalismos ni sustituye la aceptación de
conversación. Esta publicación no ejecutó modelos, entrenamiento o pruebas de
software, y no alteró fuentes, pesos, corpus o archivos congelados.

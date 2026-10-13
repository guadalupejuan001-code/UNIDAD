# Entrenamientos propios del 13 de octubre

Se publican seis selecciones de validación sin alterar sus cuatro archivos nativos. Todos los parámetros fueron entrenables y todos los tensores cambiaron respecto al padre: 76/76 en V7 y 80/80 en C7. El [resumen verificable](summary.json) conserva hashes, corpus, linaje desde pesos aleatorios, configuraciones y medidas.

| Brazo | Pasos realizados | Paso publicado | NLL de validación | Segundos nativos |
| --- | ---: | ---: | ---: | ---: |
| [v7](v7/verification.json) | 256 | 256 | 4.635460 (última) | 72.271 |
| [c7](c7/verification.json) | 256 | 256 | 4.490238 (última) | 107.333 |
| [history-ideal](history-ideal/verification.json) | 512 | 512 | 0.061610 (última) | 120.576 |
| [history-generated](history-generated/verification.json) | 512 | 384 | 0.065061 (última) | 105.919 |
| [natural-v7](natural-v7/verification.json) | 512 | 128 | 5.289259 (última) | 137.900 |
| [natural-c7](natural-c7/verification.json) | 512 | 128 | 4.793762 (última) | 160.875 |

Total: **2560 pasos**, **704.874 segundos** según los seis estados nativos. Es tiempo de las invocaciones, incluyendo preparación y validación; no cuenta autoría, generación de corpus, evaluación libre ni revisión. La NLL de la tabla es la última validación; cada verificación conserva también cuál se seleccionó. No se comparan como una única escala los distintos corpus/reducciones.

Los recibos y logs originales preservan sus rutas locales `/workspace/UNIDAD/private/...`, PID y timestamps históricos. Copiarlos no reinicia ningún proceso. Los modelos siguen experimentales y no superaron la aceptación conversacional. El informe de integración conserva las conversaciones reales y su revisión de IA del proyecto.

Los corpus públicos que reproducen estas etapas son [humano v1](../../../examples/human-short-es-v1/human-short-es.train-validation.jsonl), [historia ideal](../../../examples/on-policy-history-v1/ideal.train-validation.jsonl), [historia generada](../../../examples/on-policy-history-v1/generated.train-validation.jsonl) y [natural de varios turnos](../../../examples/natural-multiturn-v1/train-validation.jsonl). Se conservan sus hashes en los diseños.

Los paquetes de [history-generated](../../../checkpoints/conversation-integration-20261013/history-generated/README.md), [natural-v7](../../../checkpoints/conversation-integration-20261013/natural-v7/README.md) y [natural-c7](../../../checkpoints/conversation-integration-20261013/natural-c7/README.md) incluyen el checkpoint seleccionado y el último, ambos completos con pesos, Adam, programador y estados RNG. Sus guías explican cómo restaurar en un directorio nuevo sin entrenar. No incluyen caches mmap; el entrenador puede reconstruirlas desde el corpus exacto.

[Diagnóstico técnico](diagnosis.md). La atribución de datos y sus declaraciones de licencia se conserva en los README de los corpus y de los modelos ancestrales; no se infiere una licencia de pesos a partir de la del texto.

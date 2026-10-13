# Cerebro propio, NERVA y ajuste parcial: 13 de octubre de 2026

Se incorporó NERVA como backend opcional de cálculo del char-RNN de Unidad y una
política de entrenamiento parcial para su Transformer conversacional. Las dos
capacidades son distintas: el backend NERVA conserva la salida del modelo pequeño;
el experimento conversacional actualiza pesos propios de C7 mediante PyTorch.

## Código comprobado

- [Backend NERVA](nerva.md): el host selecciona el motor, los mensajes no lo cambian.
  Node.js 18+ ejecuta embedding, recurrencia, proyección y generación greedy en
  NERVA `f64`. Se mantienen NumPy por defecto y el `ConversationEngine` común.
  Los 448 caracteres de siete prompts coinciden con la referencia; Node añadió
  coste de arranque. No se atribuye aceleración ni mejora conversacional a ese backend.
- [Entrenamiento parcial](copy-head-training.md): `trainable_parameters: "copy_head"`
  ajusta sólo cuatro tensores. La política, los momentos de AdamW y el SHA del
  backbone congelado se validan al guardar, reanudar y exportar. Se comprobó
  reanudación exacta 2+2 frente a cuatro pasos, en FP32 y BF16. Omitir la opción
  conserva el camino anterior y sus metadatos.
- La suite completa de software pasó 399 pruebas, sin omisiones, además de las
  12 pruebas de ejemplos. Son comprobaciones de código, no aceptación de una IA.
  Se instalaron las dependencias conversacionales y Node; el wheel con los cinco
  recursos NERVA se ejecutó desde un directorio ajeno al checkout.

## Contraste conversacional terminado

Se fijaron 128 pasos por brazo, idéntica semilla, orden de lotes, RNG y datos,
antes de generar respuestas. Los 432 registros TRAIN y 42 VALIDATION conservan
sus IDs, familias, particiones y bytes originales. Se excluyó TEST y se comprobó
que no había colisiones literales de mensajes con el panel de desarrollo.
Esto no acredita que las familias de desarrollo fueran desconocidas para C7.

El brazo completo cambió 6.983.937 parámetros; el parcial cambió 16.641 y mantuvo
los otros 6.967.296 byte a byte. Las medidas de tiempo proceden de un único par
con orden completo/parcial; no establecen una comparación general de rendimiento.
Ambos experimentos terminaron. No se renovaron presupuestos cerrados ni quedan
entrenamientos activos por esta entrega.

| Modelo | Completas / 16 DEV | Críticas | Diálogos completos |
| --- | ---: | ---: | ---: |
| C7 | 1 | 2 | 0/4 |
| Ajuste completo, 128 pasos | 2 | 1 | 0/4 |
| Sólo copia, 128 pasos | 2 | 1 | 0/4 |

Dos revisiones de IA del proyecto coinciden en completas y críticas; se conservan
sus desacuerdos sobre parciales. El cruce ocultó identidades de los brazos y las
calificaciones primarias. No son revisiones humanas independientes.

El ajuste parcial corrige un fallo concreto: ante una corrección del propietario
del paraguas, conserva que **la linterna es de Raúl**, donde C7 respondía «Queda
de Elisa.». Ambos ajustes conservan el acierto anterior sobre Noelia. Siguen
fallando explicación, invitación y otras relaciones. No se demuestra superioridad
conversacional del ajuste parcial frente al completo.

La [comparación de desarrollo](brain-nerva-20261013/development-comparison.json),
el [diseño previo](brain-nerva-20261013/training-design.json), el
[recibo de entrenamiento](brain-nerva-20261013/training-receipt.json) y la
[auditoría pareada](brain-nerva-20261013/paired-native-audit.json) conservan
condiciones, respuestas, hashes, linaje y discrepancias. Se conserva también el
[runner histórico](brain-nerva-20261013/original-runner.py); el
[runner portátil](../examples/experiment_copy_head_retention.py) reproduce los
datos y el procedimiento sin depender de las rutas de este workspace.

## Evaluación nueva después de fijar el candidato

Se eligió el endpoint parcial antes de redactar una prueba nueva de 40 turnos
en diez diálogos. El agente redactor no había consultado modelos, corpus ni
respuestas; sus [casos](brain-nerva-20261013/reserved-cases.json) y
[rúbrica](brain-nerva-20261013/reserved-rubric.json) quedaron congelados antes de
la generación. No se entrenó ni se cambió la selección después de consultarlos.
Se usó historia realmente generada, 160 tokens, temperatura cero y semilla 42.

| Modelo | Completas / 40 nuevas | Críticas | Diálogos completos |
| --- | ---: | ---: | ---: |
| C7 | 0 | 4 | 0/10 |
| Sólo copia, 128 pasos | 0 | 8 | 0/10 |

Las revisiones primaria y cruzada del proyecto coinciden en completas y críticas;
se conservan once desacuerdos parcial/fallida. El cruce ocultó identidades y
notas primarias. No hubo fallos técnicos, vacíos, recortes ni paradas por presupuesto
en los 80 turnos. La [comparación completa](brain-nerva-20261013/reserved-comparison.json)
conserva respuestas, condiciones, hashes e integridad. La prueba nueva no es una
comparación independiente con asistentes comerciales.

**El candidato parcial queda rechazado:** el acierto adicional observado en DEV
no se generalizó y aumentaron los fallos críticos. C7 se conserva como referencia
experimental; tampoco pasa la aceptación. No se atribuye mejora conversacional
general al código de congelación ni a NERVA.

## Pesos y uso

El [candidato parcial](../models/conversation-copy-head-20261013/README.md) contiene
configuración, tokenizador, manifiesto y un delta de 66.876 bytes. Su reconstructor
verifica C7 y produce exactamente los 27.943.524 bytes de pesos evaluados. Se
comprobó la carga por `ConversationEngine` y la consola, sin promoverlo como modelo
predeterminado. El candidato se entrega para reproducción y queda rechazado.
El ajuste completo y sus checkpoints se conservan en el paquete
local del experimento, junto con el candidato parcial y su estado reanudable.

La conversación competitiva sigue pendiente. Las comprobaciones de software,
una respuesta corregida y los tiempos de entrenamiento no acreditan sustitución
de una IA comercial ni adaptación autónoma. El siguiente problema observado es
la base lingüística y la generalización, no la falta de una interfaz.

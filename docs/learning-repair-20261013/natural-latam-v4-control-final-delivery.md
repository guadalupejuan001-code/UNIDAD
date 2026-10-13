# Cierre del control matched-v4

CONTROL terminó en **4.096 actualizaciones del optimizador**, con tiempo acumulado
de **3.321,1108099389967 segundos**. Se comprobó que su trabajador había salido y
que el recibo final del runner estaba completo. El estado conserva
`status=stopped`, `reason=signal` y parada cooperativa `absolute_step_boundary`.
Esta entrega no envió señales ni modificó al entrenador o supervisor.

BEST permanece en el **paso cero**, con NLL de validación
**1,7110848086831454**. Sus pesos son exactamente los del padre **SFT072 BEST del
paso 4.096**, SHA256
`0a71ae3feb61b7dabac9f31ecc4fd312f97ead8a790ed13b8144617b60e3e4e6`.
Es la calidad heredada antes de esta etapa: el BEST seleccionado tiene **cero
actualizaciones y cero exposición nueva CONTROL**. LAST contiene los pesos
realmente ajustados en 4.096 pasos, con SHA256
`7838bd314bc35ed4b21eba31158b9ccab322dec21d298bd22002fb8527139244`.

La etapa creó un optimizador y scheduler nuevos, entrenó los 15.179.520 parámetros
y conservó supervisión de la última respuesta, lote lógico ocho y microbatch
máximo cuatro. Sus particiones son TRAIN 9.951, VALIDATION 1.704 y TEST 0. La
base heredada no se suma a la exposición de esta etapa. El
[recibo principal](natural-latam-v4-control-final-delivery.json) identifica
estado, procedencia, código y todos los archivos nativos. El
[recibo de verificación](natural-latam-v4-control-final-verification.json)
conserva la comprobación de enlaces, archivos y auditoría independiente de IA.

## Exposición ejecutada

El [recibo exacto](natural-latam-v4-control-final-target-exposure.json) y el
[registro por ventana](natural-latam-v4-control-final-window-exposure.csv)
cuentan etiquetas TRAIN con `label >= 0`, incluido EOS, y excluyen `-100`.
Verifican el índice y los bytes congelados, los barajados y el RNG completo.

| Posiciones procesadas | BEST 0 | LAST 4.096 |
| --- | ---: | ---: |
| Ventanas consumidas | 0 | 32.765 |
| Tokens de entrada sin padding | 0 | 8.760.288 |
| Etiquetas supervisadas, incluido EOS | 0 | 4.402.874 |
| Etiquetas EOS | 0 | 32.765 |
| Etiquetas de respuesta, excluido EOS | 0 | 4.370.109 |

LAST comprende tres épocas completas de 9.951 ventanas y las primeras 2.912
ventanas del cuarto orden. Cada época termina con un lote de siete ventanas;
`4096 × 8` sobrecontaría tres ventanas. Estos números describen el control
realmente ejecutado, no la exposición de los ejemplos originales dentro del
tratamiento regional. Miden posiciones procesadas y no conocimiento adquirido.

## BEST y LAST recuperables

[LAST](../../checkpoints/learning-repair-20261013/natural-latam-v4/control-final/LAST/step-000004096-b6bdfcddec62449b82825fd7b9f838e5/)
recompone **182.358.191 bytes nativos** mediante nueve partes de hasta 32 MiB.
Se verificaron sus partes y también los hashes completos recorriendo las partes
en orden. Conserva pesos, Adam/RNG, estado, scheduler, orden y nombres nativos.

[BEST](../../checkpoints/learning-repair-20261013/natural-latam-v4/control-final/BEST-reference/step-000000000-b0277331e4ee4dc0b7c185082d36571b/)
conserva sus cuatro archivos nativos de metadatos, incluido su optimizador inicial,
y una referencia verificada a los pesos propios ya publicados. No se duplicaron
los 60.727.960 bytes del padre. Su referencia permite recomponer exactamente los
cinco archivos nativos, un total de 60.798.946 bytes.

Desde la raíz del repositorio, con destinos nuevos:

```bash
mkdir -p private/control-final-recovery/checkpoints
python3 docs/learning-repair-20261013/prepare_control_final_delivery.py restore-best-reference \
  --reference-dir checkpoints/learning-repair-20261013/natural-latam-v4/control-final/BEST-reference/step-000000000-b0277331e4ee4dc0b7c185082d36571b \
  --destination private/control-final-recovery/checkpoints/step-000000000-b0277331e4ee4dc0b7c185082d36571b
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/natural-latam-v4/control-final/LAST/step-000004096-b6bdfcddec62449b82825fd7b9f838e5 \
  private/control-final-recovery/checkpoints/step-000004096-b6bdfcddec62449b82825fd7b9f838e5
```

Son instrucciones de recomposición, aún no ejecutadas. No cargan un modelo ni
arrancan entrenamiento. Ambos directorios conservan sus nombres originales para
resolver los punteros del estado. La recuperación del entrenador también exige
el corpus con SHA256
`48cc11848568c03a32727660968e47695d8eeb43c3302cbf7055eefe2e449491`,
el código congelado y el diseño. Reponer el estado conserva el límite ya alcanzado
de 4.096 pasos.

La [referencia de inferencia BEST](../../models/learning-repair-20261013/natural-latam-v4/control-final-BEST-reference/artifact-reference.json)
apunta al artefacto completo del padre. Su directorio no se carga directamente
con `ConversationEngine`: debe resolverse `artifactDirectory`. El manifiesto de
selección final CONTROL se conserva por separado sin alterar al padre. La
exportación del artefacto completo de inferencia LAST se registra en otra entrega.

El empaquetador usó biblioteca estándar, una CPU, prioridad `nice +10`, límite
de 512 MiB de espacio de direcciones y 120 segundos por operación. Terminó en
1,326 segundos con 20,7 MB de RSS máximo. No importó Torch, no interpretó pickle,
no entrenó, no ejecutó inferencia o pruebas y no leyó corpus ni conjuntos
reservados o cachés VALIDATION/TEST. La revisión es de IA del proyecto.

La preparación local todavía no verifica una subida a GitHub. El candidato
mantiene `productionAccepted=false`. Selección por pérdida, exposición y archivos
íntegros no acreditan aceptación conversacional.

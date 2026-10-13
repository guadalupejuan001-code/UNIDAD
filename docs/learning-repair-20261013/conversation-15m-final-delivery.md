# Cierre de la etapa SFT propia de 15 millones de parámetros

La etapa llegó a **8.192 actualizaciones del optimizador**. El runner pidió su
parada cooperativa al alcanzar `stopAfterStep=8192`; el estado nativo conserva
`status=stopped` y `reason=signal`, y el recibo del runner registra
`absolute_step_boundary`. El tiempo acumulado final es **5.941,433658156999
segundos**. Se conservaron el entrenamiento y sus checkpoints sin enviar señales
desde esta entrega.

BEST sigue siendo el checkpoint del paso **4.096**, con NLL de validación
**1,7110848086831454**. Los cuatro archivos de su artefacto final son idénticos,
byte por byte, a la entrega intermedia. Se verificaron otra vez tanto esos
archivos como todas las partes de su checkpoint nativo; se reutilizan las
copias existentes. LAST corresponde al paso **8.192** y tiene su propio paquete
nativo nuevo.

| Papel | Archivo entregado |
| --- | --- |
| BEST para inferencia | [Artefacto completo](../../models/learning-repair-20261013/conversation-15m/intermediate-step004096/) |
| Referencia de la selección final | [Puntero con hashes](../../models/learning-repair-20261013/conversation-15m/final-best-step004096/artifact-reference.json) |
| BEST nativo | [Paquete intermedio reutilizado](../../checkpoints/learning-repair-20261013/conversation-15m/intermediate-step004096/step-000004096-37b0b60acf4546d398078b3c407e600d/) |
| LAST nativo | [Paquete final](../../checkpoints/learning-repair-20261013/conversation-15m/final-last-step008192/step-000008192-82822b3e01a7488dbb1141a1174ebb94/) |
| Estado, diseño, configuración y caché | [Copias congeladas](conversation-15m-final-inputs/) |

El [recibo principal](conversation-15m-final-delivery.json), el
[recibo de empaquetado](conversation-15m-final-package.json) y el
[recibo de entradas](conversation-15m-final-inputs-receipt.json) identifican
todos los tamaños, hashes y procedencias. Esta preparación local no verifica
todavía una subida a GitHub.

## Exposición real de entrenamiento

El [recibo exacto](conversation-15m-final-target-exposure.json) deriva los
totales de los estados nativos, el índice TRAIN y sus etiquetas. El
[registro por ventana](conversation-15m-final-window-exposure.csv) conserva
longitudes reales, posiciones supervisadas, EOS y el número de veces que cada
ventana llegó a BEST y a LAST. No incluye el entrenamiento de la base heredada.

| Posiciones procesadas | BEST 4.096 | LAST 8.192 |
| --- | ---: | ---: |
| Ventanas consumidas | 32.764 | 65.524 |
| Tokens de entrada sin padding | 7.841.381 | 15.681.293 |
| Etiquetas supervisadas, incluido EOS | 2.638.841 | 5.286.178 |
| Etiquetas EOS | 32.764 | 65.524 |
| Etiquetas de respuesta, excluido EOS | 2.606.077 | 5.220.654 |

LAST comprende tres épocas completas de 18.588 ventanas y las primeras 9.760
ventanas del cuarto orden guardado. Cada época completa termina con un lote de
cuatro ventanas; por eso `8192 × 8` sobrecontaría doce ventanas. El lote lógico
es ocho y el microbatch máximo cuatro. Los órdenes y el estado completo del RNG
de Python coinciden con la reproducción de sus barajados usando la semilla
20261013. Las etiquetas se contaron con `label >= 0`; `-100` queda excluido.

Estos números miden posiciones procesadas. La pérdida y la exposición no
demuestran conocimiento adquirido ni aceptación conversacional. El artefacto
conserva `productionAccepted=false`. La generación del LAST y las evaluaciones
conversacionales se registran por separado.

## Reponer BEST y LAST

El transporte conserva los cinco archivos originales de cada checkpoint,
incluidos Adam y RNG en `resume.pt`, sin interpretar su pickle. Todas las partes
de ambos paquetes se verificaron por separado y también se recorrieron en orden
para comprobar el hash completo de cada archivo nativo. No se cargó el modelo ni
el optimizador, y no se ejecutó una restauración del entrenador.

Desde la raíz del repositorio, usando un destino nuevo:

```bash
mkdir -p private/conversation-15m-final-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/conversation-15m/intermediate-step004096/step-000004096-37b0b60acf4546d398078b3c407e600d \
  private/conversation-15m-final-recovery/checkpoints/step-000004096-37b0b60acf4546d398078b3c407e600d
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/conversation-15m/final-last-step008192/step-000008192-82822b3e01a7488dbb1141a1174ebb94 \
  private/conversation-15m-final-recovery/checkpoints/step-000008192-82822b3e01a7488dbb1141a1174ebb94
```

Son instrucciones de recomposición de bytes, aún no ejecutadas en esta entrega.
El nombre de cada directorio debe conservarse porque el estado nativo referencia
ambos checkpoints. Una recuperación del entrenador también exige el corpus con
SHA256 `4f51ffdf549439707ddee9ea75127264fdd33bcaadc52d585d82a93e52825510`,
los archivos fuente identificados en el recibo de entradas y las rutas originales
del diseño. El estado final mantiene `max_steps=8192`, su tiempo, scheduler,
optimizador y orden; restaurarlo conserva ese límite ya alcanzado. Una etapa
posterior requiere un diseño explícito propio.

La preparación usó biblioteca estándar, una CPU, hasta 512 MiB de espacio de
direcciones y límites de 120 segundos por operación. No leyó corpus ni conjuntos
reservados, no importó Torch, no entrenó y no ejecutó pruebas de software. La
revisión es de IA del proyecto; no se presenta como revisión humana.

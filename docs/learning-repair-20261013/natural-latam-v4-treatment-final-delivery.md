# Cierre del tratamiento LATAM v4

El tratamiento terminó en **4.096 actualizaciones del optimizador**, tras
**3.337,367299398 segundos**. El runner solicitó su parada cooperativa al alcanzar
el límite absoluto configurado; conserva `status=stopped`, `reason=signal` y
`cooperativeStopReasons=[absolute_step_boundary]`. Esta entrega no envió señales
al entrenador ni al supervisor.

El modelo propio ec9 conserva BEST en el paso **1.024**, con NLL de validación
**1,7603046832825393**. Sus pesos, configuración y tokenizador coinciden byte por
byte con el [artefacto intermedio](../../models/learning-repair-20261013/natural-latam-v4/intermediate-step001024/).
El manifiesto intermedio conserva su selección explícita del checkpoint; el
[manifiesto final exacto](../../models/learning-repair-20261013/natural-latam-v4/final-treatment-best-step001024/manifest.final-selection.json)
registra la selección por validación. Se mantienen ambos, sin duplicar pesos.
La [referencia final](../../models/learning-repair-20261013/natural-latam-v4/final-treatment-best-step001024/artifact-reference.json)
identifica los archivos y sus diferencias de metadatos. Ese directorio de
referencia no se carga directamente con el motor: el campo `artifactDirectory`
apunta al artefacto completo que debe cargarse.

La procedencia sigue siendo **SFT072 BEST del paso 4.096**, con pesos
`0a71ae3feb61b7dabac9f31ecc4fd312f97ead8a790ed13b8144617b60e3e4e6`.
Esta etapa creó un optimizador y scheduler nuevos y entrenó los 15.179.520
parámetros. Conserva TRAIN 10.359, VALIDATION 1.759 y TEST 0, con supervisión
exclusiva de la última respuesta. No se suma el entrenamiento heredado a las
cifras siguientes.

## Exposición ejecutada

El [recibo exacto](natural-latam-v4-treatment-final-target-exposure.json) y el
[registro por ventana](natural-latam-v4-treatment-final-window-exposure.csv)
derivan sus números del índice y etiquetas TRAIN, el orden guardado y el estado
nativo. Los barajados y el RNG completo coinciden con su reproducción mediante
la semilla 20261013.

| Posiciones procesadas | BEST 1.024 | LAST 4.096 |
| --- | ---: | ---: |
| Ventanas consumidas | 8.192 | 32.765 |
| Tokens de entrada sin padding | 2.161.348 | 8.609.206 |
| Etiquetas supervisadas, incluido EOS | 1.074.907 | 4.277.235 |
| Etiquetas EOS | 8.192 | 32.765 |
| Etiquetas de respuesta, excluido EOS | 1.066.715 | 4.244.470 |

LAST comprende tres épocas completas de 10.359 ventanas y 1.688 ventanas del
cuarto orden. Cada época acaba con un lote de siete ventanas; `4096 × 8`
sobrecontaría tres ventanas. El lote lógico es ocho y el microbatch máximo cuatro.

| Cohorte dentro del tratamiento | Ventanas LAST | Etiquetas LAST, incluido EOS |
| --- | ---: | ---: |
| Ejemplos del control original | 31.470 | 4.225.047 |
| Ejemplos regionales nuevos | 1.295 | 52.188 |

La correspondencia de ventanas y cohortes se comprobó con IDs y hashes de cada
fila TRAIN de los componentes congelados. Las filas TRAIN del tratamiento
coinciden byte por byte y en orden con el control seguido de los ejemplos
regionales. El [audit nativo previo](natural-latam-v4-train-audit/receipt.json)
conserva la comprobación de una ventana por registro. Estas cifras de la cohorte
original describen su exposición **dentro del tratamiento**; no son resultados
ejecutados por la corrida separada CONTROL.

## Checkpoint recuperable

El [paquete LAST](../../checkpoints/learning-repair-20261013/natural-latam-v4/final-treatment-last-step004096/step-000004096-b39d6d921c7740c48a289e62d715654e/)
recompone **182,360,596 bytes** en nueve partes de como máximo 32 MiB.
Conserva los cinco archivos nativos, incluido `resume.pt` con Adam y RNG, la
configuración, el scheduler, el orden y los punteros BEST/LAST originales. Se
verificaron todas las partes y también los hashes completos recorriendo sus
bytes en orden. No se interpretó el pickle ni se cargó el modelo.

Desde la raíz del repositorio, con un destino nuevo:

```bash
mkdir -p private/natural-latam-v4-treatment-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/natural-latam-v4/final-treatment-last-step004096/step-000004096-b39d6d921c7740c48a289e62d715654e \
  private/natural-latam-v4-treatment-recovery/checkpoints/step-000004096-b39d6d921c7740c48a289e62d715654e
```

La instrucción recompone LAST y no arranca entrenamiento. No se ejecutó en esta
entrega. El BEST nativo completo se conserva en la corrida privada original;
sus hashes y copias exactas de estado y manifiesto figuran en el
[recibo del paquete](natural-latam-v4-treatment-final-package.json).
Una recuperación de la corrida completa también requiere ese directorio BEST
nativo, porque el estado lo referencia. El artefacto de inferencia BEST no
contiene su optimizador y no sustituye al checkpoint nativo. Se conservan el
corpus con SHA256 `cf52daaaa0bfbd6dd539977ebaabb1b4b265d0ad192d9d6cea6c49e18da5515f`,
el código identificado y el diseño congelado. El límite `max_steps=4096` ya fue
alcanzado; restaurarlo no reinicia el contador.

El [recibo principal](natural-latam-v4-treatment-final-delivery.json) conserva
los archivos, hashes, procedencia y límites de esta entrega. Su preparación local
no acredita aún la subida a GitHub. La revisión es de IA del proyecto.
No se importó Torch, no se entrenó, no se ejecutaron pruebas ni inferencia, no se
leyeron evaluaciones reservadas ni cachés VALIDATION/TEST y no se tocó la corrida
CONTROL. Se decodificaron únicamente filas TRAIN para comprobar las cohortes.

La selección y su pérdida no demuestran conversación aceptada. El artefacto
conserva `productionAccepted=false`; la generación y revisión de respuestas se
registran por separado. La exportación de inferencia LAST también es una entrega
separada.

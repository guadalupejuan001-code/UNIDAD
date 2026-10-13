# Cierre nativo e inferencia: copia de argumentos v2

Ambos brazos cerraron **256 actualizaciones nativas** desde los mismos pesos
propios E202, con configuración idéntica y límite de 600 segundos. Tratamiento
usó el corpus nuevo de nombres y objetivos; control volvió a entrenar el corpus
v1. Se conserva el BEST seleccionado por validación de cada corrida, ambos del
paso 256, mediante copia exacta de los cuatro archivos de inferencia. Esta
entrega no promovió modelos ni ejecutó generación, entrenamiento o señales.

| Brazo | Modelo propio | Tiempo nativo | NLL interna | Targets TRAIN presentados, incluido EOS |
| --- | --- | ---: | ---: | ---: |
| Tratamiento v2 | `e70ea28e8d5dfb07…` | 157,476 s | 0,322699 | 39.764 |
| Control v1 | `734c92f9440d2d03…` | 131,484 s | 0,023745 | 36.385 |

Las NLL proceden de corpus de validación diferentes y no comparan directamente
la calidad de las respuestas. La copia de archivos, una NLL baja y la validez
JSON no demuestran argumentos autorizados ni trabajo terminado. La evaluación
real de los modelos sobre tareas queda en los registros separados del hilo
principal; esta entrega tampoco acredita competencia conversacional en español.

Los modelos completos son
[tratamiento](../../models/learning-repair-20261013/autonomy/action-copy-v2-treatment-final-best256/manifest.json)
y [control](../../models/learning-repair-20261013/autonomy/action-copy-v2-control-final-best256/manifest.json).
El [recibo conjunto](autonomy-action-copy-v2-final-delivery.json) fija sus IDs
completos, SHA de pesos/manifiestos, cierre nativo, linaje y exposición.
Las exportaciones exactas de
[tratamiento](autonomy-action-copy-v2-treatment-final-inference-export.json) y
[control](autonomy-action-copy-v2-control-final-inference-export.json) identifican
los cuatro archivos cargables de cada artefacto. No se editó ninguno de los
originales privados ni los corpus de entrenamiento.

En cada brazo se presentaron 512 ventanas TRAIN: dos épocas completas de 192
ventanas y 128 ventanas de la época siguiente. Tratamiento presentó 551.915
posiciones de entrada sin padding y 39.764 targets; control, 519.866 y 36.385.
Cada uno incluyó 512 EOS supervisados. Los recibos de exposición de
[tratamiento](autonomy-action-copy-v2-treatment-final-best-target-exposure.json)
y [control](autonomy-action-copy-v2-control-final-best-target-exposure.json)
leen las etiquetas nativas y el cursor del sampler; no interpretan pesos, Adam,
ventanas de VALIDATION ni TEST. Estas cantidades no representan conocimiento
aprendido ni palabras únicas.

Los checkpoints nativos BEST están partidos en archivos de como máximo 32 MiB.
En cada brazo LAST conserva su `state.json` y `manifest.json` propios, con
referencias explícitas a los pesos, Adam/RNG y tokenizador byte-idénticos de
BEST. Se restauraron realmente los cinco archivos nativos de BEST y LAST en
directorios privados temporales, se comprobaron sus SHA completos y se
destruyeron los directorios de comprobación. Los recibos de recuperación son
[tratamiento](autonomy-action-copy-v2-treatment-final-recovery-verification.json)
y [control](autonomy-action-copy-v2-control-final-recovery-verification.json).

Para materializar el BEST de tratamiento, desde la raíz y con una carpeta de
salida nueva:

```sh
mkdir -p private/action-copy-v2-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/autonomy-action-copy-v2/treatment-final/BEST/step-000000256-e61ac7faaac24747b0257c0e8982c643 \
  private/action-copy-v2-recovery/checkpoints/step-000000256-e61ac7faaac24747b0257c0e8982c643
```

Para reconstruir su LAST con metadatos propios:

```sh
python3 docs/learning-repair-20261013/restore_action_native_reference.py \
  checkpoints/learning-repair-20261013/autonomy-action-copy-v2/treatment-final/LAST-reference/step-000000256-ab2bcad8cf884b508f20e0ab700a2901 \
  private/action-copy-v2-recovery/checkpoints/step-000000256-ab2bcad8cf884b508f20e0ab700a2901
```

Las rutas del control figuran completas en el recibo conjunto. La carpeta de
partes o referencias no se carga directamente con `ConversationEngine`; los
artefactos de inferencia completos enlazados arriba sí conservan su contrato.
La exportación usó biblioteca estándar, un CPU y nice 19 con límite de 120
segundos por invocación; tardó 1,214 s y 1,242 s. La preparación local no equivale
a una subida a GitHub comprobada.

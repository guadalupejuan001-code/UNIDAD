# Cierre de natural-action-retention

La etapa terminó en **1024 actualizaciones del optimizador**, tras
**638.604964 segundos**. El estado nativo conserva
`status=stopped`, motivo `signal` y motivo cooperativo `absolute_step_boundary`.
El umbral configurado era 1800 segundos y el tiempo observado
quedó por debajo. Es una parada cooperativa: validación, guardado y exportación
pueden sobrepasar el umbral en otra ejecución; no se presenta como límite duro.

Objetivo: **assistant_last_reply, última respuesta y EOS supervisados, reducción sequence_mean**. BEST y LAST son checkpoints distintos del paso
1024; se preservan sus estados y manifiestos propios. Comparten los
pesos, tokenizador y Adam/RNG, cuyos bytes se reutilizan mediante referencias.
La NLL seleccionada es **1.722448436666094**. La copia y esta pérdida
no acreditan conversación aceptada ni nuevas capacidades.

El [artefacto completo de inferencia BEST](../../models/learning-repair-20261013/wide-latam-v3/natural-action-retention-final-best1024/)
conserva exactamente sus cuatro archivos originales; puede cargarse directamente
con el motor. El [recibo de exportación](wide-latam-v3-natural-action-retention-final-inference-export.json)
identifica modelo, configuración, tokenizador, pesos y hashes. La copia no ejecutó
inferencia ni cambió la aceptación.

## Exposición ejecutada

El [ledger exacto](wide-latam-v3-natural-action-retention-final-target-exposure.json) contiene
índice, etiquetas TRAIN, orden/RNG y multiplicidad por ventana comprobados.

| Posiciones procesadas por BEST y LAST | Total |
| --- | ---: |
| Ventanas únicas consumidas | 2,048 |
| Tokens de entrada sin padding | 569,326 |
| Posiciones supervisadas, incluido EOS | 265,687 |
| Posiciones EOS | 2,048 |
| Posiciones supervisadas sin EOS | 263,639 |

La corrida terminó durante su primera época. De 10,551 ventanas TRAIN,
**8,503 quedaron sin consumir**. El inventario completo no se
cuenta como ejecutado. Se usa lote lógico dos, microbatch máximo uno y semilla
20261013. Los números excluyen aprendizaje heredado, padding y ponderación de la
pérdida; no miden palabras, fuentes o conocimiento adquirido.

## Recuperación de BEST y LAST

[LAST](../../checkpoints/learning-repair-20261013/wide-latam-v3/natural-action-retention-final/LAST/step-000001024-c94984fb51974e4baea668710c5a552d/) recompone **189,439,653 bytes** en nueve
partes de hasta 32 MiB. [BEST-reference](../../checkpoints/learning-repair-20261013/wide-latam-v3/natural-action-retention-final/BEST-reference/step-000001024-7e22012058dc4945b4dfcd4578581b5b/) conserva los archivos
distintos y referencia las partes idénticas de LAST. Se verificaron los cinco
flujos completos de reconstrucción contra el BEST nativo original. El directorio
de referencia requiere recomposición antes de cargarse como checkpoint.

Desde la raíz del repositorio, con destinos nuevos:

```bash
mkdir -p private/natural-action-retention-final-recovery/checkpoints
python3 examples/package_native_checkpoint.py restore \
  checkpoints/learning-repair-20261013/wide-latam-v3/natural-action-retention-final/LAST/step-000001024-c94984fb51974e4baea668710c5a552d \
  private/natural-action-retention-final-recovery/checkpoints/step-000001024-c94984fb51974e4baea668710c5a552d
python3 docs/learning-repair-20261013/restore_action_native_reference.py \
  checkpoints/learning-repair-20261013/wide-latam-v3/natural-action-retention-final/BEST-reference/step-000001024-7e22012058dc4945b4dfcd4578581b5b \
  private/natural-action-retention-final-recovery/checkpoints/step-000001024-7e22012058dc4945b4dfcd4578581b5b
```

El helper de referencias existente conserva el nombre de su primera aplicación
ACTION y recompone este mismo formato nativo. Los comandos no se ejecutaron en
esta entrega y no arrancan entrenamiento. La recuperación exige además el corpus
SHA256 `808e1bdd98f1447fea82f83e3d936e8724f86dc23c124d8b8e3cc9504937b7a7`, diseño y código congelados. Conserva pasos y
tiempo ya consumidos. El checkpoint inicial del paso cero no está en la corrida;
no se inventó su optimizador/RNG.

El [recibo principal](wide-latam-v3-natural-action-retention-final-delivery.json) conserva los
textos originales de los metadatos, hashes, procedencia y límites. La preparación
usó biblioteca estándar, una CPU, prioridad nice19, 512 MiB de espacio de
direcciones y hasta 180 segundos por operación. No leyó corpus, cachés
VALIDATION/TEST o evaluaciones reservadas; no cargó Torch, modelos o pickle, no
entrenó, ejecutó pruebas o inferencia, restauró estados ni modificó fuentes,
supervisor o control V2 activo. La revisión es de IA del proyecto. La subida a
GitHub se verifica por separado; se conserva `conversationAccepted=false`.

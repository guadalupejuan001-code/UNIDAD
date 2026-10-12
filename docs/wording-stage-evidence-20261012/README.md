# Evidencia de la etapa de redacción variada

El [informe principal](../unidad-wording-stage-20261012.md) distingue desarrollo,
prueba nueva y aceptación. El modelo no se promociona. Las revisiones son de
una IA del proyecto y conservan razones, hashes y turnos reales.

Los pesos, corpus completos y checkpoints nativos se entregan en
`Unidad-Prueba-Propia.zip`. El código y la evidencia textual permanecen aquí.
`diagnosis.json.gz` y `training-events.jsonl.gz` son compresión sin pérdida;
`evidence-index.json` conserva también hashes y tamaños descomprimidos.

Con el paquete de Unidad instalado, desde la raíz del repositorio se pueden
reproducir los resultados sin reentrenar:

```bash
python docs/wording-stage-evidence-20261012/evaluate.py \
  --model /ruta/restaurada/models/varied-s2-last \
  --cases docs/wording-stage-evidence-20261012/known-dev.json \
  --output /ruta/nueva/repeticion.json
```

Cambiar `known-dev.json` por `fresh-transfer.json` repite la comparación nueva
publicada. Cambiar la carpeta del modelo por `conversation-reference-c7` carga
la referencia. La salida debe escribirse fuera de esta evidencia archivada.
El conjunto nuevo ya está expuesto y no debe reutilizarse como test sin exposición.

Los demás scripts son el código histórico ejecutado en la sesión y conservan
sus rutas originales: carpetas hermanas `unidad-principios`, `unidad-fix` y
`UNIDAD`. No son lanzadores para el árbol restaurado. El presupuesto de la etapa
se cerró en `stage-closed.json`; copiar un checkpoint no inicia otro entrenamiento.

La subcarpeta `beam-executed-source/` conserva el experimento descartado. Ese
código no está activo en `src/`; `probe_beam.py` requiere expresamente esa
versión experimental para reproducirse. No modificar el motor habitual para
probar los pesos entregados.

`candidate-selection.json` se escribió antes de las generaciones nuevas y fija
la identidad seleccionada. `candidate-cli-verification.json` y la transcripción
comprueban la carga real por la CLI y cuatro respuestas iguales a la evaluación.

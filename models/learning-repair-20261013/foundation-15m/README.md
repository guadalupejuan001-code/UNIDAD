# Base propia de español: exportaciones experimentales

`intermediate-step004096/` conserva la primera exportación de inferencia de la
pausa de la etapa causal `foundation-15m`. Es un Transformer propio de
15.179.520 parámetros iniciado desde pesos aleatorios, con BPE de 2.048 tokens
ajustado sólo con TRAIN y contexto de 512. No usa pesos preentrenados ajenos.

La exportación contiene exactamente cuatro archivos: configuración,
tokenizador, pesos y manifiesto, con un total de **60.784.469 bytes**. Sus hashes
coincidieron antes de copiar, en la copia y después de copiar. El
[recibo de entrega](../../../docs/learning-repair-20261013/foundation-15m-intermediate-step004096-delivery.json)
conserva esos hashes y la procedencia.

Identifica el modelo
`unidad-conversation-1e46abcba0e6d5cbf25380bc98b92190ef556afbd5acd3463c92e399f32e3666`,
seleccionado en step 4.096 por NLL de validación **3,4665581824610308**.
Es una base lingüística causal intermedia; todavía no acredita conversación
competente, no reemplaza C7 y no supone que la etapa de entrenamiento haya
terminado. El manifiesto conserva `experimental=true` y
`productionAccepted=false`.

Este artefacto sirve para cargar pesos en inferencia y conserva una versión
histórica. No contiene Adam ni RNG para reanudar. Los checkpoints nativos
recuperables se distribuyen por separado, según
[checkpoint-packaging.md](../../../docs/learning-repair-20261013/checkpoint-packaging.md).
La publicación remota y las respuestas conversacionales se comprueban por
separado; no se atribuyen a esta operación de copia.
